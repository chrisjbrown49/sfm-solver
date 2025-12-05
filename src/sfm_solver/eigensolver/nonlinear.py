"""
Nonlinear self-consistent eigenvalue solver.

Solves the nonlinear eigenvalue problem:
    (H_0 + g|χ|²) χ = E χ
    
using iterative self-consistent field methods.

Includes:
- Simple linear mixing: χ_new = (1-α)χ_old + α χ_step
- DIIS (Direct Inversion in the Iterative Subspace) for accelerated convergence
- Anderson mixing as an alternative acceleration scheme
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import lstsq
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.eigensolver.linear import LinearEigensolver


@dataclass
class ConvergenceInfo:
    """Information about solver convergence."""
    converged: bool
    iterations: int
    energy_history: List[float]
    residual_history: List[float]
    final_residual: float
    mixing_method: str = "linear"


class DIISMixer:
    """
    Direct Inversion in the Iterative Subspace (DIIS) mixing.
    
    DIIS accelerates convergence by extrapolating from recent iterates
    to find a linear combination that minimizes the residual norm.
    
    The method stores recent iterates {x_i} and residuals {r_i}, then
    finds coefficients {c_i} such that sum(c_i) = 1 and ||sum(c_i r_i)||
    is minimized. The next iterate is then x_new = sum(c_i x_i).
    
    Reference: Pulay, Chem. Phys. Lett. 73, 393 (1980)
    
    Attributes:
        max_vectors: Maximum number of vectors to store.
        vectors: List of recent iterates.
        residuals: List of corresponding residuals.
    """
    
    def __init__(self, max_vectors: int = 6):
        """
        Initialize DIIS mixer.
        
        Args:
            max_vectors: Maximum number of vectors to keep in history.
        """
        self.max_vectors = max_vectors
        self.vectors: List[NDArray] = []
        self.residuals: List[NDArray] = []
    
    def reset(self):
        """Clear the stored history."""
        self.vectors = []
        self.residuals = []
    
    def add(self, x: NDArray, r: NDArray):
        """
        Add a new iterate and residual to history.
        
        Args:
            x: Current iterate (wavefunction).
            r: Residual (H·x - E·x or similar).
        """
        self.vectors.append(x.copy())
        self.residuals.append(r.copy())
        
        # Keep only most recent vectors
        if len(self.vectors) > self.max_vectors:
            self.vectors.pop(0)
            self.residuals.pop(0)
    
    def extrapolate(self) -> Optional[NDArray]:
        """
        Compute DIIS extrapolation from stored history.
        
        Solves the constrained least squares problem:
            min ||sum(c_i r_i)||²  subject to sum(c_i) = 1
        
        Returns:
            Extrapolated iterate, or None if insufficient history.
        """
        n = len(self.vectors)
        if n < 2:
            return None
        
        # Build the DIIS B matrix: B_ij = <r_i | r_j>
        B = np.zeros((n + 1, n + 1), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                B[i, j] = np.vdot(self.residuals[i], self.residuals[j])
        
        # Add Lagrange multiplier row/column for constraint sum(c) = 1
        B[n, :n] = 1.0
        B[:n, n] = 1.0
        B[n, n] = 0.0
        
        # Right-hand side: [0, 0, ..., 0, 1]
        rhs = np.zeros(n + 1, dtype=complex)
        rhs[n] = 1.0
        
        # Solve the linear system
        try:
            # Use least squares for numerical stability
            c, residual, rank, s = lstsq(B, rhs, lapack_driver='gelsy')
            coeffs = c[:n]
            
            # Check for reasonable coefficients
            if np.any(np.abs(coeffs) > 10):
                # Coefficients too large, DIIS unstable
                return None
            
            # Compute extrapolated iterate
            x_new = np.zeros_like(self.vectors[0])
            for i in range(n):
                x_new += coeffs[i] * self.vectors[i]
            
            return x_new
            
        except np.linalg.LinAlgError:
            return None


class AndersonMixer:
    """
    Anderson mixing (also known as Anderson acceleration).
    
    An alternative to DIIS that uses the change in iterates and residuals
    to compute an optimal mixing. Often more stable than DIIS for
    difficult problems.
    
    Reference: Anderson, J. ACM 12, 547 (1965)
    
    Attributes:
        max_vectors: Maximum history length.
        beta: Base mixing parameter.
    """
    
    def __init__(self, max_vectors: int = 6, beta: float = 0.3):
        """
        Initialize Anderson mixer.
        
        Args:
            max_vectors: Maximum history length.
            beta: Base linear mixing parameter.
        """
        self.max_vectors = max_vectors
        self.beta = beta
        
        self.x_history: List[NDArray] = []
        self.f_history: List[NDArray] = []  # f = x_new - x_old (update step)
    
    def reset(self):
        """Clear history."""
        self.x_history = []
        self.f_history = []
    
    def mix(self, x_old: NDArray, x_step: NDArray) -> NDArray:
        """
        Compute Anderson-mixed iterate.
        
        Args:
            x_old: Previous iterate.
            x_step: Result of applying iteration operator to x_old.
            
        Returns:
            Mixed iterate.
        """
        f = x_step - x_old  # The "residual" in Anderson's formulation
        
        # Store history
        self.x_history.append(x_old.copy())
        self.f_history.append(f.copy())
        
        # Keep limited history
        if len(self.x_history) > self.max_vectors:
            self.x_history.pop(0)
            self.f_history.pop(0)
        
        n = len(self.x_history)
        
        if n < 2:
            # Simple linear mixing
            return x_old + self.beta * f
        
        # Build the matrix of residual differences
        # df_j = f_n - f_j for j = 0, ..., n-2
        m = n - 1
        df = np.column_stack([
            self.f_history[-1] - self.f_history[j] 
            for j in range(m)
        ])
        
        # Solve least squares: min ||f_n - sum(gamma_j df_j)||²
        try:
            gamma, residual, rank, s = lstsq(
                df, self.f_history[-1], lapack_driver='gelsy'
            )
            
            # Compute Anderson update
            # x_new = (1-beta)(x_n + sum(gamma_j (x_{n-j} - x_n))) + beta*(...)
            x_new = x_old + self.beta * f
            
            for j in range(m):
                dx = self.x_history[-1] - self.x_history[j]
                x_new -= gamma[j] * (dx + self.beta * (self.f_history[-1] - self.f_history[j]))
            
            return x_new
            
        except np.linalg.LinAlgError:
            # Fall back to simple mixing
            return x_old + self.beta * f
    

class NonlinearEigensolver:
    """
    Nonlinear self-consistent field solver.
    
    Solves the Gross-Pitaevskii-like equation:
        (-ℏ²/2m d²/dσ² + V(σ) + g₁|χ|²) χ = E χ
    
    using iterative methods with mixing for stability.
    
    The nonlinear term g₁|χ|² provides self-interaction that
    is important for mass generation and confinement.
    
    Attributes:
        grid: SpectralGrid instance.
        potential: Base potential function.
        g1: Nonlinear coupling constant.
        operators: SpectralOperators for FFT-based calculations.
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g1: float = 0.1,
        m_eff: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize the nonlinear eigensolver.
        
        Args:
            grid: SpectralGrid defining the discretization.
            potential: Base potential function.
            g1: Nonlinear |χ|² coupling constant.
            m_eff: Effective mass parameter.
            hbar: Reduced Planck constant.
        """
        self.grid = grid
        self.potential = potential
        self.g1 = g1
        self.m_eff = m_eff
        self.hbar = hbar
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
        
        # Linear solver for initial guess
        self.linear_solver = LinearEigensolver(grid, potential, m_eff, hbar)
    
    def solve(
        self,
        k: int = 1,
        max_iter: int = 100,
        tol: float = 1e-8,
        mixing: float = 0.3,
        initial_guess: Optional[NDArray] = None,
        verbose: bool = False,
        mixing_method: str = "linear"
    ) -> Tuple[float, NDArray[np.complexfloating], ConvergenceInfo]:
        """
        Solve the nonlinear eigenvalue problem self-consistently.
        
        Supports multiple mixing methods for convergence:
        - "linear": Simple linear mixing χ_new = (1-α)χ_old + α χ_step
        - "diis": Direct Inversion in the Iterative Subspace (faster convergence)
        - "anderson": Anderson mixing (often more stable than DIIS)
        
        Args:
            k: Winding number sector.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance on energy.
            mixing: Mixing parameter α (0 < α ≤ 1) for linear/Anderson methods.
            initial_guess: Initial wavefunction guess, or None for linear solution.
            verbose: If True, print convergence info.
            mixing_method: "linear", "diis", or "anderson".
            
        Returns:
            Tuple of (energy, wavefunction, convergence_info).
        """
        # Get initial guess from linear solver if not provided
        if initial_guess is None:
            E0, chi = self.linear_solver.ground_state(k)
        else:
            chi = self.grid.normalize(initial_guess)
            E0 = self._compute_energy(chi)
        
        energy_history = [E0]
        residual_history = []
        
        E_old = E0
        converged = False
        
        # Initialize mixer based on method
        if mixing_method == "diis":
            diis = DIISMixer(max_vectors=6)
        elif mixing_method == "anderson":
            anderson = AndersonMixer(max_vectors=6, beta=mixing)
        
        for iteration in range(max_iter):
            # Compute effective potential with current density
            density = np.abs(chi)**2
            V_eff = self._V_grid + self.g1 * density
            
            # Solve linear eigenvalue problem with effective potential
            chi_step = self._solve_linear_step(V_eff, k)
            
            # Apply mixing
            if mixing_method == "diis":
                # Compute residual for DIIS
                E_step = self._compute_energy(chi_step)
                residual_vec = self._compute_residual_vector(chi, E_step)
                diis.add(chi, residual_vec)
                
                # Try DIIS extrapolation
                chi_diis = diis.extrapolate()
                if chi_diis is not None:
                    chi = chi_diis
                else:
                    # Fall back to linear mixing
                    chi = (1 - mixing) * chi + mixing * chi_step
                    
            elif mixing_method == "anderson":
                chi = anderson.mix(chi, chi_step)
                
            else:  # linear mixing
                chi = (1 - mixing) * chi + mixing * chi_step
            
            chi = self.grid.normalize(chi)
            
            # Compute new energy
            E_new = self._compute_energy(chi)
            
            # Check convergence
            dE = abs(E_new - E_old)
            residual = self._compute_residual(chi, E_new)
            
            energy_history.append(E_new)
            residual_history.append(residual)
            
            if verbose:
                print(f"Iter {iteration+1}: E = {E_new:.8f}, dE = {dE:.2e}, res = {residual:.2e}")
            
            if dE < tol and residual < tol * 10:
                converged = True
                break
            
            E_old = E_new
        
        info = ConvergenceInfo(
            converged=converged,
            iterations=iteration + 1,
            energy_history=energy_history,
            residual_history=residual_history,
            final_residual=residual_history[-1] if residual_history else float('inf'),
            mixing_method=mixing_method
        )
        
        return E_new, chi, info
    
    def _compute_residual_vector(self, chi: NDArray, E: float) -> NDArray:
        """
        Compute residual vector H_eff χ - E χ.
        
        This is the full vector, not just the norm (needed for DIIS).
        """
        density = np.abs(chi)**2
        V_eff = self._V_grid + self.g1 * density
        H_chi = self.operators.apply_hamiltonian(chi, V_eff)
        return H_chi - E * chi
    
    def solve_excited_states(
        self,
        k: int = 1,
        n_states: int = 3,
        max_iter: int = 100,
        tol: float = 1e-8,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> Tuple[List[float], List[NDArray], List[ConvergenceInfo]]:
        """
        Solve for multiple excited states.
        
        Uses Gram-Schmidt orthogonalization to ensure orthogonality
        to previously found states.
        
        Args:
            k: Winding number sector.
            n_states: Number of states to find.
            max_iter: Maximum iterations per state.
            tol: Convergence tolerance.
            mixing: Mixing parameter.
            verbose: Print progress.
            
        Returns:
            Tuple of (energies, wavefunctions, convergence_infos).
        """
        energies = []
        wavefunctions = []
        infos = []
        
        # Get linear solutions as initial guesses
        E_lin, psi_lin = self.linear_solver.solve_with_winding(k, n_states=n_states)
        
        for n in range(n_states):
            if verbose:
                print(f"\n--- Solving for state {n} ---")
            
            # Use linear solution as initial guess
            initial = psi_lin[n] if n < len(psi_lin) else None
            
            # Solve with orthogonalization constraint
            E, chi, info = self._solve_with_orthogonality(
                k, wavefunctions, initial, max_iter, tol, mixing, verbose
            )
            
            energies.append(E)
            wavefunctions.append(chi)
            infos.append(info)
        
        return energies, wavefunctions, infos
    
    def _solve_with_orthogonality(
        self,
        k: int,
        lower_states: List[NDArray],
        initial_guess: Optional[NDArray],
        max_iter: int,
        tol: float,
        mixing: float,
        verbose: bool
    ) -> Tuple[float, NDArray, ConvergenceInfo]:
        """
        Solve for a state orthogonal to given lower states.
        """
        # Initial guess
        if initial_guess is not None:
            chi = self._orthogonalize(initial_guess, lower_states)
            chi = self.grid.normalize(chi)
        else:
            chi = self._random_orthogonal_guess(k, lower_states)
        
        E_old = self._compute_energy(chi)
        energy_history = [E_old]
        residual_history = []
        converged = False
        
        for iteration in range(max_iter):
            # Compute effective potential
            density = np.abs(chi)**2
            V_eff = self._V_grid + self.g1 * density
            
            # Solve linear problem
            chi_new = self._solve_linear_step(V_eff, k)
            
            # Orthogonalize to lower states
            chi_new = self._orthogonalize(chi_new, lower_states)
            
            # Mix
            chi = (1 - mixing) * chi + mixing * chi_new
            chi = self._orthogonalize(chi, lower_states)
            chi = self.grid.normalize(chi)
            
            # Energy and convergence
            E_new = self._compute_energy(chi)
            dE = abs(E_new - E_old)
            residual = self._compute_residual(chi, E_new)
            
            energy_history.append(E_new)
            residual_history.append(residual)
            
            if verbose:
                print(f"Iter {iteration+1}: E = {E_new:.8f}, dE = {dE:.2e}")
            
            if dE < tol and residual < tol * 10:
                converged = True
                break
            
            E_old = E_new
        
        info = ConvergenceInfo(
            converged=converged,
            iterations=iteration + 1,
            energy_history=energy_history,
            residual_history=residual_history,
            final_residual=residual_history[-1] if residual_history else float('inf')
        )
        
        return E_new, chi, info
    
    def _solve_linear_step(
        self, 
        V_eff: NDArray, 
        k: int
    ) -> NDArray[np.complexfloating]:
        """
        Solve linear eigenvalue problem with effective potential.
        """
        # Build Hamiltonian
        H = self.operators.build_hamiltonian_matrix(V_eff)
        H = 0.5 * (H + np.conj(H.T))  # Ensure Hermitian
        
        # Get ground state
        energies, wavefunctions = np.linalg.eigh(H)
        
        # Find state with correct winding number
        for i in range(len(energies)):
            psi = wavefunctions[:, i]
            winding = self.grid.winding_number(psi)
            if abs(winding - k) < 0.5:
                return self.grid.normalize(psi)
        
        # Fallback: return lowest state
        return self.grid.normalize(wavefunctions[:, 0])
    
    def _compute_energy(self, chi: NDArray) -> float:
        """
        Compute total energy including nonlinear term.
        
        E = <T> + <V> + (g₁/2)<|χ|⁴>
        """
        T = self.operators.kinetic_expectation(chi)
        V = self.operators.potential_expectation(chi, self._V_grid)
        
        # Nonlinear energy: (g₁/2) ∫|χ|⁴ dσ
        density = np.abs(chi)**2
        E_nl = (self.g1 / 2) * np.real(self.grid.integrate(density**2))
        
        return T + V + E_nl
    
    def _compute_residual(self, chi: NDArray, E: float) -> float:
        """
        Compute residual ||H_eff χ - E χ||.
        """
        density = np.abs(chi)**2
        V_eff = self._V_grid + self.g1 * density
        H_chi = self.operators.apply_hamiltonian(chi, V_eff)
        residual = H_chi - E * chi
        return self.grid.norm(residual)
    
    def _orthogonalize(
        self, 
        psi: NDArray, 
        lower_states: List[NDArray]
    ) -> NDArray:
        """
        Orthogonalize psi against lower states using Gram-Schmidt.
        """
        result = psi.copy()
        for phi in lower_states:
            overlap = self.grid.inner_product(phi, result)
            result = result - overlap * phi
        
        norm = self.grid.norm(result)
        if norm > 1e-10:
            return result
        else:
            # State is in span of lower states, return random orthogonal
            return self._random_orthogonal_guess(1, lower_states)
    
    def _random_orthogonal_guess(
        self, 
        k: int, 
        lower_states: List[NDArray]
    ) -> NDArray:
        """
        Generate a random initial guess orthogonal to lower states.
        """
        # Start with a localized winding mode
        chi = self.grid.create_localized_mode(k, center=0.0, width=0.5)
        chi = self._orthogonalize(chi, lower_states)
        return self.grid.normalize(chi)
    
    def amplitude_squared(self, chi: NDArray) -> float:
        """
        Compute integrated amplitude squared A²_χ.
        """
        return np.real(self.grid.integrate(np.abs(chi)**2))
    
    def mass(self, chi: NDArray, beta: float) -> float:
        """
        Compute particle mass from wavefunction.
        
        m = β A²_χ
        """
        return beta * self.amplitude_squared(chi)
    
    def energy_components(self, chi: NDArray) -> Dict[str, float]:
        """
        Break down energy into components.
        """
        T = self.operators.kinetic_expectation(chi)
        V = self.operators.potential_expectation(chi, self._V_grid)
        density = np.abs(chi)**2
        E_nl = (self.g1 / 2) * np.real(self.grid.integrate(density**2))
        
        return {
            'kinetic': T,
            'potential': V,
            'nonlinear': E_nl,
            'total': T + V + E_nl
        }

