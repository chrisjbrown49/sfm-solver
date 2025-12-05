"""
Nonlinear self-consistent eigenvalue solver.

Solves the nonlinear eigenvalue problem:
    (H_0 + g|χ|²) χ = E χ
    
using iterative self-consistent field methods.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from sft_solver.core.grid import SpectralGrid
from sft_solver.potentials.three_well import ThreeWellPotential
from sft_solver.eigensolver.spectral import SpectralOperators
from sft_solver.eigensolver.linear import LinearEigensolver


@dataclass
class ConvergenceInfo:
    """Information about solver convergence."""
    converged: bool
    iterations: int
    energy_history: List[float]
    residual_history: List[float]
    final_residual: float
    

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
        verbose: bool = False
    ) -> Tuple[float, NDArray[np.complexfloating], ConvergenceInfo]:
        """
        Solve the nonlinear eigenvalue problem self-consistently.
        
        Uses simple mixing iteration:
            χ_new = (1 - α) χ_old + α χ_step
        
        where χ_step is the ground state of H[χ_old].
        
        Args:
            k: Winding number sector.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance on energy.
            mixing: Mixing parameter α (0 < α ≤ 1).
            initial_guess: Initial wavefunction guess, or None for linear solution.
            verbose: If True, print convergence info.
            
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
        
        for iteration in range(max_iter):
            # Compute effective potential with current density
            density = np.abs(chi)**2
            V_eff = self._V_grid + self.g1 * density
            
            # Solve linear eigenvalue problem with effective potential
            chi_new = self._solve_linear_step(V_eff, k)
            
            # Mix old and new
            chi = (1 - mixing) * chi + mixing * chi_new
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
            final_residual=residual_history[-1] if residual_history else float('inf')
        )
        
        return E_new, chi, info
    
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

