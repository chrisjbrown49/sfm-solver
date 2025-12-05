"""
Self-consistent amplitude eigensolver for SFM.

This solver implements the amplitude quantization scheme where the amplitude A
emerges self-consistently from the nonlinear eigenvalue problem:

    H_eff χ = E χ
    
where:
    H_eff = -ℏ²/(2m_σR²) ∂²/∂σ² + V(σ) + g₁·A²_χ
    
and A²_χ = ∫|χ|² dσ is the integrated amplitude.

Key features:
1. The nonlinear term g₁·A² is a UNIFORM potential shift (not spatially varying)
2. No rescaling after solving - amplitude emerges from self-consistency
3. Mixing preserves amplitude between iterations
4. Convergence is checked on both A² and E

Different self-consistent branches (with different A values) correspond to
different particle masses via m = β·A²_χ.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.eigensolver.linear import LinearEigensolver


@dataclass
class AmplitudeEigenstate:
    """
    Self-consistent eigenstate with amplitude quantization.
    
    Attributes:
        wavefunction: χ(σ), the converged wavefunction (NOT normalized to 1)
        amplitude_squared: A² = ∫|χ|² dσ (determines mass: m = β·A²)
        energy: Eigenvalue E from H_eff χ = E χ
        effective_potential_shift: g₁·A² (the uniform nonlinear contribution)
        converged: Whether the iteration converged
        iterations: Number of iterations to convergence
        amplitude_history: A² values during iteration (for diagnostics)
        energy_history: E values during iteration
    """
    wavefunction: NDArray
    amplitude_squared: float
    energy: float
    effective_potential_shift: float
    converged: bool
    iterations: int
    amplitude_history: List[float]
    energy_history: List[float]


class AmplitudeEigensolver:
    """
    Self-consistent eigensolver where amplitude A emerges from iteration.
    
    The iterative scheme:
    1. Start with initial guess χ⁽⁰⁾ (no predetermined amplitude)
    2. At each iteration n:
       a. Compute A²_χ⁽ⁿ⁾ = ∫|χ⁽ⁿ⁾|² dσ
       b. Build H = H₀ + g₁·A²_χ⁽ⁿ⁾  (uniform potential shift)
       c. Solve H χ = E χ
       d. NO rescaling - keep χ as obtained
       e. Mix: χ = (1-α)χ⁽ⁿ⁾ + α·χ_new
    3. Converge when |A²⁽ⁿ⁺¹⁾ - A²⁽ⁿ⁾| < tol AND |E⁽ⁿ⁺¹⁾ - E⁽ⁿ⁾| < tol
    
    The amplitude A² is both input and output of the iteration.
    Self-consistent solutions have different A values for different branches.
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g1: float = 0.1,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        R: float = 1.0
    ):
        """
        Initialize the amplitude eigensolver.
        
        Args:
            grid: SpectralGrid for subspace discretization.
            potential: Base potential V(σ).
            g1: Nonlinear coupling constant for g₁·A² term.
            m_eff: Effective mass in subspace.
            hbar: Reduced Planck constant.
            R: Subspace radius.
        """
        self.grid = grid
        self.potential = potential
        self.g1 = g1
        self.m_eff = m_eff
        self.hbar = hbar
        self.R = R
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
        
        # Linear solver for initial guess
        self.linear_solver = LinearEigensolver(grid, potential, m_eff, hbar)
    
    def solve(
        self,
        k: int = 1,
        initial_amplitude: Optional[float] = None,
        max_iter: int = 500,
        tol_A: float = 1e-8,
        tol_E: float = 1e-8,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> AmplitudeEigenstate:
        """
        Solve for self-consistent amplitude eigenstate.
        
        Args:
            k: Winding number sector.
            initial_amplitude: Starting A value, or None for linear solution.
            max_iter: Maximum iterations.
            tol_A: Tolerance for amplitude convergence.
            tol_E: Tolerance for energy convergence.
            mixing: Mixing parameter α (0 < α ≤ 1).
            verbose: Print progress.
            
        Returns:
            AmplitudeEigenstate with converged amplitude.
        """
        # Step 1: Initial guess
        if initial_amplitude is not None:
            # Create initial guess with specified amplitude
            chi = self._create_initial_guess(k, initial_amplitude)
        else:
            # Use linear solution as starting point
            E0, chi = self.linear_solver.ground_state(k)
        
        # Initial amplitude
        A_sq = self._compute_amplitude_squared(chi)
        E_old = self._compute_energy(chi, A_sq)
        
        amplitude_history = [A_sq]
        energy_history = [E_old]
        
        A_sq_old = A_sq
        converged = False
        
        if verbose:
            print("Starting amplitude self-consistent iteration")
            print("  Initial: A^2 = %.6f, E = %.6f" % (A_sq, E_old))
        
        for iteration in range(max_iter):
            # Step 2a: A² is already computed from previous iteration
            
            # Step 2b: Build H_eff = H₀ + g₁·A² (uniform shift)
            # The potential becomes V_eff(σ) = V(σ) + g₁·A²
            V_eff = self._V_grid + self.g1 * A_sq
            
            # Step 2c: Solve eigenvalue problem
            H = self.operators.build_hamiltonian_matrix(V_eff)
            H = 0.5 * (H + np.conj(H.T))  # Ensure Hermitian
            
            energies, wavefunctions = np.linalg.eigh(H)
            
            # Find state with correct winding number
            chi_new = None
            E_new = None
            for i in range(len(energies)):
                psi = wavefunctions[:, i]
                winding = self.grid.winding_number(psi)
                if abs(winding - k) < 0.5:
                    chi_new = psi
                    E_new = energies[i]
                    break
            
            if chi_new is None:
                chi_new = wavefunctions[:, 0]
                E_new = energies[0]
            
            # Step 2d: NO RESCALING - keep χ_new as obtained from solver
            # (The solver returns normalized eigenvector, but mixing will change amplitude)
            
            # Step 2e: Mix for stability
            chi = (1 - mixing) * chi + mixing * chi_new
            
            # Compute new amplitude (after mixing, NOT normalized!)
            A_sq = self._compute_amplitude_squared(chi)
            
            amplitude_history.append(A_sq)
            energy_history.append(E_new)
            
            # Step 3: Check convergence
            dA = abs(A_sq - A_sq_old)
            dE = abs(E_new - E_old)
            
            if verbose and iteration % 20 == 0:
                print("  Iter %d: A^2 = %.6f, E = %.6f, dA = %.2e, dE = %.2e" % 
                      (iteration, A_sq, E_new, dA, dE))
            
            if dA < tol_A and dE < tol_E:
                converged = True
                if verbose:
                    print("  Converged at iteration %d" % (iteration + 1))
                break
            
            A_sq_old = A_sq
            E_old = E_new
        
        # Final energy with correct amplitude
        E_final = self._compute_energy(chi, A_sq)
        
        return AmplitudeEigenstate(
            wavefunction=chi,
            amplitude_squared=A_sq,
            energy=E_final,
            effective_potential_shift=self.g1 * A_sq,
            converged=converged,
            iterations=iteration + 1,
            amplitude_history=amplitude_history,
            energy_history=energy_history
        )
    
    def find_branches(
        self,
        k: int = 1,
        n_branches: int = 5,
        A_range: Tuple[float, float] = (0.1, 50.0),
        max_iter: int = 500,
        tol_A: float = 1e-8,
        tol_E: float = 1e-8,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> List[AmplitudeEigenstate]:
        """
        Find multiple amplitude branches by trying different initial conditions.
        
        Different initial amplitudes may converge to different self-consistent
        branches, each with a distinct A value.
        
        Args:
            k: Winding number sector.
            n_branches: Number of initial conditions to try.
            A_range: Range of initial amplitudes (A_min, A_max).
            max_iter: Maximum iterations per branch.
            tol_A: Amplitude tolerance.
            tol_E: Energy tolerance.
            mixing: Mixing parameter.
            verbose: Print progress.
            
        Returns:
            List of converged AmplitudeEigenstates (may have duplicates).
        """
        A_min, A_max = A_range
        A_initials = np.linspace(A_min, A_max, n_branches)
        
        if verbose:
            print("Searching for amplitude branches in A in [%.2f, %.2f]" % (A_min, A_max))
        
        solutions = []
        
        for A_init in A_initials:
            if verbose:
                print("\n  Trying A_init = %.2f..." % A_init)
            
            sol = self.solve(
                k=k,
                initial_amplitude=A_init,
                max_iter=max_iter,
                tol_A=tol_A,
                tol_E=tol_E,
                mixing=mixing,
                verbose=False
            )
            
            if sol.converged:
                solutions.append(sol)
                if verbose:
                    print("    Converged to A^2 = %.6f, E = %.6f" % 
                          (sol.amplitude_squared, sol.energy))
            else:
                if verbose:
                    print("    Did not converge (A^2 = %.6f after %d iters)" % 
                          (sol.amplitude_squared, sol.iterations))
        
        # Remove duplicates (same A² within tolerance)
        unique = self._remove_duplicate_branches(solutions)
        
        if verbose:
            print("\nFound %d unique branches:" % len(unique))
            for i, sol in enumerate(unique):
                print("  Branch %d: A^2 = %.6f, E = %.6f" % (i+1, sol.amplitude_squared, sol.energy))
        
        return unique
    
    def _compute_amplitude_squared(self, chi: NDArray) -> float:
        """Compute A² = ∫|χ|² dσ."""
        return np.real(self.grid.integrate(np.abs(chi)**2))
    
    def _compute_energy(self, chi: NDArray, A_sq: float) -> float:
        """
        Compute eigenvalue E = <χ|H_eff|χ> / <χ|χ>.
        
        Note: We compute the expectation value, not the eigenvalue from the solver,
        to account for the current amplitude.
        """
        # Effective potential with current amplitude
        V_eff = self._V_grid + self.g1 * A_sq
        
        # Kinetic energy
        T = self.operators.kinetic_expectation(chi)
        
        # Potential energy
        V = np.real(self.grid.integrate(np.conj(chi) * V_eff * chi))
        
        # Normalize by <χ|χ>
        norm_sq = np.real(self.grid.integrate(np.abs(chi)**2))
        
        if norm_sq > 1e-15:
            return (T + V) / norm_sq
        return T + V
    
    def _create_initial_guess(self, k: int, amplitude: float) -> NDArray:
        """Create initial guess with specified amplitude."""
        # Start with linear solution shape
        E0, chi = self.linear_solver.ground_state(k)
        
        # Scale to desired amplitude: A² = ∫|χ|² dσ = amplitude²
        current_A_sq = self._compute_amplitude_squared(chi)
        if current_A_sq > 1e-15:
            chi = chi * np.sqrt(amplitude**2 / current_A_sq)
        
        return chi
    
    def _remove_duplicate_branches(
        self, 
        solutions: List[AmplitudeEigenstate],
        rel_tol: float = 0.01
    ) -> List[AmplitudeEigenstate]:
        """Remove duplicate branches (same A² within tolerance)."""
        if not solutions:
            return []
        
        unique = [solutions[0]]
        
        for sol in solutions[1:]:
            is_duplicate = False
            for u in unique:
                if abs(sol.amplitude_squared - u.amplitude_squared) / max(sol.amplitude_squared, 1e-10) < rel_tol:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(sol)
        
        return sorted(unique, key=lambda s: s.amplitude_squared)
    
    def compute_mass(self, sol: AmplitudeEigenstate, beta: float = 1.0) -> float:
        """
        Compute particle mass from amplitude.
        
        m = β · A²_χ
        
        Args:
            sol: Converged eigenstate.
            beta: Mass coupling constant.
            
        Returns:
            Particle mass.
        """
        return beta * sol.amplitude_squared
    
    def compute_mass_ratios(
        self, 
        solutions: List[AmplitudeEigenstate]
    ) -> Dict[str, float]:
        """
        Compute mass ratios from multiple branches.
        
        Args:
            solutions: List of converged eigenstates (different branches).
            
        Returns:
            Dictionary with amplitudes and mass ratios.
        """
        if len(solutions) < 2:
            return {'error': 'Need at least 2 branches'}
        
        # Sort by amplitude
        sorted_sols = sorted(solutions, key=lambda s: s.amplitude_squared)
        
        result = {}
        A_ref = sorted_sols[0].amplitude_squared
        
        for i, sol in enumerate(sorted_sols):
            result[f'A_squared_{i}'] = sol.amplitude_squared
            result[f'E_{i}'] = sol.energy
            if A_ref > 0:
                result[f'mass_ratio_{i}_0'] = sol.amplitude_squared / A_ref
        
        return result


def find_amplitude_quantization_subspace(
    g1: float = 0.1,
    V0: float = 1.0,
    V1: float = 0.1,
    A_range: Tuple[float, float] = (0.1, 50.0),
    n_trials: int = 20,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Search for amplitude-quantized branches in subspace-only problem.
    
    This tests whether the nonlinear term g₁·A² produces self-consistent
    solutions with different amplitudes.
    
    Args:
        g1: Nonlinear coupling.
        V0: Primary well depth.
        V1: Secondary modulation.
        A_range: Range of initial amplitudes to try.
        n_trials: Number of initial conditions.
        verbose: Print progress.
        
    Returns:
        Dictionary with branches and mass ratios.
    """
    grid = SpectralGrid(N=128)
    potential = ThreeWellPotential(V0=V0, V1=V1)
    
    solver = AmplitudeEigensolver(grid, potential, g1=g1)
    
    if verbose:
        print("Searching for amplitude-quantized branches")
        print("  g1 = %.4f, V0 = %.2f, V1 = %.2f" % (g1, V0, V1))
        print("  A range: [%.2f, %.2f]" % A_range)
        print()
    
    branches = solver.find_branches(
        k=1,
        n_branches=n_trials,
        A_range=A_range,
        verbose=verbose
    )
    
    result = {
        'n_branches': len(branches),
        'branches': branches,
    }
    
    if len(branches) >= 2:
        ratios = solver.compute_mass_ratios(branches)
        result['mass_ratios'] = ratios
        
        if verbose:
            print("\nMass ratios from amplitude quantization:")
            for key, val in sorted(ratios.items()):
                if 'mass_ratio' in key:
                    print("  %s: %.4f" % (key, val))
    
    return result

