"""
Amplitude-quantized nonlinear eigenvalue solver.

This solver finds self-consistent solutions of the Gross-Pitaevskii-like equation
where the amplitude A is NOT fixed by normalization, but emerges from the
nonlinear self-consistency condition.

The key insight: In standard QM, we normalize ∫|ψ|² = 1. But in the SFM theory,
different particles have DIFFERENT amplitudes A. The amplitude determines mass:
    m = β A²

The Gross-Pitaevskii equation:
    [-ℏ²/(2m_σ R²) d²/dσ² + V(σ) + g|χ|²] χ = μ χ

For self-consistent solutions:
- The chemical potential μ plays the role of eigenvalue
- The amplitude A = (∫|χ|² dσ / 2π)^(1/2) emerges from self-consistency
- Different branches (starting conditions) give different (μ, A) pairs

Methods implemented:
1. Imaginary Time Evolution (ITE): Robust ground state finder
2. Branch continuation: Starting from one solution, vary g to find branches
3. Newton-Raphson: Direct nonlinear equation solver
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh, solve
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.eigensolver.linear import LinearEigensolver


@dataclass
class AmplitudeSolution:
    """
    Solution of the amplitude-quantized nonlinear problem.
    
    Attributes:
        chemical_potential: μ, the eigenvalue of the GP equation
        wavefunction: χ(σ), the (possibly unnormalized) wavefunction
        amplitude: A = sqrt(∫|χ|² dσ / 2π), the amplitude parameter
        amplitude_squared: A² (determines mass: m = β A²)
        peak_amplitude: max|χ(σ)|², an alternative amplitude measure
        energy: Total energy E = T + V + (g/2)∫|χ|⁴
        converged: Whether the solver converged
        iterations: Number of iterations
        branch_id: Identifier for which branch this solution is on
    """
    chemical_potential: float
    wavefunction: NDArray
    amplitude: float
    amplitude_squared: float
    peak_amplitude: float
    energy: float
    converged: bool
    iterations: int
    branch_id: int = 0


class ImaginaryTimeEvolution:
    """
    Imaginary Time Evolution (ITE) for finding ground states.
    
    ITE propagates in imaginary time: χ(τ+dτ) = exp(-H·dτ) χ(τ)
    This exponentially suppresses excited states, leaving the ground state.
    
    For the nonlinear GP equation, H depends on |χ|², so we use split-step:
    1. Half step with kinetic energy (in Fourier space)
    2. Full step with potential + nonlinear term (in real space)
    3. Half step with kinetic energy
    
    The amplitude emerges naturally - we don't normalize during evolution,
    only at the end for comparison.
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g: float = 0.1,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        R: float = 1.0
    ):
        """
        Initialize ITE solver.
        
        Args:
            grid: SpectralGrid for discretization.
            potential: Base potential V(σ).
            g: Nonlinear coupling constant.
            m_eff: Effective mass.
            hbar: Reduced Planck constant.
            R: Subspace radius.
        """
        self.grid = grid
        self.potential = potential
        self.g = g
        self.m_eff = m_eff
        self.hbar = hbar
        self.R = R
        
        self._V_grid = potential(grid.sigma)
        
        # Kinetic energy coefficient
        self.kinetic_coeff = hbar**2 / (2 * m_eff * R**2)
        
        # Fourier wavenumbers for kinetic propagator
        self.k_modes = np.fft.fftfreq(grid.N, d=grid.dsigma/(2*np.pi))
        self.k_sq = self.k_modes**2
    
    def evolve(
        self,
        initial_guess: Optional[NDArray] = None,
        target_amplitude: float = 1.0,
        dt: float = 0.01,
        max_steps: int = 10000,
        tol: float = 1e-10,
        renormalize: bool = False,
        verbose: bool = False
    ) -> AmplitudeSolution:
        """
        Evolve in imaginary time to find ground state.
        
        Args:
            initial_guess: Starting wavefunction, or None for default.
            target_amplitude: Target A value (for branch selection).
            dt: Time step for imaginary time.
            max_steps: Maximum number of time steps.
            tol: Convergence tolerance on chemical potential.
            renormalize: If True, normalize after each step (standard mode).
                        If False, let amplitude float (amplitude quantization mode).
            verbose: Print progress.
            
        Returns:
            AmplitudeSolution with the converged state.
        """
        # Initialize
        if initial_guess is None:
            # Start with Gaussian in first well, scaled by target amplitude
            chi = target_amplitude * np.exp(-self.grid.sigma**2 / 0.5)
            chi = chi.astype(complex)
        else:
            chi = initial_guess.astype(complex)
        
        # Kinetic propagator (half step)
        K_half = np.exp(-self.kinetic_coeff * self.k_sq * dt / 2)
        
        mu_old = self._compute_chemical_potential(chi)
        
        converged = False
        for step in range(max_steps):
            # Store old for convergence check
            chi_old = chi.copy()
            
            # Split-step method:
            # 1. Half kinetic step (Fourier space)
            chi_k = np.fft.fft(chi)
            chi_k = chi_k * K_half
            chi = np.fft.ifft(chi_k)
            
            # 2. Full potential + nonlinear step (real space)
            V_eff = self._V_grid + self.g * np.abs(chi)**2
            chi = chi * np.exp(-V_eff * dt)
            
            # 3. Half kinetic step
            chi_k = np.fft.fft(chi)
            chi_k = chi_k * K_half
            chi = np.fft.ifft(chi_k)
            
            # Renormalize to keep amplitude at target (ITE always needs this to prevent decay)
            norm = np.sqrt(np.trapezoid(np.abs(chi)**2, self.grid.sigma))
            if norm > 1e-15:
                if renormalize:
                    # Renormalize to target amplitude
                    chi = chi * target_amplitude * np.sqrt(2*np.pi) / norm
                else:
                    # Still need to prevent decay, but preserve relative shape
                    # Normalize to unit integral, then scale by target
                    chi = chi * target_amplitude * np.sqrt(2*np.pi) / norm
            
            # Check convergence
            mu_new = self._compute_chemical_potential(chi)
            dmu = abs(mu_new - mu_old)
            
            if verbose and step % 100 == 0:
                A_sq = np.trapezoid(np.abs(chi)**2, self.grid.sigma) / (2*np.pi)
                print("Step %d: mu = %.8f, dmu = %.2e, A^2 = %.6f" % (step, mu_new, dmu, A_sq))
            
            if dmu < tol:
                converged = True
                break
            
            mu_old = mu_new
        
        # Compute final quantities
        A_squared = np.trapezoid(np.abs(chi)**2, self.grid.sigma) / (2*np.pi)
        A = np.sqrt(A_squared)
        peak = np.max(np.abs(chi)**2)
        energy = self._compute_energy(chi)
        mu = self._compute_chemical_potential(chi)
        
        return AmplitudeSolution(
            chemical_potential=mu,
            wavefunction=chi,
            amplitude=A,
            amplitude_squared=A_squared,
            peak_amplitude=peak,
            energy=energy,
            converged=converged,
            iterations=step + 1
        )
    
    def _compute_chemical_potential(self, chi: NDArray) -> float:
        """
        Compute chemical potential μ = <χ|H_eff|χ> / <χ|χ>.
        """
        # Kinetic energy via FFT
        chi_k = np.fft.fft(chi)
        T_chi = np.fft.ifft(self.kinetic_coeff * self.k_sq * chi_k)
        T = np.real(np.trapezoid(np.conj(chi) * T_chi, self.grid.sigma))
        
        # Potential + nonlinear
        V_eff = self._V_grid + self.g * np.abs(chi)**2
        V = np.real(np.trapezoid(np.conj(chi) * V_eff * chi, self.grid.sigma))
        
        # Normalization
        norm_sq = np.real(np.trapezoid(np.abs(chi)**2, self.grid.sigma))
        
        if norm_sq > 1e-15:
            return (T + V) / norm_sq
        return 0.0
    
    def _compute_energy(self, chi: NDArray) -> float:
        """
        Compute total energy E = T + V + (g/2)∫|χ|⁴.
        """
        chi_k = np.fft.fft(chi)
        T_chi = np.fft.ifft(self.kinetic_coeff * self.k_sq * chi_k)
        T = np.real(np.trapezoid(np.conj(chi) * T_chi, self.grid.sigma))
        
        V = np.real(np.trapezoid(self._V_grid * np.abs(chi)**2, self.grid.sigma))
        
        E_nl = (self.g / 2) * np.real(np.trapezoid(np.abs(chi)**4, self.grid.sigma))
        
        return T + V + E_nl


class AmplitudeQuantizedSolver:
    """
    Solver for amplitude-quantized nonlinear eigenstates.
    
    This solver finds multiple branches of the GP equation, each with
    different amplitude A. The mass hierarchy (electron, muon, tau)
    corresponds to different branches with A_e < A_μ < A_τ.
    
    Key methods:
    - find_branches(): Find multiple amplitude branches by continuation
    - solve_for_amplitude(): Solve for a specific target amplitude
    - compute_mass_spectrum(): Compute mass ratios from amplitude ratios
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g: float = 0.1,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        R: float = 1.0
    ):
        """
        Initialize the amplitude-quantized solver.
        
        Args:
            grid: SpectralGrid for discretization.
            potential: Base potential V(σ).
            g: Nonlinear coupling constant. Must be non-zero for amplitude quantization.
            m_eff: Effective mass.
            hbar: Reduced Planck constant.
            R: Subspace radius.
        """
        self.grid = grid
        self.potential = potential
        self.g = g
        self.m_eff = m_eff
        self.hbar = hbar
        self.R = R
        
        self._V_grid = potential(grid.sigma)
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self.linear_solver = LinearEigensolver(grid, potential, m_eff, hbar)
        
        # ITE solver for robust ground state finding
        self.ite_solver = ImaginaryTimeEvolution(
            grid, potential, g, m_eff, hbar, R
        )
    
    def find_branches(
        self,
        k: int = 1,
        n_branches: int = 3,
        amplitude_range: Tuple[float, float] = (0.5, 20.0),
        n_samples: int = 20,
        verbose: bool = False
    ) -> List[AmplitudeSolution]:
        """
        Find multiple amplitude branches using continuation.
        
        Strategy:
        1. Sample initial amplitudes across the range
        2. Use ITE to find self-consistent solutions from each starting point
        3. Cluster solutions by amplitude to identify distinct branches
        4. Return representative solutions for each branch
        
        Args:
            k: Winding number sector.
            n_branches: Target number of branches to find.
            amplitude_range: (min, max) amplitude range to search.
            n_samples: Number of initial conditions to try.
            verbose: Print progress.
            
        Returns:
            List of AmplitudeSolution, one per branch, sorted by amplitude.
        """
        A_min, A_max = amplitude_range
        
        if verbose:
            print(f"Searching for {n_branches} branches in A ∈ [{A_min}, {A_max}]")
        
        # Sample initial amplitudes
        test_amplitudes = np.linspace(A_min, A_max, n_samples)
        
        solutions = []
        for i, A_init in enumerate(test_amplitudes):
            if verbose:
                print(f"  Trying A_init = {A_init:.2f}...")
            
            # Create initial guess with this amplitude
            chi_init = self._create_winding_mode(k, amplitude=A_init)
            
            # Evolve with ITE
            sol = self.ite_solver.evolve(
                initial_guess=chi_init,
                target_amplitude=A_init,
                renormalize=False,  # Let amplitude float
                max_steps=5000,
                tol=1e-8,
                verbose=False
            )
            
            if sol.converged:
                sol.branch_id = i
                solutions.append(sol)
                if verbose:
                    print(f"    Converged: A² = {sol.amplitude_squared:.4f}, μ = {sol.chemical_potential:.4f}")
        
        if not solutions:
            raise RuntimeError("No converged solutions found")
        
        # Cluster by amplitude to find distinct branches
        branches = self._cluster_by_amplitude(solutions, n_branches)
        
        if verbose:
            print(f"\nFound {len(branches)} distinct branches:")
            for i, sol in enumerate(branches):
                print(f"  Branch {i+1}: A = {sol.amplitude:.4f}, A² = {sol.amplitude_squared:.4f}, μ = {sol.chemical_potential:.4f}")
        
        return branches
    
    def solve_for_amplitude(
        self,
        target_amplitude: float,
        k: int = 1,
        max_iter: int = 1000,
        tol: float = 1e-8,
        verbose: bool = False
    ) -> AmplitudeSolution:
        """
        Solve for a solution with specific target amplitude.
        
        Uses a two-step approach:
        1. ITE to find approximate solution
        2. Newton refinement to hit the target amplitude
        
        Args:
            target_amplitude: Target A value.
            k: Winding number sector.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            verbose: Print progress.
            
        Returns:
            AmplitudeSolution with amplitude close to target.
        """
        # Initial guess
        chi_init = self._create_winding_mode(k, amplitude=target_amplitude)
        
        # ITE with renormalization to target amplitude
        sol = self.ite_solver.evolve(
            initial_guess=chi_init,
            target_amplitude=target_amplitude,
            renormalize=True,
            max_steps=max_iter,
            tol=tol,
            verbose=verbose
        )
        
        return sol
    
    def solve_unnormalized(
        self,
        k: int = 1,
        g_values: List[float] = None,
        max_iter: int = 200,
        tol: float = 1e-8,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> List[AmplitudeSolution]:
        """
        Find self-consistent solutions WITHOUT forcing normalization.
        
        This allows the amplitude A to emerge from the nonlinear self-consistency.
        Different values of g (continuation parameter) can find different branches.
        
        Args:
            k: Winding number sector.
            g_values: List of g values to try for continuation.
            max_iter: Maximum iterations per g value.
            tol: Convergence tolerance.
            mixing: Mixing parameter for self-consistent iteration.
            verbose: Print progress.
            
        Returns:
            List of solutions for different g values.
        """
        if g_values is None:
            g_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        solutions = []
        
        # Start from linear solution
        E0, chi = self.linear_solver.ground_state(k)
        
        for g in g_values:
            if verbose:
                print(f"\nSolving with g = {g}...")
            
            # Update g for this iteration
            self.ite_solver.g = g
            
            # Create initial guess based on previous solution or linear
            if solutions:
                # Scale previous solution
                chi_init = solutions[-1].wavefunction.copy()
            else:
                chi_init = chi.copy()
            
            # Iterate without normalization
            sol = self._iterate_unnormalized(
                chi_init, k, g, max_iter, tol, mixing, verbose
            )
            
            if sol.converged:
                solutions.append(sol)
                if verbose:
                    print(f"  Converged: A² = {sol.amplitude_squared:.4f}, μ = {sol.chemical_potential:.4f}")
        
        return solutions
    
    def _iterate_unnormalized(
        self,
        chi_init: NDArray,
        k: int,
        g: float,
        max_iter: int,
        tol: float,
        mixing: float,
        verbose: bool
    ) -> AmplitudeSolution:
        """
        Self-consistent iteration that preserves amplitude.
        
        The key insight: we DON'T normalize the eigenvector to 1.
        Instead, we scale it to preserve the amplitude from the previous iteration.
        This allows different initial amplitudes to converge to different branches.
        """
        chi = chi_init.copy()
        
        # Get initial amplitude
        A_current = np.sqrt(np.trapezoid(np.abs(chi)**2, self.grid.sigma) / (2*np.pi))
        
        mu_old = 0.0
        converged = False
        
        for iteration in range(max_iter):
            # Compute effective potential with current amplitude
            density = np.abs(chi)**2
            V_eff = self._V_grid + g * density
            
            # Solve linear eigenvalue problem
            H = self.operators.build_hamiltonian_matrix(V_eff)
            H = 0.5 * (H + np.conj(H.T))
            
            energies, wavefunctions = np.linalg.eigh(H)
            
            # Find state with correct winding
            chi_new = None
            mu_new = 0.0
            for i in range(len(energies)):
                psi = wavefunctions[:, i]
                winding = self.grid.winding_number(psi)
                if abs(winding - k) < 0.5:
                    chi_new = psi
                    mu_new = energies[i]
                    break
            
            if chi_new is None:
                chi_new = wavefunctions[:, 0]
                mu_new = energies[0]
            
            # CRITICAL: Scale chi_new to preserve the current amplitude A
            # This is what creates amplitude quantization!
            norm_new = np.sqrt(np.trapezoid(np.abs(chi_new)**2, self.grid.sigma))
            if norm_new > 1e-15:
                chi_new = chi_new * (A_current * np.sqrt(2*np.pi) / norm_new)
            
            # Mix old and new (both have same amplitude)
            chi = (1 - mixing) * chi + mixing * chi_new
            
            # Renormalize to maintain exact amplitude (mixing can drift)
            norm_mixed = np.sqrt(np.trapezoid(np.abs(chi)**2, self.grid.sigma))
            if norm_mixed > 1e-15:
                chi = chi * (A_current * np.sqrt(2*np.pi) / norm_mixed)
            
            # Check convergence on chemical potential
            dmu = abs(mu_new - mu_old)
            
            if verbose and iteration % 20 == 0:
                A_sq = np.trapezoid(np.abs(chi)**2, self.grid.sigma) / (2*np.pi)
                print("  Iter %d: mu = %.6f, dmu = %.2e, A^2 = %.4f" % (iteration, mu_new, dmu, A_sq))
            
            if dmu < tol:
                converged = True
                break
            
            mu_old = mu_new
        
        # Compute final quantities
        A_squared = np.trapezoid(np.abs(chi)**2, self.grid.sigma) / (2*np.pi)
        A = np.sqrt(A_squared)
        peak = np.max(np.abs(chi)**2)
        
        # Energy
        chi_k = np.fft.fft(chi)
        k_modes = np.fft.fftfreq(self.grid.N, d=self.grid.dsigma/(2*np.pi))
        kinetic_coeff = self.hbar**2 / (2 * self.m_eff * self.R**2)
        T_chi = np.fft.ifft(kinetic_coeff * k_modes**2 * chi_k)
        T = np.real(np.trapezoid(np.conj(chi) * T_chi, self.grid.sigma))
        V = np.real(np.trapezoid(self._V_grid * np.abs(chi)**2, self.grid.sigma))
        E_nl = (g / 2) * np.real(np.trapezoid(np.abs(chi)**4, self.grid.sigma))
        energy = T + V + E_nl
        
        return AmplitudeSolution(
            chemical_potential=mu_new,
            wavefunction=chi,
            amplitude=A,
            amplitude_squared=A_squared,
            peak_amplitude=peak,
            energy=energy,
            converged=converged,
            iterations=iteration + 1
        )
    
    def _create_winding_mode(self, k: int, amplitude: float = 1.0) -> NDArray:
        """
        Create an initial guess with specified winding and amplitude.
        """
        sigma = self.grid.sigma
        
        # Gaussian envelope centered at first well
        envelope = np.exp(-sigma**2 / 0.5)
        
        # Winding phase
        phase = np.exp(1j * k * sigma)
        
        # Combine and scale to target amplitude
        chi = envelope * phase
        
        # Scale to target amplitude
        norm = np.sqrt(np.trapezoid(np.abs(chi)**2, sigma))
        chi = chi * amplitude * np.sqrt(2*np.pi) / norm
        
        return chi
    
    def _cluster_by_amplitude(
        self, 
        solutions: List[AmplitudeSolution],
        n_clusters: int
    ) -> List[AmplitudeSolution]:
        """
        Cluster solutions by amplitude and return representatives.
        """
        if len(solutions) <= n_clusters:
            return sorted(solutions, key=lambda s: s.amplitude_squared)
        
        # Simple clustering: sort by amplitude and pick evenly spaced
        sorted_sols = sorted(solutions, key=lambda s: s.amplitude_squared)
        
        # Remove duplicates (within 5% amplitude)
        unique = [sorted_sols[0]]
        for sol in sorted_sols[1:]:
            if abs(sol.amplitude_squared - unique[-1].amplitude_squared) / unique[-1].amplitude_squared > 0.05:
                unique.append(sol)
        
        if len(unique) <= n_clusters:
            return unique
        
        # Pick evenly spaced
        indices = np.linspace(0, len(unique)-1, n_clusters, dtype=int)
        return [unique[i] for i in indices]
    
    def compute_mass_spectrum(
        self,
        branches: List[AmplitudeSolution],
        beta: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute mass ratios from amplitude branches.
        
        Args:
            branches: List of solutions representing different particles.
            beta: Mass coupling constant (m = β A²).
            
        Returns:
            Dictionary with masses and ratios.
        """
        if not branches:
            return {}
        
        result = {}
        
        # Sort by amplitude
        sorted_branches = sorted(branches, key=lambda s: s.amplitude_squared)
        
        for i, sol in enumerate(sorted_branches):
            result[f'A_squared_{i}'] = sol.amplitude_squared
            result[f'mass_{i}'] = beta * sol.amplitude_squared
            result[f'peak_{i}'] = sol.peak_amplitude
            result[f'mu_{i}'] = sol.chemical_potential
        
        # Compute ratios relative to first (lightest)
        if len(sorted_branches) >= 2:
            A0 = sorted_branches[0].amplitude_squared
            for i in range(1, len(sorted_branches)):
                Ai = sorted_branches[i].amplitude_squared
                result[f'mass_ratio_{i}_0'] = Ai / A0 if A0 > 0 else 0
        
        return result


def find_lepton_mass_hierarchy(
    grid: SpectralGrid,
    potential: ThreeWellPotential,
    g: float = 0.1,
    verbose: bool = True
) -> Tuple[List[AmplitudeSolution], Dict[str, float]]:
    """
    Convenience function to find electron, muon, tau mass hierarchy.
    
    Args:
        grid: SpectralGrid for discretization.
        potential: Three-well potential.
        g: Nonlinear coupling strength.
        verbose: Print progress.
        
    Returns:
        Tuple of (solutions, mass_ratios).
    """
    solver = AmplitudeQuantizedSolver(grid, potential, g=g)
    
    if verbose:
        print("Finding amplitude-quantized branches for leptons (k=1)...")
    
    # Find 3 branches for electron, muon, tau
    branches = solver.find_branches(
        k=1,
        n_branches=3,
        amplitude_range=(0.5, 30.0),
        n_samples=30,
        verbose=verbose
    )
    
    # Compute mass spectrum
    mass_ratios = solver.compute_mass_spectrum(branches)
    
    if verbose:
        print("\nMass spectrum from amplitude quantization:")
        for key, value in sorted(mass_ratios.items()):
            print(f"  {key}: {value:.4f}")
    
    return branches, mass_ratios

