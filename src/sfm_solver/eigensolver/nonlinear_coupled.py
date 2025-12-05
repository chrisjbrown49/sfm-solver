"""
Nonlinear coupled subspace-spacetime solver for amplitude quantization.

This solver finds self-consistent solutions of the coupled nonlinear system
where different radial modes (n=1,2,3 for e,μ,τ) produce different
peak amplitudes A in subspace, leading to the mass hierarchy m = β A².

The key physics:
1. Radial modes φ_n(r) couple to subspace modes χ(σ) via the coupling term
2. The nonlinear term g|χ|² makes the problem self-consistent
3. Different radial structures lead to different self-consistent A values
4. This breaks the degeneracy that linear solvers have

The coupled nonlinear equation:
    [H_radial + H_subspace + H_coupling + g|ψ|²] ψ(r,σ) = μ ψ(r,σ)

where:
    H_radial = -1/(2m_r) ∇²_r + V_conf(r)
    H_subspace = -1/(2m_σ R²) d²/dσ² + V_3well(σ)
    H_coupling = -α ∂²/∂r∂σ
    
The amplitude quantization emerges because different radial modes 
have different spatial extent, creating different effective g|ψ|².
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from sfm_solver.spatial.radial import RadialGrid
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.coupled_solver import CoupledGrids, CoupledHamiltonian


@dataclass
class NonlinearCoupledSolution:
    """Solution of the nonlinear coupled problem."""
    radial_mode: int  # n = 1, 2, 3 for e, μ, τ
    subspace_winding: int  # k = 1 for leptons
    chemical_potential: float  # μ
    energy: float  # Total energy
    wavefunction: NDArray  # Full ψ(r,σ)
    peak_amplitude: float  # max|χ(σ)|²
    integrated_amplitude: float  # ∫|χ|² dσ / 2π
    converged: bool
    iterations: int


class NonlinearCoupledSolver:
    """
    Nonlinear self-consistent solver for coupled subspace-spacetime.
    
    This solver iterates to find self-consistent solutions where the
    nonlinear term g|ψ|² creates amplitude quantization.
    
    Different radial modes (n=1,2,3) lead to different self-consistent
    peak amplitudes, producing the lepton mass hierarchy.
    """
    
    def __init__(
        self,
        radial_grid: RadialGrid,
        subspace_grid: SpectralGrid,
        potential: ThreeWellPotential,
        alpha: float = 0.1,
        g: float = 0.1,
        omega: float = 0.5,
        m_radial: float = 1.0,
        m_subspace: float = 1.0,
        R_subspace: float = 1.0
    ):
        """
        Initialize the nonlinear coupled solver.
        
        Args:
            radial_grid: Grid for radial dimension.
            subspace_grid: Grid for subspace (periodic).
            potential: Three-well potential in subspace.
            alpha: Coupling strength between radial and subspace.
            g: Nonlinear coupling constant g|ψ|².
            omega: Radial confinement frequency.
            m_radial: Radial effective mass.
            m_subspace: Subspace effective mass.
            R_subspace: Subspace radius.
        """
        self.radial = radial_grid
        self.subspace = subspace_grid
        self.potential = potential
        self.alpha = alpha
        self.g = g
        self.omega = omega
        self.m_radial = m_radial
        self.m_subspace = m_subspace
        self.R_subspace = R_subspace
        
        self.grids = CoupledGrids(radial=radial_grid, subspace=subspace_grid)
        
        # Build linear Hamiltonian components
        self._build_linear_hamiltonian()
    
    def _build_linear_hamiltonian(self):
        """Build the linear part of the coupled Hamiltonian."""
        # Coupled Hamiltonian without nonlinear term
        self.ham = CoupledHamiltonian(
            self.grids, 
            self.potential, 
            alpha=self.alpha,
            m_spatial=self.m_radial,
            m_subspace=self.m_subspace,
            R_subspace=self.R_subspace
        )
        
        # Radial confinement potential
        V_radial = 0.5 * self.omega**2 * self.radial.r**2
        V_radial_matrix = diags([V_radial], [0], 
                                 shape=(self.radial.N, self.radial.N), 
                                 format='csr')
        self.V_radial_sparse = kron(V_radial_matrix, 
                                     eye(self.subspace.N, format='csr'), 
                                     format='csr')
        
        # Base linear Hamiltonian
        self.H_linear = self.ham.build_matrix() + self.V_radial_sparse
        self.H_linear = 0.5 * (self.H_linear + self.H_linear.T.conj())
    
    def solve_for_radial_mode(
        self,
        n_radial: int,
        k_subspace: int = 1,
        max_iter: int = 200,
        tol: float = 1e-8,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> NonlinearCoupledSolution:
        """
        Solve for a specific radial mode with nonlinear self-consistency.
        
        Args:
            n_radial: Target radial quantum number (1, 2, 3 for e, μ, τ).
            k_subspace: Target subspace winding number.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            mixing: Mixing parameter.
            verbose: Print progress.
            
        Returns:
            NonlinearCoupledSolution with self-consistent amplitude.
        """
        # Step 1: Get linear eigenstates to find initial guess
        psi = self._get_initial_guess(n_radial, k_subspace)
        
        mu_old = 0.0
        converged = False
        
        for iteration in range(max_iter):
            # Step 2: Build effective Hamiltonian with nonlinear term
            H_eff = self._build_effective_hamiltonian(psi)
            
            # Step 3: Solve eigenvalue problem
            n_target = max(n_radial + 5, 20)  # Get enough states
            try:
                energies, wavefunctions = eigsh(H_eff, k=n_target, which='SA')
            except Exception:
                # Fall back to more states if needed
                energies, wavefunctions = eigsh(H_eff, k=min(n_target*2, H_eff.shape[0]-2), which='SA')
            
            idx = np.argsort(energies)
            energies = energies[idx]
            wavefunctions = wavefunctions[:, idx]
            
            # Step 4: Find the state matching our target (n_radial, k_subspace)
            psi_new, mu_new = self._find_target_state(
                wavefunctions, energies, n_radial, k_subspace
            )
            
            if psi_new is None:
                if verbose:
                    print("  Could not find target state, using ground state")
                psi_new = wavefunctions[:, 0]
                mu_new = energies[0]
            
            # Step 5: Normalize (preserving shape, unit total probability)
            psi_new = psi_new / np.sqrt(self.grids.integrate(psi_new))
            
            # Step 6: Mix with previous
            psi = (1 - mixing) * psi + mixing * psi_new
            psi = psi / np.sqrt(self.grids.integrate(psi))
            
            # Check convergence
            dmu = abs(mu_new - mu_old)
            
            if verbose and iteration % 10 == 0:
                peak = self._compute_peak_amplitude(psi)
                print("  Iter %d: mu = %.6f, dmu = %.2e, peak = %.4f" % 
                      (iteration, mu_new, dmu, peak))
            
            if dmu < tol:
                converged = True
                break
            
            mu_old = mu_new
        
        # Compute final quantities
        peak = self._compute_peak_amplitude(psi)
        integrated = self._compute_integrated_amplitude(psi)
        energy = self._compute_energy(psi, mu_new)
        
        return NonlinearCoupledSolution(
            radial_mode=n_radial,
            subspace_winding=k_subspace,
            chemical_potential=mu_new,
            energy=energy,
            wavefunction=psi,
            peak_amplitude=peak,
            integrated_amplitude=integrated,
            converged=converged,
            iterations=iteration + 1
        )
    
    def solve_lepton_spectrum(
        self,
        verbose: bool = True
    ) -> Tuple[List[NonlinearCoupledSolution], Dict[str, float]]:
        """
        Solve for electron, muon, tau (n=1,2,3 with k=1).
        
        Returns:
            Tuple of (solutions, mass_ratios).
        """
        if verbose:
            print("Solving nonlinear coupled problem for lepton spectrum")
            print("="*60)
        
        solutions = []
        
        for n in [1, 2, 3]:
            if verbose:
                print("\nSolving for n=%d (radial mode):" % n)
            
            sol = self.solve_for_radial_mode(
                n_radial=n,
                k_subspace=1,
                verbose=verbose
            )
            solutions.append(sol)
            
            if verbose:
                print("  Result: peak_A^2 = %.4f, mu = %.4f, converged = %s" % 
                      (sol.peak_amplitude, sol.chemical_potential, sol.converged))
        
        # Compute mass ratios from peak amplitudes
        mass_ratios = {}
        if solutions:
            A_e = solutions[0].peak_amplitude
            mass_ratios['m_e_rel'] = 1.0
            mass_ratios['peak_e'] = A_e
            
            if len(solutions) > 1:
                A_mu = solutions[1].peak_amplitude
                mass_ratios['peak_mu'] = A_mu
                mass_ratios['m_mu_m_e'] = A_mu / A_e if A_e > 0 else 0
            
            if len(solutions) > 2:
                A_tau = solutions[2].peak_amplitude
                mass_ratios['peak_tau'] = A_tau
                mass_ratios['m_tau_m_e'] = A_tau / A_e if A_e > 0 else 0
        
        if verbose:
            print("\n" + "="*60)
            print("Mass ratios from peak amplitude:")
            print("  m_mu/m_e = %.4f (expected: 206.77)" % mass_ratios.get('m_mu_m_e', 0))
            print("  m_tau/m_e = %.4f (expected: 3477.15)" % mass_ratios.get('m_tau_m_e', 0))
        
        return solutions, mass_ratios
    
    def _get_initial_guess(self, n_radial: int, k_subspace: int) -> NDArray:
        """Get initial guess from linear eigenstates."""
        # Solve linear problem
        n_states = max(n_radial * 10, 30)
        energies, wavefunctions = eigsh(self.H_linear, k=n_states, which='SA')
        idx = np.argsort(energies)
        energies = energies[idx]
        wavefunctions = wavefunctions[:, idx]
        
        # Find state with target quantum numbers
        psi, _ = self._find_target_state(
            wavefunctions, energies, n_radial, k_subspace
        )
        
        if psi is None:
            # Default to ground state
            psi = wavefunctions[:, 0]
        
        # Normalize
        psi = psi / np.sqrt(self.grids.integrate(psi))
        
        return psi
    
    def _find_target_state(
        self,
        wavefunctions: NDArray,
        energies: NDArray,
        n_radial: int,
        k_subspace: int
    ) -> Tuple[Optional[NDArray], float]:
        """Find state with target quantum numbers."""
        candidates = []
        
        for i in range(wavefunctions.shape[1]):
            psi = wavefunctions[:, i]
            psi_2d = psi.reshape(self.grids.shape)
            
            # Normalize for analysis
            norm = np.sqrt(self.grids.integrate(psi))
            if norm > 1e-10:
                psi_2d = psi_2d / norm
            
            # Check subspace winding
            subspace_proj = np.sum(psi_2d, axis=0)
            fft_sub = np.fft.fft(subspace_proj)
            k_dom = np.argmax(np.abs(fft_sub)[1:6]) + 1
            
            if k_dom != k_subspace:
                continue
            
            # Check radial mode
            phi_r = np.zeros(self.radial.N)
            for ii in range(self.radial.N):
                phi_r[ii] = np.trapezoid(np.abs(psi_2d[ii, :])**2, self.subspace.sigma)
            
            phi_norm = phi_r / np.max(phi_r) if np.max(phi_r) > 0 else phi_r
            peaks = np.where((phi_norm[1:-1] > phi_norm[:-2]) & 
                            (phi_norm[1:-1] > phi_norm[2:]) &
                            (phi_norm[1:-1] > 0.05))[0]
            n_peaks = len(peaks) + 1  # +1 for ground state convention
            
            candidates.append({
                'idx': i,
                'n': n_peaks,
                'E': energies[i],
                'psi': psi
            })
        
        # Find best match
        for c in candidates:
            if c['n'] == n_radial:
                return c['psi'], c['E']
        
        # If no exact match, return closest
        if candidates:
            candidates.sort(key=lambda x: abs(x['n'] - n_radial))
            return candidates[0]['psi'], candidates[0]['E']
        
        return None, 0.0
    
    def _build_effective_hamiltonian(self, psi: NDArray) -> csr_matrix:
        """Build H_eff = H_linear + g|ψ|²."""
        density = np.abs(psi)**2
        V_nl = self.g * density
        V_nl_diag = diags([V_nl], [0], shape=self.H_linear.shape, format='csr')
        
        H_eff = self.H_linear + V_nl_diag
        return 0.5 * (H_eff + H_eff.T.conj())
    
    def _compute_peak_amplitude(self, psi: NDArray) -> float:
        """Compute peak |χ(σ)|² from subspace marginal."""
        psi_2d = psi.reshape(self.grids.shape)
        
        # Integrate over radial dimension to get χ(σ)
        chi_sigma = np.zeros(self.subspace.N)
        for j in range(self.subspace.N):
            integrand = np.abs(psi_2d[:, j])**2 * self.radial.r**2
            chi_sigma[j] = 4 * np.pi * np.trapezoid(integrand, self.radial.r)
        
        return np.max(chi_sigma)
    
    def _compute_integrated_amplitude(self, psi: NDArray) -> float:
        """Compute ∫|χ|² dσ / 2π."""
        psi_2d = psi.reshape(self.grids.shape)
        
        chi_sigma = np.zeros(self.subspace.N)
        for j in range(self.subspace.N):
            integrand = np.abs(psi_2d[:, j])**2 * self.radial.r**2
            chi_sigma[j] = 4 * np.pi * np.trapezoid(integrand, self.radial.r)
        
        return np.trapezoid(chi_sigma, self.subspace.sigma) / (2 * np.pi)
    
    def _compute_energy(self, psi: NDArray, mu: float) -> float:
        """Compute total energy including nonlinear term."""
        # Linear energy
        E_lin = np.real(np.vdot(psi, self.H_linear @ psi))
        
        # Nonlinear energy: (g/2) ∫|ψ|⁴
        density = np.abs(psi)**2
        E_nl = (self.g / 2) * self.grids.integrate(density**2)
        
        return E_lin + E_nl


def find_amplitude_quantization(
    g: float = 0.1,
    alpha: float = 0.1,
    omega: float = 0.5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Find amplitude quantization for lepton mass hierarchy.
    
    This is the main entry point for testing amplitude quantization.
    
    Args:
        g: Nonlinear coupling strength.
        alpha: Radial-subspace coupling.
        omega: Radial confinement.
        verbose: Print progress.
        
    Returns:
        Dictionary with mass ratios and other results.
    """
    radial = RadialGrid(N=48, r_max=12.0)
    subspace = SpectralGrid(N=32)
    potential = ThreeWellPotential(V0=1.0, V1=0.1)
    
    solver = NonlinearCoupledSolver(
        radial, subspace, potential,
        alpha=alpha, g=g, omega=omega
    )
    
    solutions, mass_ratios = solver.solve_lepton_spectrum(verbose=verbose)
    
    return mass_ratios

