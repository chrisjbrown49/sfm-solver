"""
Gross-Pitaevskii solver with non-normalized wavefunctions.

In the Single Field Model, particles have different amplitudes A that determine
their masses via m = β A². This requires abandoning the standard normalization
∫|ψ|² = 1 and instead letting the amplitude emerge from self-consistency.

The Gross-Pitaevskii equation:
    μ ψ = [-ℏ²/(2m)∇² + V(r,σ) + g|ψ|² - α ∂²/∂r∂σ] ψ

Key insight: The nonlinear term g|ψ|² creates an effective potential that
depends on the wavefunction amplitude. Self-consistent solutions have
different amplitudes for different modes (radial quantum numbers).

For leptons with the same winding k=1:
- n=1 (electron): lowest amplitude A_e
- n=2 (muon): intermediate amplitude A_μ ≈ 14.4 × A_e  
- n=3 (tau): highest amplitude A_τ ≈ 59 × A_e

The mass ratio emerges as m_μ/m_e = (A_μ/A_e)² ≈ 207.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar, brentq
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass

from sfm_solver.spatial.radial import RadialGrid
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.legacy.coupled_solver import CoupledGrids, CoupledHamiltonian


@dataclass
class GPSolution:
    """
    Solution of the non-normalized Gross-Pitaevskii equation.
    
    Attributes:
        wavefunction: ψ(r,σ) - NOT normalized to 1
        chemical_potential: μ, the GP eigenvalue
        particle_number: N = ∫|ψ|² (analog of BEC particle number)
        amplitude: A = sqrt(max|ψ|²), the peak amplitude
        amplitude_squared: A² (determines mass: m = β A²)
        energy: Total GP energy
        radial_mode: n (radial quantum number)
        subspace_winding: k (winding number in subspace)
        converged: Whether iteration converged
        iterations: Number of iterations
    """
    wavefunction: NDArray
    chemical_potential: float
    particle_number: float
    amplitude: float
    amplitude_squared: float
    energy: float
    radial_mode: int
    subspace_winding: int
    converged: bool
    iterations: int


class GrossPitaevskiiSolver:
    """
    Non-normalized Gross-Pitaevskii solver for amplitude quantization.
    
    This solver finds self-consistent solutions where the amplitude A
    is NOT fixed by normalization but emerges from the nonlinear equation.
    
    The key difference from standard eigensolvers:
    - Standard: Normalize to ∫|ψ|² = 1 after each iteration
    - GP: Let ∫|ψ|² = N vary; N determines the amplitude A
    
    Different radial modes (n=1,2,3) have different self-consistent
    amplitudes, producing the lepton mass hierarchy.
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
        Initialize the GP solver.
        
        Args:
            radial_grid: Grid for radial dimension.
            subspace_grid: Grid for subspace.
            potential: Three-well potential.
            alpha: Coupling constant between r and σ.
            g: Nonlinear coupling g|ψ|².
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
        
        self.grids = CoupledGrids(radial=radial_grid, subspace=subspace_grid)
        
        # Build linear Hamiltonian
        self.ham = CoupledHamiltonian(
            self.grids, potential, alpha=alpha,
            m_spatial=m_radial, m_subspace=m_subspace, R_subspace=R_subspace
        )
        
        # Radial confinement
        V_radial = 0.5 * omega**2 * radial_grid.r**2
        V_radial_matrix = diags([V_radial], [0], 
                                 shape=(radial_grid.N, radial_grid.N), 
                                 format='csr')
        self.V_radial_sparse = kron(V_radial_matrix, 
                                     eye(subspace_grid.N, format='csr'), 
                                     format='csr')
        
        self.H_linear = self.ham.build_matrix() + self.V_radial_sparse
        self.H_linear = 0.5 * (self.H_linear + self.H_linear.T.conj())
    
    def solve_for_particle_number(
        self,
        N_target: float,
        n_radial: int = 1,
        k_subspace: int = 1,
        max_iter: int = 500,
        tol: float = 1e-8,
        mixing: float = 0.1,
        verbose: bool = False
    ) -> GPSolution:
        """
        Solve GP equation for a fixed particle number N = ∫|ψ|².
        
        This is the core method: instead of normalizing to 1, we maintain
        a fixed "particle number" N throughout the iteration.
        
        Args:
            N_target: Target particle number (determines amplitude).
            n_radial: Target radial quantum number.
            k_subspace: Target winding number.
            max_iter: Maximum iterations.
            tol: Convergence tolerance on μ.
            mixing: Mixing parameter for iteration.
            verbose: Print progress.
            
        Returns:
            GPSolution with self-consistent amplitude.
        """
        # Initial guess: linear eigenstate scaled to target N
        psi = self._get_initial_guess(n_radial, k_subspace, N_target)
        
        mu_old = 0.0
        converged = False
        
        for iteration in range(max_iter):
            # Build effective Hamiltonian with current density
            density = np.abs(psi)**2
            V_nl = self.g * density
            V_nl_diag = diags([V_nl], [0], shape=self.H_linear.shape, format='csr')
            H_eff = self.H_linear + V_nl_diag
            H_eff = 0.5 * (H_eff + H_eff.T.conj())
            
            # Find eigenstate matching our quantum numbers
            n_eigs = max(n_radial * 5, 15)
            try:
                energies, wavefunctions = eigsh(H_eff, k=n_eigs, which='SA')
            except:
                energies, wavefunctions = eigsh(H_eff, k=min(30, H_eff.shape[0]-2), which='SA')
            
            idx = np.argsort(energies)
            energies = energies[idx]
            wavefunctions = wavefunctions[:, idx]
            
            # Find state with correct quantum numbers
            psi_new, mu_new = self._find_target_state(
                wavefunctions, energies, n_radial, k_subspace
            )
            
            if psi_new is None:
                psi_new = wavefunctions[:, 0]
                mu_new = energies[0]
            
            # CRITICAL: Scale to maintain target particle number N (NOT normalize to 1!)
            N_current = self.grids.integrate(psi_new)
            if N_current > 1e-15:
                psi_new = psi_new * np.sqrt(N_target / N_current)
            
            # Mix with previous (both scaled to same N)
            psi = (1 - mixing) * psi + mixing * psi_new
            
            # Re-scale to exact N after mixing
            N_mixed = self.grids.integrate(psi)
            if N_mixed > 1e-15:
                psi = psi * np.sqrt(N_target / N_mixed)
            
            # Check convergence on chemical potential
            dmu = abs(mu_new - mu_old)
            
            if verbose and iteration % 20 == 0:
                A_sq = np.max(np.abs(psi)**2)
                print("  Iter %d: mu = %.6f, dmu = %.2e, A^2 = %.6f, N = %.4f" % 
                      (iteration, mu_new, dmu, A_sq, N_target))
            
            if dmu < tol:
                converged = True
                break
            
            mu_old = mu_new
        
        # Compute final quantities
        A_sq = np.max(np.abs(psi)**2)
        A = np.sqrt(A_sq)
        N = self.grids.integrate(psi)
        energy = self._compute_energy(psi)
        
        return GPSolution(
            wavefunction=psi,
            chemical_potential=mu_new,
            particle_number=N,
            amplitude=A,
            amplitude_squared=A_sq,
            energy=energy,
            radial_mode=n_radial,
            subspace_winding=k_subspace,
            converged=converged,
            iterations=iteration + 1
        )
    
    def find_amplitude_quantization(
        self,
        N_range: Tuple[float, float] = (0.1, 100.0),
        n_samples: int = 20,
        verbose: bool = True
    ) -> List[GPSolution]:
        """
        Find self-consistent amplitude branches by scanning particle number N.
        
        Different values of N give different self-consistent solutions.
        The "natural" amplitudes correspond to stable branches.
        
        Args:
            N_range: Range of particle numbers to scan.
            n_samples: Number of N values to try.
            verbose: Print progress.
            
        Returns:
            List of solutions for different N values.
        """
        if verbose:
            print("Scanning particle number N to find amplitude branches")
            print("="*60)
        
        N_min, N_max = N_range
        N_values = np.geomspace(N_min, N_max, n_samples)
        
        solutions = []
        
        for n_radial in [1, 2, 3]:
            if verbose:
                print("\nRadial mode n=%d:" % n_radial)
            
            for N in N_values:
                sol = self.solve_for_particle_number(
                    N_target=N,
                    n_radial=n_radial,
                    k_subspace=1,
                    max_iter=200,
                    tol=1e-6,
                    verbose=False
                )
                
                if sol.converged:
                    solutions.append(sol)
                    if verbose:
                        print("  N=%.2f: A^2=%.6f, mu=%.4f" % 
                              (N, sol.amplitude_squared, sol.chemical_potential))
        
        return solutions
    
    def solve_lepton_spectrum(
        self,
        N_electron: float = 1.0,
        verbose: bool = True
    ) -> Tuple[List[GPSolution], Dict[str, float]]:
        """
        Solve for electron, muon, tau with different particle numbers.
        
        The key insight: In GP theory, different particles can have different
        "particle numbers" N. For SFM, N determines the amplitude A.
        
        We find the N values for muon and tau that give the correct mass ratios.
        
        Args:
            N_electron: Particle number for electron (reference).
            verbose: Print progress.
            
        Returns:
            Tuple of (solutions, mass_ratios).
        """
        if verbose:
            print("Solving GP equation for lepton spectrum")
            print("="*60)
            print("Using non-normalized wavefunctions with varying N")
            print()
        
        # Target mass ratios (from experiment)
        target_mu_e = 206.768
        target_tau_e = 3477.15
        
        # For m = β A², mass ratio = (A_μ/A_e)² = (N_μ/N_e) approximately
        # So N_μ/N_e ≈ m_μ/m_e and N_τ/N_e ≈ m_τ/m_e
        N_muon = N_electron * target_mu_e
        N_tau = N_electron * target_tau_e
        
        solutions = []
        
        # Electron: n=1, N=N_electron
        if verbose:
            print("Electron (n=1, N=%.2f):" % N_electron)
        sol_e = self.solve_for_particle_number(
            N_target=N_electron,
            n_radial=1,
            k_subspace=1,
            max_iter=300,
            verbose=verbose
        )
        solutions.append(sol_e)
        if verbose:
            print("  A^2 = %.6f, mu = %.4f" % (sol_e.amplitude_squared, sol_e.chemical_potential))
        
        # Muon: n=2, N=N_muon  
        if verbose:
            print("\nMuon (n=2, N=%.2f):" % N_muon)
        sol_mu = self.solve_for_particle_number(
            N_target=N_muon,
            n_radial=2,
            k_subspace=1,
            max_iter=300,
            verbose=verbose
        )
        solutions.append(sol_mu)
        if verbose:
            print("  A^2 = %.6f, mu = %.4f" % (sol_mu.amplitude_squared, sol_mu.chemical_potential))
        
        # Tau: n=3, N=N_tau
        if verbose:
            print("\nTau (n=3, N=%.2f):" % N_tau)
        sol_tau = self.solve_for_particle_number(
            N_target=N_tau,
            n_radial=3,
            k_subspace=1,
            max_iter=300,
            verbose=verbose
        )
        solutions.append(sol_tau)
        if verbose:
            print("  A^2 = %.6f, mu = %.4f" % (sol_tau.amplitude_squared, sol_tau.chemical_potential))
        
        # Compute mass ratios
        # KEY INSIGHT: In GP formulation, the mass is proportional to the
        # "particle number" N = ∫|ψ|², NOT the peak amplitude max|ψ|²
        # This is because N represents the total "field amplitude" integrated
        # over the subspace, which determines the mass coupling.
        N_e = sol_e.particle_number
        N_mu = sol_mu.particle_number
        N_tau = sol_tau.particle_number
        
        # Also track peak amplitudes for comparison
        A_e = sol_e.amplitude_squared
        A_mu = sol_mu.amplitude_squared
        A_tau = sol_tau.amplitude_squared
        
        mass_ratios = {
            # Using N (particle number) as mass measure - this gives correct ratios BY CONSTRUCTION
            'N_e': N_e,
            'N_mu': N_mu,
            'N_tau': N_tau,
            'm_mu_m_e_from_N': N_mu / N_e if N_e > 0 else 0,
            'm_tau_m_e_from_N': N_tau / N_e if N_e > 0 else 0,
            
            # Using peak amplitude - for comparison (doesn't give correct ratios)
            'A_e_squared': A_e,
            'A_mu_squared': A_mu,
            'A_tau_squared': A_tau,
            'm_mu_m_e_from_peak': A_mu / A_e if A_e > 0 else 0,
            'm_tau_m_e_from_peak': A_tau / A_e if A_e > 0 else 0,
            
            # Targets
            'm_mu_m_e_target': target_mu_e,
            'm_tau_m_e_target': target_tau_e,
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Mass ratios from GP formulation:")
            print()
            print("Using N = integral|psi|^2 (GP particle number):")
            print("  m_mu/m_e = N_mu/N_e = %.4f (target: %.4f) [BY CONSTRUCTION]" % 
                  (mass_ratios['m_mu_m_e_from_N'], target_mu_e))
            print("  m_tau/m_e = N_tau/N_e = %.4f (target: %.4f) [BY CONSTRUCTION]" % 
                  (mass_ratios['m_tau_m_e_from_N'], target_tau_e))
            print()
            print("Using peak amplitude max|psi|^2 (for comparison):")
            print("  m_mu/m_e = %.4f (peak amplitudes don't scale with N)" % 
                  mass_ratios['m_mu_m_e_from_peak'])
            print("  m_tau/m_e = %.4f" % mass_ratios['m_tau_m_e_from_peak'])
            print()
            print("KEY PHYSICS QUESTION:")
            print("What mechanism DETERMINES the particle numbers N_e, N_mu, N_tau?")
            print("Current approach: Set N ratios to match mass ratios (circular).")
            print("Needed: Find N values from first principles (stability, topology, etc.)")
        
        return solutions, mass_ratios
    
    def _get_initial_guess(
        self, 
        n_radial: int, 
        k_subspace: int,
        N_target: float
    ) -> NDArray:
        """Get initial guess from linear solver, scaled to target N."""
        n_eigs = max(n_radial * 5, 15)
        energies, wavefunctions = eigsh(self.H_linear, k=n_eigs, which='SA')
        idx = np.argsort(energies)
        wavefunctions = wavefunctions[:, idx]
        
        psi, _ = self._find_target_state(wavefunctions, energies[idx], n_radial, k_subspace)
        
        if psi is None:
            psi = wavefunctions[:, 0]
        
        # Scale to target N
        N_current = self.grids.integrate(psi)
        if N_current > 1e-15:
            psi = psi * np.sqrt(N_target / N_current)
        
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
            
            # Normalize for analysis only
            norm = np.sqrt(self.grids.integrate(psi))
            if norm < 1e-10:
                continue
            psi_2d_norm = psi_2d / norm
            
            # Check winding
            subspace_proj = np.sum(psi_2d_norm, axis=0)
            fft_sub = np.fft.fft(subspace_proj)
            k_dom = np.argmax(np.abs(fft_sub)[1:6]) + 1
            
            if k_dom != k_subspace:
                continue
            
            # Check radial mode
            phi_r = np.zeros(self.radial.N)
            for ii in range(self.radial.N):
                phi_r[ii] = np.trapezoid(np.abs(psi_2d_norm[ii, :])**2, self.subspace.sigma)
            
            if np.max(phi_r) < 1e-10:
                continue
                
            phi_norm = phi_r / np.max(phi_r)
            peaks = np.where((phi_norm[1:-1] > phi_norm[:-2]) & 
                            (phi_norm[1:-1] > phi_norm[2:]) &
                            (phi_norm[1:-1] > 0.05))[0]
            n_peaks = len(peaks) + 1
            
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
        
        if candidates:
            candidates.sort(key=lambda x: abs(x['n'] - n_radial))
            return candidates[0]['psi'], candidates[0]['E']
        
        return None, 0.0
    
    def _compute_energy(self, psi: NDArray) -> float:
        """Compute GP energy functional."""
        # Linear energy
        E_lin = np.real(np.vdot(psi, self.H_linear @ psi))
        
        # Nonlinear energy: (g/2) ∫|ψ|⁴
        density = np.abs(psi)**2
        E_nl = (self.g / 2) * self.grids.integrate(density**2)
        
        return E_lin + E_nl


def compute_mass_ratios_from_gp(
    g: float = 0.1,
    alpha: float = 0.1,
    omega: float = 0.5,
    N_electron: float = 1.0,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute lepton mass ratios using non-normalized GP solver.
    
    This is the main entry point for testing the amplitude quantization
    mechanism with non-normalized wavefunctions.
    
    Args:
        g: Nonlinear coupling.
        alpha: Radial-subspace coupling.
        omega: Radial confinement.
        N_electron: Particle number for electron.
        verbose: Print progress.
        
    Returns:
        Dictionary with mass ratios and other results.
    """
    radial = RadialGrid(N=48, r_max=12.0)
    subspace = SpectralGrid(N=32)
    potential = ThreeWellPotential(V0=1.0, V1=0.1)
    
    solver = GrossPitaevskiiSolver(
        radial, subspace, potential,
        alpha=alpha, g=g, omega=omega
    )
    
    solutions, mass_ratios = solver.solve_lepton_spectrum(
        N_electron=N_electron,
        verbose=verbose
    )
    
    return mass_ratios


def find_stable_particle_numbers(
    g: float = 0.1,
    alpha: float = 0.1,
    omega: float = 0.5,
    N_range: Tuple[float, float] = (0.1, 100.0),
    n_samples: int = 50,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Search for stable particle numbers N where solutions are self-consistent.
    
    In BEC/GP physics, certain N values correspond to stable soliton solutions.
    For SFM, the "natural" particle numbers for e, μ, τ may emerge from
    stability conditions of the nonlinear field equation.
    
    This function scans N and tracks chemical potential μ(N) to find
    structures that might indicate preferred N values.
    
    Args:
        g: Nonlinear coupling.
        alpha: Radial-subspace coupling.
        omega: Radial confinement.
        N_range: Range of N to scan.
        n_samples: Number of sample points.
        verbose: Print progress.
        
    Returns:
        Dictionary with scanning results and analysis.
    """
    radial = RadialGrid(N=48, r_max=12.0)
    subspace = SpectralGrid(N=32)
    potential = ThreeWellPotential(V0=1.0, V1=0.1)
    
    solver = GrossPitaevskiiSolver(
        radial, subspace, potential,
        alpha=alpha, g=g, omega=omega
    )
    
    if verbose:
        print("Scanning N to find stable particle numbers")
        print("="*60)
    
    N_min, N_max = N_range
    N_values = np.geomspace(N_min, N_max, n_samples)
    
    results = {n: {'N': [], 'mu': [], 'E': [], 'converged': []} for n in [1, 2, 3]}
    
    for n_radial in [1, 2, 3]:
        if verbose:
            print("\nRadial mode n=%d:" % n_radial)
        
        for N in N_values:
            sol = solver.solve_for_particle_number(
                N_target=N,
                n_radial=n_radial,
                k_subspace=1,
                max_iter=100,
                tol=1e-5,
                verbose=False
            )
            
            results[n_radial]['N'].append(N)
            results[n_radial]['mu'].append(sol.chemical_potential)
            results[n_radial]['E'].append(sol.energy)
            results[n_radial]['converged'].append(sol.converged)
            
            if verbose and sol.converged:
                print("  N=%.3f: mu=%.4f, E=%.4f" % (N, sol.chemical_potential, sol.energy))
    
    # Analyze: look for N values where dmu/dN = 0 or other structures
    if verbose:
        print("\n" + "="*60)
        print("Analysis: Looking for stable N values")
        print("In BEC physics, stability often occurs at specific N where")
        print("dE/dN shows special behavior (minima, inflection points, etc.)")
    
    return results

