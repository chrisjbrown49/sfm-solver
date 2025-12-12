"""
SFM Lepton Solver - Pure First-Principles Implementation.

CRITICAL: Uses WAVEFUNCTION-BASED energy minimization.

The actual wavefunction structure matters:
- ONE peak at a well with k=1 winding
- Actual ∫V(σ)|χ|² captures single-well interaction
- Actual ∫|χ|⁴ captures nonlinear self-interaction
- k_eff emerges from ∫|∂χ/∂σ|²

This is fundamentally different from mesons (2 peaks) and baryons (3 peaks)!
The lepton occupies ONE well, not multiple wells like hadrons.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


LEPTON_WINDING = 1

LEPTON_SPATIAL_MODE = {
    'electron': 1,
    'muon': 2,
    'tau': 3,
}


@dataclass
class SFMLeptonState:
    """Result of SFM lepton solver."""
    chi: NDArray[np.complexfloating]
    particle: str
    
    # From minimizer
    amplitude: float
    amplitude_squared: float
    delta_x: float
    delta_sigma: float
    
    # Energy breakdown
    energy_total: float
    energy_subspace: float
    energy_spatial: float
    energy_coupling: float
    energy_curvature: float
    energy_kinetic: float
    energy_potential: float
    energy_nonlinear: float
    energy_circulation: float
    
    # Winding
    k: int = LEPTON_WINDING
    k_eff: float = 1.0
    
    # Spatial mode
    n_spatial: int = 1
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 0.0


class SFMLeptonSolver:
    """
    Lepton solver using WAVEFUNCTION-BASED energy minimization.
    
    Key difference from hadrons:
    - ONE peak (single particle) vs TWO (meson) or THREE (baryon)
    - Occupies ONE well of the three-well potential
    - Different overlap with V(σ), different ∫|χ|⁴
    """
    
    LEPTON_K = 1
    WELL_POSITIONS = [0.0, 2*np.pi/3, 4*np.pi/3]
    
    def __init__(
        self,
        grid: Optional[SpectralGrid] = None,
        potential: Optional[ThreeWellPotential] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        kappa: Optional[float] = None,
        g1: Optional[float] = None,
        g2: Optional[float] = None,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        use_physical: Optional[bool] = None,
    ):
        """Initialize with fundamental parameters."""
        if grid is None:
            grid = SpectralGrid(N=128)
        if potential is None:
            potential = ThreeWellPotential(V0=1.0, V1=0.1)
        
        self.grid = grid
        self.potential = potential
        
        if use_physical is None:
            use_physical = SFM_CONSTANTS.use_physical
        self.use_physical = use_physical
        
        self.beta = beta if beta is not None else SFM_CONSTANTS.beta_physical
        
        alpha_em = 1.0 / 137.036
        electron_mass_gev = 0.000511
        
        self.kappa = kappa if kappa is not None else (1.0 / (self.beta ** 2))
        self.alpha = alpha if alpha is not None else (0.5 * self.beta)
        self.g1 = g1 if g1 is not None else (alpha_em * self.beta / electron_mass_gev)
        self.g2 = g2 if g2 is not None else SFM_CONSTANTS.g2_alpha
        
        self.m_eff = m_eff
        self.hbar = hbar
        
        # Create WAVEFUNCTION-BASED minimizer
        self.minimizer = WavefunctionEnergyMinimizer(
            grid=grid,
            potential=potential,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            g1=self.g1,
            g2=self.g2,
            m_eff=m_eff,
            hbar=hbar,
        )
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
    
    def _initialize_lepton_wavefunction(
        self,
        n_spatial: int,
        initial_amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Initialize lepton as ONE-PEAK wavefunction with RADIAL NODES.
        
        Different spatial modes n create different wavefunction structures:
        - n=1 (electron): Ground state, no radial nodes
        - n=2 (muon): First excitation, 1 radial node
        - n=3 (tau): Second excitation, 2 radial nodes
        
        The nodes create DIFFERENT integrals:
        - Different ∫V(σ)|χ|² (node positions vs potential structure)
        - Different ∫|χ|⁴ (nodes reduce overlap → smaller integral)
        - Different ∫|∇χ|² (more nodes → higher kinetic energy)
        
        This is the physics of the lepton mass hierarchy!
        """
        sigma = self.grid.sigma
        
        # Lepton at first well
        well_pos = self.WELL_POSITIONS[0]
        
        # Width narrows with generation (more localized)
        width = 0.5 / n_spatial
        
        # Gaussian envelope at ONE well
        dist = np.angle(np.exp(1j * (sigma - well_pos)))
        envelope = np.exp(-0.5 * (dist / width)**2)
        
        # Add radial nodes for excited states!
        # This is crucial - nodes create different overlap integrals
        if n_spatial >= 2:
            # Hermite polynomial structure (nodes)
            # n=2: one node at center → H₁(x) ~ x
            # n=3: two nodes → H₂(x) ~ x² - 1
            x = dist / width
            if n_spatial == 2:
                radial_structure = x  # One node at center
            elif n_spatial == 3:
                radial_structure = x**2 - 1.0  # Two nodes
            else:
                radial_structure = np.polynomial.hermite.hermval(x, [0]*(n_spatial-1) + [1])
            envelope = envelope * radial_structure
        
        # k=1 winding
        winding = np.exp(1j * self.LEPTON_K * sigma)
        
        chi = envelope * winding
        
        # Scale to initial amplitude
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        if current_amp_sq > 1e-10:
            chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def solve_lepton(
        self,
        particle: str = 'electron',
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> SFMLeptonState:
        """
        Solve for lepton using wavefunction-based minimization.
        
        The ONE-PEAK structure creates different physics than hadrons:
        - Occupies ONE well only
        - Different ∫V|χ|² (sees only one well minimum)
        - Different ∫|χ|⁴ (single concentrated peak)
        """
        n_spatial = LEPTON_SPATIAL_MODE.get(particle, 1)
        
        if verbose:
            print("=" * 60)
            print(f"LEPTON SOLVER (Wavefunction-Based): {particle.upper()}")
            print(f"  ONE-PEAK structure at ONE well")
            print(f"  k = {self.LEPTON_K}, n_spatial = {n_spatial}")
            print(f"  Energy from ACTUAL INTEGRALS over χ")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}, κ={self.kappa:.6g}")
            print("=" * 60)
        
        # Initialize ONE-PEAK lepton wavefunction
        chi_initial = self._initialize_lepton_wavefunction(n_spatial, initial_amplitude)
        
        if verbose:
            k_eff_init = self.minimizer.compute_k_eff(chi_initial)
            print(f"  Initial k_eff: {k_eff_init:.4f}")
        
        # Minimize using WAVEFUNCTION-BASED minimizer
        result = self.minimizer.minimize(
            chi_initial=chi_initial,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        
        if verbose:
            mass_mev = self.beta * result.A_squared * 1000
            print(f"\n  Results:")
            print(f"    A² = {result.A_squared:.6f}")
            print(f"    k_eff = {result.k_eff:.4f} (EMERGENT)")
            print(f"    Δx = {result.delta_x:.6g}")
            print(f"    Δσ = {result.delta_sigma:.4f} (EMERGENT)")
            print(f"    Mass = {mass_mev:.2f} MeV")
            print(f"    E_potential = {result.energy.E_potential:.4f} (from ∫V|χ|²)")
            print(f"    E_nonlinear = {result.energy.E_nonlinear:.4f} (from ∫|χ|⁴)")
            print(f"    Converged: {result.converged}")
        
        return SFMLeptonState(
            chi=result.chi,
            particle=particle,
            amplitude=result.A,
            amplitude_squared=result.A_squared,
            delta_x=result.delta_x,
            delta_sigma=result.delta_sigma,
            energy_total=result.energy.E_total,
            energy_subspace=result.energy.E_subspace,
            energy_spatial=result.energy.E_spatial,
            energy_coupling=result.energy.E_coupling,
            energy_curvature=result.energy.E_curvature,
            energy_kinetic=result.energy.E_kinetic,
            energy_potential=result.energy.E_potential,
            energy_nonlinear=result.energy.E_nonlinear,
            energy_circulation=result.energy.E_circulation,
            k=self.LEPTON_K,
            k_eff=result.k_eff,
            n_spatial=n_spatial,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=result.final_residual,
        )
    
    def solve_electron(self, **kwargs) -> SFMLeptonState:
        return self.solve_lepton(particle='electron', **kwargs)
    
    def solve_muon(self, **kwargs) -> SFMLeptonState:
        return self.solve_lepton(particle='muon', **kwargs)
    
    def solve_tau(self, **kwargs) -> SFMLeptonState:
        return self.solve_lepton(particle='tau', **kwargs)
    
    def solve_lepton_spectrum(
        self,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, SFMLeptonState]:
        """Solve for all three leptons."""
        results = {}
        for particle in ['electron', 'muon', 'tau']:
            results[particle] = self.solve_lepton(particle=particle, verbose=verbose, **kwargs)
        return results
    
    def compute_mass_ratios(
        self,
        results: Dict[str, SFMLeptonState]
    ) -> Dict[str, float]:
        """Compute mass ratios."""
        e_A2 = results['electron'].amplitude_squared
        mu_A2 = results['muon'].amplitude_squared
        tau_A2 = results['tau'].amplitude_squared
        
        return {
            'mu_e': mu_A2 / e_A2 if e_A2 > 1e-10 else 0.0,
            'tau_e': tau_A2 / e_A2 if e_A2 > 1e-10 else 0.0,
            'tau_mu': tau_A2 / mu_A2 if mu_A2 > 1e-10 else 0.0,
        }


def solve_lepton_masses(verbose: bool = True) -> Dict[str, float]:
    """Convenience function."""
    solver = SFMLeptonSolver()
    results = solver.solve_lepton_spectrum(verbose=verbose)
    ratios = solver.compute_mass_ratios(results)
    
    return {
        'A2_e': results['electron'].amplitude_squared,
        'A2_mu': results['muon'].amplitude_squared,
        'A2_tau': results['tau'].amplitude_squared,
        'm_mu/m_e': ratios['mu_e'],
        'm_tau/m_e': ratios['tau_e'],
        'm_tau/m_mu': ratios['tau_mu'],
    }
