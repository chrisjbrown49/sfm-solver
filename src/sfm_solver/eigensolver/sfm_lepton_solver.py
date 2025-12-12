"""
SFM Lepton Solver - Pure First-Principles Implementation.

CRITICAL PHYSICS:
- E_coupling computed from ACTUAL WAVEFUNCTIONS:
  E_coupling = -α ∫∫ (∇φ · ∂χ/∂σ) φ χ d³x dσ

- Spatial wavefunction φ_n(r) has n-1 radial nodes:
  * Electron (n=1): no nodes, small gradient
  * Muon (n=2): 1 node, larger gradient  
  * Tau (n=3): 2 nodes, largest gradient

- Mass hierarchy EMERGES from the gradient structure!
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


# Spatial mode for each lepton (determines radial node structure)
LEPTON_SPATIAL_MODE = {
    'electron': 1,
    'muon': 2,
    'tau': 3,
}

# For backward compatibility
LEPTON_WINDING = 1


@dataclass
class SFMLeptonState:
    """Result of SFM lepton solver."""
    chi: NDArray[np.complexfloating]
    particle: str
    
    # Spatial mode
    n_spatial: int
    
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
    
    # Coupling components
    spatial_factor: float
    subspace_factor: float
    
    # For compatibility
    k_eff: float = 1.0
    k_winding: int = 1
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 0.0


class SFMLeptonSolver:
    """
    Lepton solver using actual wavefunctions for coupling.
    
    Key physics:
    - φ_n(r) has n-1 radial nodes (more nodes → larger gradient)
    - χ(σ) is a single peak with k=1 winding
    - E_coupling = -α × [∫(dφ/dr)φ r²dr] × [Re∫χ*(∂χ/∂σ)dσ]
    - Mass hierarchy emerges from radial node structure!
    """
    
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
        
        # Create minimizer
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
        initial_amplitude: float = 0.01
    ) -> NDArray[np.complexfloating]:
        """
        Initialize lepton as ONE-PEAK wavefunction with k=1 winding.
        
        The subspace structure is the same for all leptons - what differs
        is the SPATIAL wavefunction φ_n(r) which has different radial nodes.
        """
        sigma = self.grid.sigma
        
        well_pos = self.WELL_POSITIONS[0]
        width = 0.5
        
        # Gaussian at one well
        dist = np.angle(np.exp(1j * (sigma - well_pos)))
        envelope = np.exp(-0.5 * (dist / width)**2)
        
        # k=1 winding
        winding = np.exp(1j * sigma)
        
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
        initial_amplitude: float = 0.01,
        verbose: bool = False
    ) -> SFMLeptonState:
        """
        Solve for lepton.
        
        The spatial mode n determines the radial wavefunction φ_n(r):
        - electron: n=1, no radial nodes
        - muon: n=2, 1 radial node
        - tau: n=3, 2 radial nodes
        
        Different radial structures → different coupling → different mass!
        """
        n_spatial = LEPTON_SPATIAL_MODE.get(particle, 1)
        
        if verbose:
            print("=" * 60)
            print(f"LEPTON SOLVER: {particle.upper()}")
            print(f"  Spatial mode n={n_spatial} → φ_{n_spatial}(r) has {n_spatial-1} nodes")
            print(f"  E_coupling from actual wavefunction integrals")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}")
            print("=" * 60)
        
        # Initialize subspace wavefunction
        chi_initial = self._initialize_lepton_wavefunction(initial_amplitude)
        
        # Minimize - spatial wavefunction created internally based on n_spatial
        result = self.minimizer.minimize(
            chi_initial=chi_initial,
            n_spatial=n_spatial,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        
        if verbose:
            mass_mev = self.beta * result.A_squared * 1000
            print(f"\n  Results:")
            print(f"    A² = {result.A_squared:.6g}")
            print(f"    Mass = {mass_mev:.4f} MeV")
            print(f"    spatial_factor = {result.energy.spatial_factor:.4g}")
            print(f"    subspace_factor = {result.energy.subspace_factor:.4g}")
            print(f"    E_coupling = {result.energy.E_coupling:.4g}")
            print(f"    Converged: {result.converged}")
        
        return SFMLeptonState(
            chi=result.chi,
            particle=particle,
            n_spatial=n_spatial,
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
            spatial_factor=result.energy.spatial_factor,
            subspace_factor=result.energy.subspace_factor,
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
    """Convenience function to solve for all leptons."""
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
