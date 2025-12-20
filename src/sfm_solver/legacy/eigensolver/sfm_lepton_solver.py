"""
SFM Lepton Solver - Using Non-Separable Wavefunction Architecture.

ARCHITECTURE:
Stage 1: NonSeparableWavefunctionSolver finds wavefunction STRUCTURE
Stage 2: UniversalEnergyMinimizer optimizes SCALE (A, dx, ds)

CRITICAL PHYSICS:
- Wavefunction is NON-SEPARABLE: psi = Sum R_nl Y_lm chi_nlm(sigma)
- Each angular component (n,l,m) has its OWN subspace function
- Coupling emerges from cross-terms between l=0 and l=1 components
- Mass hierarchy from radial node structure in spatial wavefunction
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.universal_energy_minimizer import UniversalEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential


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
    
    # Angular composition (from non-separable solver)
    l_composition: Dict[int, float] = None
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 0.0


class SFMLeptonSolver:
    """
    Lepton solver using non-separable wavefunction architecture.
    
    Two-stage process:
    1. NonSeparableWavefunctionSolver: Find entangled wavefunction structure
    2. UniversalEnergyMinimizer: Optimize (A, dx, ds) for minimum energy
    
    Key physics:
    - phi_n(r) has n-1 radial nodes (more nodes -> larger gradient)
    - chi(sigma) has k=1 winding
    - Mass hierarchy EMERGES from the gradient structure!
    """
    
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
        self.V0 = potential.V0
        
        # Create non-separable wavefunction solver (Stage 1)
        self.wf_solver = NonSeparableWavefunctionSolver(
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            g1=self.g1,
            g2=self.g2,
            V0=self.V0,
            n_max=5,
            l_max=2,
            N_sigma=64,
        )
        
        # Create universal energy minimizer (Stage 2)
        self.minimizer = UniversalEnergyMinimizer(
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            g1=self.g1,
            g2=self.g2,
            V0=self.V0,
            m_eff=m_eff,
            hbar=hbar,
        )
    
    def solve_lepton(
        self,
        particle: str = 'electron',
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 0.01,
        verbose: bool = False
    ) -> SFMLeptonState:
        """
        Solve for lepton using two-stage architecture.
        
        Stage 1: Solve for wavefunction STRUCTURE (entangled components)
        Stage 2: Minimize energy over SCALE (A, dx, ds)
        
        Args:
            particle: 'electron', 'muon', or 'tau'
            max_iter: Maximum iterations for energy minimization
            tol: Convergence tolerance
            initial_amplitude: Starting amplitude for minimization
            verbose: Print progress
        """
        n_spatial = LEPTON_SPATIAL_MODE.get(particle, 1)
        k_winding = LEPTON_WINDING
        
        if verbose:
            print("=" * 60)
            print(f"LEPTON SOLVER: {particle.upper()}")
            print(f"  Using non-separable wavefunction architecture")
            print(f"  Spatial mode n={n_spatial} -> phi_{n_spatial}(r) has {n_spatial-1} nodes")
            print(f"  Parameters: alpha={self.alpha:.4g}, beta={self.beta:.4g}")
            print("=" * 60)
        
        # === STAGE 1: Solve for wavefunction structure ===
        if verbose:
            print("\nStage 1: Solving wavefunction structure...")
        
        structure = self.wf_solver.solve_lepton(
            n_target=n_spatial,
            k_winding=k_winding,
            verbose=verbose,
        )
        
        if verbose:
            print(f"  Angular composition: l=0: {structure.l_composition.get(0,0)*100:.1f}%, "
                  f"l=1: {structure.l_composition.get(1,0)*100:.1f}%")
            print(f"  k_eff from wavefunction: {structure.k_eff:.4f}")
        
        # === STAGE 2: Minimize energy over (A, dx, ds) ===
        if verbose:
            print("\nStage 2: Minimizing energy over (A, dx, ds)...")
        
        result = self.minimizer.minimize(
            structure=structure,
            sigma_grid=self.wf_solver.get_sigma_grid(),
            V_sigma=self.wf_solver.get_V_sigma(),
            spatial_coupling=self.wf_solver.get_spatial_coupling_matrix(),
            state_index_map=self.wf_solver.get_state_index_map(),
            initial_A=np.sqrt(initial_amplitude),
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        
        if verbose:
            mass_mev = self.beta * result.A_squared * 1000
            print(f"\n  Results:")
            print(f"    A^2 = {result.A_squared:.6g}")
            print(f"    Mass = {mass_mev:.4f} MeV")
            print(f"    E_coupling = {result.E_coupling:.4g}")
            print(f"    Converged: {result.converged}")
        
        # Extract primary chi component for compatibility
        primary_key = (n_spatial, 0, 0)
        chi_primary = structure.chi_components.get(
            primary_key, 
            np.zeros(self.wf_solver.basis.N_sigma, dtype=complex)
        )
        
        # Scale chi by converged amplitude
        chi_scaled = chi_primary * result.A
        
        # E_subspace = E_kinetic + E_potential + E_nonlinear + E_circulation
        E_subspace = result.E_kinetic + result.E_potential + result.E_nonlinear + result.E_circulation
        
        return SFMLeptonState(
            chi=chi_scaled,
            particle=particle,
            n_spatial=n_spatial,
            amplitude=result.A,
            amplitude_squared=result.A_squared,
            delta_x=result.delta_x,
            delta_sigma=result.delta_sigma,
            energy_total=result.E_total,
            energy_subspace=E_subspace,
            energy_spatial=result.E_spatial,
            energy_coupling=result.E_coupling,
            energy_curvature=result.E_curvature,
            energy_kinetic=result.E_kinetic,
            energy_potential=result.E_potential,
            energy_nonlinear=result.E_nonlinear,
            energy_circulation=result.E_circulation,
            spatial_factor=0.0,  # Now computed internally
            subspace_factor=structure.k_eff,
            k_eff=structure.k_eff,
            k_winding=k_winding,
            l_composition=structure.l_composition,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=0.0,
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
