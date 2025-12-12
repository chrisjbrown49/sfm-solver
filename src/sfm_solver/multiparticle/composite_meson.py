"""
Composite Meson Solver for SFM - Pure First-Principles Implementation.

CRITICAL: Uses WAVEFUNCTION-BASED energy minimization.

The actual wavefunction structure matters:
- TWO peaks (quark + antiquark) with different windings
- Actual ∫V(σ)|χ|² captures two-well interaction
- Actual ∫|χ|⁴ captures nonlinear self-interaction with interference
- k_eff emerges from ∫|∂χ/∂σ|² (captures destructive interference!)

This is fundamentally different from baryons (3 peaks) and leptons (1 peak)!
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential


# Quark winding numbers (SIGNED)
QUARK_WINDING = {
    'u': +5, 'd': -3,
    'c': +5, 's': -3,
    'b': +5, 't': -3,
}

QUARK_SPATIAL_MODE = {
    'u': 1, 'd': 1,
    'c': 2, 's': 2,
    'b': 3, 't': 3,
}

MESON_CONFIGS = {
    'pion_plus': {'quark': 'u', 'antiquark': 'd', 'mass_mev': 139.57},
    'pion_zero': {'quark': 'u', 'antiquark': 'u', 'mass_mev': 134.98},
    'pion_minus': {'quark': 'd', 'antiquark': 'u', 'mass_mev': 139.57},
    'kaon_plus': {'quark': 'u', 'antiquark': 's', 'mass_mev': 493.68},
    'jpsi': {'quark': 'c', 'antiquark': 'c', 'mass_mev': 3096.90},
    'psi_2s': {'quark': 'c', 'antiquark': 'c', 'mass_mev': 3686.10},
    'upsilon_1s': {'quark': 'b', 'antiquark': 'b', 'mass_mev': 9460.30},
    'upsilon_2s': {'quark': 'b', 'antiquark': 'b', 'mass_mev': 10023.30},
}


@dataclass
class CompositeMesonState:
    """Result of composite meson solver."""
    chi_meson: NDArray[np.complexfloating]
    meson_type: str
    quark: str
    antiquark: str
    
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
    
    # Winding structure
    k_quark: int
    k_antiquark: int
    k_net: int
    k_eff: float  # EMERGENT from wavefunction
    
    # Generation
    generation: int
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeMesonSolver:
    """
    Meson solver using WAVEFUNCTION-BASED energy minimization.
    
    The key difference from baryons:
    - TWO peaks (q + q̄) vs THREE peaks
    - Different winding interference pattern
    - Different overlap with three-well potential
    """
    
    WELL_POSITIONS = [0.0, 2*np.pi/3, 4*np.pi/3]
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
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
    
    def _get_quark_windings(self, quark: str, antiquark: str) -> Tuple[int, int, int]:
        """Get signed winding numbers. Antiquarks have opposite sign."""
        k_q = QUARK_WINDING.get(quark, -3)
        antiquark_base = antiquark.replace('_bar', '')
        k_qbar = -QUARK_WINDING.get(antiquark_base, -3)  # Opposite sign
        k_net = k_q + k_qbar
        return k_q, k_qbar, k_net
    
    def _initialize_meson(
        self,
        quark: str,
        antiquark: str,
        generation: int,
        initial_amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Initialize meson as TWO-PEAK composite wavefunction.
        
        Different generations have different spatial structures:
        - Gen 1 (pion): Wide peaks, strong overlap
        - Gen 2 (charm): Narrower peaks, less overlap
        - Gen 3 (bottom): Narrowest peaks, minimal overlap
        
        This creates DIFFERENT integrals for heavy quarkonia!
        """
        sigma = self.grid.sigma
        
        k_q, k_qbar, _ = self._get_quark_windings(quark, antiquark)
        
        # Width DECREASES with generation (heavier quarks more localized)
        # This is physically motivated: heavier quarks have smaller Compton wavelengths
        width = 0.5 / generation
        
        # Quark at well 0
        well_q = self.WELL_POSITIONS[0]
        dist_q = np.angle(np.exp(1j * (sigma - well_q)))
        envelope_q = np.exp(-0.5 * (dist_q / width)**2)
        chi_q = envelope_q * np.exp(1j * k_q * sigma)
        
        # Antiquark at well 1
        well_qbar = self.WELL_POSITIONS[1]
        dist_qbar = np.angle(np.exp(1j * (sigma - well_qbar)))
        envelope_qbar = np.exp(-0.5 * (dist_qbar / width)**2)
        chi_qbar = envelope_qbar * np.exp(1j * k_qbar * sigma)
        
        # TWO-PEAK composite
        chi = chi_q + chi_qbar
        
        # Scale to initial amplitude
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        if current_amp_sq > 1e-10:
            chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def solve(
        self,
        meson_type: str = 'pion_plus',
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> CompositeMesonState:
        """
        Solve for meson using wavefunction-based minimization.
        
        The TWO-PEAK structure creates different physics than baryons:
        - Occupies TWO wells, not three
        - Different ∫V|χ|² due to different overlap
        - Different ∫|χ|⁴ due to two-lobe structure
        """
        config = MESON_CONFIGS.get(meson_type, MESON_CONFIGS['pion_plus'])
        quark = config['quark']
        antiquark = config['antiquark']
        
        k_q, k_qbar, k_net = self._get_quark_windings(quark, antiquark)
        generation = max(QUARK_SPATIAL_MODE.get(quark, 1),
                        QUARK_SPATIAL_MODE.get(antiquark, 1))
        
        if verbose:
            print("=" * 60)
            print(f"MESON SOLVER (Wavefunction-Based): {meson_type.upper()}")
            print(f"  TWO-PEAK structure (q at well 0, q̄ at well 1)")
            print(f"  Quark: {quark} (k={k_q}), Antiquark: {antiquark}-bar (k={k_qbar})")
            print(f"  k_net = {k_net}, generation = {generation}")
            print(f"  Energy from ACTUAL INTEGRALS over χ")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}, κ={self.kappa:.6g}")
            print("=" * 60)
        
        # Initialize TWO-PEAK meson wavefunction with generation-dependent width
        chi_initial = self._initialize_meson(quark, antiquark, generation, initial_amplitude)
        
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
            print(f"    Mass = {mass_mev:.2f} MeV (exp: {config['mass_mev']:.2f})")
            print(f"    E_potential = {result.energy.E_potential:.4f} (from ∫V|χ|²)")
            print(f"    E_nonlinear = {result.energy.E_nonlinear:.4f} (from ∫|χ|⁴)")
            print(f"    Converged: {result.converged}")
        
        return CompositeMesonState(
            chi_meson=result.chi,
            meson_type=meson_type,
            quark=quark,
            antiquark=antiquark,
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
            k_quark=k_q,
            k_antiquark=k_qbar,
            k_net=k_net,
            k_eff=result.k_eff,
            generation=generation,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=result.final_residual,
        )
    
    def solve_pion(self, **kwargs) -> CompositeMesonState:
        return self.solve(meson_type='pion_plus', **kwargs)
    
    def solve_jpsi(self, **kwargs) -> CompositeMesonState:
        return self.solve(meson_type='jpsi', **kwargs)
    
    def solve_upsilon(self, **kwargs) -> CompositeMesonState:
        return self.solve(meson_type='upsilon_1s', **kwargs)
