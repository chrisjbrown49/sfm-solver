"""
Composite Meson Solver for SFM - Pure First-Principles Implementation.

CRITICAL PHYSICS:
- Mesons are QUARK-ANTIQUARK composites
- Quantum numbers: k_net = k_quark + k_antiquark
- n = generation (1 for pion, 2 for charm, 3 for bottom)
- Binding from NONLINEAR term (∫|χ_q+χ_q̄|⁴)
- Different generations have different effective coupling
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential


# Quark winding numbers
QUARK_WINDING = {
    'u': +2, 'd': -1,
    'c': +2, 's': -1,
    'b': +2, 't': -1,
}

QUARK_GENERATION = {
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
    
    # Input quantum numbers
    n_spatial: int  # Generation
    k_net: int      # Net winding (charge related)
    
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
    k_eff: float  # For compatibility
    
    # Generation
    generation: int
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeMesonSolver:
    """
    Meson solver using quantum numbers as inputs.
    
    For mesons:
    - n = generation (determines effective coupling)
    - k_net = k_quark - k_antiquark (antiquark has opposite sign)
    - Binding from NONLINEAR term: 2-body interference
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
        """Get winding numbers. Antiquarks have opposite sign."""
        k_q = QUARK_WINDING.get(quark, 0)
        antiquark_base = antiquark.replace('_bar', '')
        k_qbar = -QUARK_WINDING.get(antiquark_base, 0)
        k_net = k_q + k_qbar
        return k_q, k_qbar, k_net
    
    def _initialize_meson(
        self,
        quark: str,
        antiquark: str,
        initial_amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Initialize meson as TWO-PEAK composite wavefunction.
        """
        sigma = self.grid.sigma
        
        k_q, k_qbar, _ = self._get_quark_windings(quark, antiquark)
        width = 0.4
        
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
        
        chi = chi_q + chi_qbar
        
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        if current_amp_sq > 1e-10:
            chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def solve(
        self,
        meson_type: str = 'pion_plus',
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 0.01,
        verbose: bool = False
    ) -> CompositeMesonState:
        """
        Solve for meson with quantum numbers as inputs.
        """
        config = MESON_CONFIGS.get(meson_type, MESON_CONFIGS['pion_plus'])
        quark = config['quark']
        antiquark = config['antiquark']
        
        k_q, k_qbar, k_net = self._get_quark_windings(quark, antiquark)
        generation = max(QUARK_GENERATION.get(quark, 1),
                        QUARK_GENERATION.get(antiquark, 1))
        
        # n_spatial = generation for mesons
        n_spatial = generation
        
        if verbose:
            print("=" * 60)
            print(f"MESON SOLVER: {meson_type.upper()}")
            print(f"  Quarks: {quark} (k={k_q}), {antiquark}-bar (k={k_qbar})")
            print(f"  Quantum numbers: n={n_spatial}, k_net={k_net}")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}")
            print("=" * 60)
        
        # Initialize TWO-PEAK composite wavefunction
        chi_initial = self._initialize_meson(quark, antiquark, initial_amplitude)
        
        # Minimize - is_lepton=False for composites
        result = self.minimizer.minimize(
            chi_initial=chi_initial,
            n_spatial=n_spatial,
            k_winding=max(abs(k_net), 1),  # At least 1 for coupling
            is_lepton=False,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        
        if verbose:
            mass_mev = self.beta * result.A_squared * 1000
            print(f"\n  Results:")
            print(f"    A² = {result.A_squared:.6g}")
            print(f"    Mass = {mass_mev:.2f} MeV (exp: {config['mass_mev']:.2f})")
            print(f"    Converged: {result.converged}")
        
        return CompositeMesonState(
            chi_meson=result.chi,
            meson_type=meson_type,
            quark=quark,
            antiquark=antiquark,
            n_spatial=n_spatial,
            k_net=k_net,
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
            k_eff=float(abs(k_net)),
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
