"""
Composite Meson Solver for SFM - Using Non-Separable Wavefunction Architecture.

ARCHITECTURE:
Stage 1: NonSeparableWavefunctionSolver finds wavefunction STRUCTURE
Stage 2: UniversalEnergyMinimizer optimizes SCALE (A, dx, ds)

CRITICAL PHYSICS:
- chi(sigma) = chi_q + chi_qbar is the TWO-PEAK composite wavefunction
- For neutral mesons (q qbar with opposite winding):
  * Interference captured by entangled wavefunction structure
  * EMERGES from wavefunction, not imposed!
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.universal_energy_minimizer import UniversalEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential


# Quark winding numbers
QUARK_WINDING = {
    'u': +2, 'd': -1,
    'c': +2, 's': -1,
    'b': +2, 't': -1,
}

# Quark generation
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
    
    # Spatial mode
    n_spatial: int
    generation: int
    
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
    
    # Winding structure
    k_quark: int
    k_antiquark: int
    k_net: int
    k_eff: float
    
    # Angular composition (from non-separable solver)
    l_composition: Dict[int, float] = None
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 0.0


class CompositeMesonSolver:
    """
    Meson solver using non-separable wavefunction architecture.
    
    Two-stage process:
    1. NonSeparableWavefunctionSolver: Find entangled wavefunction structure
    2. UniversalEnergyMinimizer: Optimize (A, dx, ds) for minimum energy
    
    Key physics:
    - chi = chi_q + chi_qbar (quark + antiquark with different windings)
    - Antiquark has OPPOSITE winding from quark
    - Interference captured by entangled wavefunction structure
    """
    
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
    
    def _get_quark_windings(self, quark: str, antiquark: str) -> Tuple[int, int, int]:
        """Get winding numbers. Antiquarks have OPPOSITE sign."""
        k_q = QUARK_WINDING.get(quark, 0)
        antiquark_base = antiquark.replace('_bar', '')
        k_qbar = -QUARK_WINDING.get(antiquark_base, 0)  # Opposite!
        k_net = k_q + k_qbar
        return k_q, k_qbar, k_net
    
    def solve(
        self,
        meson_type: str = 'pion_plus',
        n_radial: int = None,
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 0.01,
        verbose: bool = False
    ) -> CompositeMesonState:
        """
        Solve for meson using two-stage architecture.
        
        Stage 1: Solve for wavefunction STRUCTURE (entangled components)
        Stage 2: Minimize energy over SCALE (A, dx, ds)
        
        Args:
            meson_type: Type of meson ('pion_plus', 'jpsi', etc.)
            n_radial: Radial excitation (1 = ground state, 2 = first excited)
                      If None, determined from meson_type (1S vs 2S)
        """
        config = MESON_CONFIGS.get(meson_type, MESON_CONFIGS['pion_plus'])
        quark = config['quark']
        antiquark = config['antiquark']
        
        k_q, k_qbar, k_net = self._get_quark_windings(quark, antiquark)
        quark_gen = QUARK_GENERATION.get(quark, 1)
        antiquark_gen = QUARK_GENERATION.get(antiquark, 1)
        generation = max(quark_gen, antiquark_gen)
        
        # Spatial mode = radial excitation (NOT quark generation!)
        if n_radial is not None:
            n_spatial = n_radial
        elif '2s' in meson_type.lower():
            n_spatial = 2  # Radially excited state
        else:
            n_spatial = 1  # Ground state
        
        if verbose:
            print("=" * 60)
            print(f"MESON SOLVER: {meson_type.upper()}")
            print(f"  Using non-separable wavefunction architecture")
            print(f"  TWO-PEAK composite: {quark}(k={k_q}) + {antiquark}-bar(k={k_qbar})")
            print(f"  Spatial mode n={n_spatial} (generation {generation})")
            print(f"  Parameters: alpha={self.alpha:.4g}, beta={self.beta:.4g}")
            print("=" * 60)
        
        # === STAGE 1: Solve for wavefunction structure ===
        if verbose:
            print("\nStage 1: Solving meson wavefunction structure...")
        
        structure = self.wf_solver.solve_meson(
            quark_gen=quark_gen,
            antiquark_gen=antiquark_gen,
            k_quark=k_q,
            k_antiquark=k_qbar,
            n_radial=n_spatial,
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
            print(f"    Mass = {mass_mev:.2f} MeV (exp: {config['mass_mev']:.2f})")
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
        
        return CompositeMesonState(
            chi_meson=chi_scaled,
            meson_type=meson_type,
            quark=quark,
            antiquark=antiquark,
            n_spatial=n_spatial,
            generation=generation,
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
            k_quark=k_q,
            k_antiquark=k_qbar,
            k_net=k_net,
            k_eff=abs(structure.k_eff),
            l_composition=structure.l_composition,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=0.0,
        )
    
    def solve_pion(self, **kwargs) -> CompositeMesonState:
        return self.solve(meson_type='pion_plus', **kwargs)
    
    def solve_jpsi(self, **kwargs) -> CompositeMesonState:
        return self.solve(meson_type='jpsi', **kwargs)
    
    def solve_upsilon(self, **kwargs) -> CompositeMesonState:
        return self.solve(meson_type='upsilon_1s', **kwargs)
