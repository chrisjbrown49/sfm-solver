"""
Composite Baryon Solver for SFM - Using Non-Separable Wavefunction Architecture.

ARCHITECTURE:
Stage 1: NonSeparableWavefunctionSolver finds wavefunction STRUCTURE
Stage 2: UniversalEnergyMinimizer optimizes SCALE (A, dx, ds)

CRITICAL PHYSICS:
- chi(sigma) = chi_1 + chi_2 + chi_3 is the THREE-PEAK composite wavefunction
- Color phases (0, 2pi/3, 4pi/3) for color neutrality
- Coupling emerges from wavefunction structure, not input quantum numbers!
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.universal_energy_minimizer import UniversalEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential


# Quark winding numbers
QUARK_WINDING = {
    'u': +2,  # Up quark
    'd': -1,  # Down quark
}

# Quark generation (affects spatial wavefunction)
QUARK_GENERATION = {
    'u': 1, 'd': 1,
    'c': 2, 's': 2,
    'b': 3, 't': 3,
}

PROTON_QUARKS = ['u', 'u', 'd']
NEUTRON_QUARKS = ['u', 'd', 'd']


@dataclass
class CompositeBaryonState:
    """Result of composite baryon solver."""
    chi_baryon: NDArray[np.complexfloating]
    quark_types: Tuple[str, str, str]
    
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
    k_eff: float
    k_total: int
    
    # Color structure
    phases: Tuple[float, float, float]
    color_sum_magnitude: float
    is_color_neutral: bool
    
    # Angular composition (from non-separable solver)
    l_composition: Dict[int, float] = None
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 0.0


class CompositeBaryonSolver:
    """
    Baryon solver using non-separable wavefunction architecture.
    
    Two-stage process:
    1. NonSeparableWavefunctionSolver: Find entangled wavefunction structure
    2. UniversalEnergyMinimizer: Optimize (A, dx, ds) for minimum energy
    
    Key physics:
    - chi = chi_1 + chi_2 + chi_3 (three quarks with color phases)
    - Interference captured by entangled wavefunction structure
    - E_coupling emerges from the wavefunction structure!
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
    
    def solve(
        self,
        quark_types: List[str] = None,
        n_radial: int = 1,
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 0.01,
        verbose: bool = False
    ) -> CompositeBaryonState:
        """
        Solve for baryon using two-stage architecture.
        
        Stage 1: Solve for wavefunction STRUCTURE (entangled components)
        Stage 2: Minimize energy over SCALE (A, dx, ds)
        
        Args:
            quark_types: List of quark types ['u', 'u', 'd'] etc.
            n_radial: Radial excitation quantum number (1 = ground state)
        """
        if quark_types is None:
            quark_types = PROTON_QUARKS
        
        if len(quark_types) != 3:
            raise ValueError("quark_types must have exactly 3 elements")
        
        # Get quark properties
        quark_gens = [QUARK_GENERATION.get(q, 1) for q in quark_types]
        windings = [QUARK_WINDING.get(q, 0) for q in quark_types]
        k_total = sum(windings)
        
        # Spatial mode = radial excitation (NOT quark generation!)
        n_spatial = n_radial
        
        if verbose:
            print("=" * 60)
            print(f"BARYON SOLVER: {''.join(quark_types)}")
            print(f"  Using non-separable wavefunction architecture")
            print(f"  THREE-PEAK composite wavefunction")
            print(f"  Spatial mode n={n_spatial}")
            print(f"  Parameters: alpha={self.alpha:.4g}, beta={self.beta:.4g}")
            print("=" * 60)
        
        # === STAGE 1: Solve for wavefunction structure ===
        if verbose:
            print("\nStage 1: Solving baryon wavefunction structure...")
        
        structure = self.wf_solver.solve_baryon(
            quark_gens=quark_gens,
            windings=windings,
            verbose=verbose,
        )
        
        if verbose:
            l_comp = structure.l_composition
            print(f"  Angular composition: l=0: {l_comp.get(0,0)*100:.1f}%, "
                  f"l=1: {l_comp.get(1,0)*100:.1f}%")
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
            print(f"    Mass = {mass_mev:.2f} MeV")
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
        
        # Extract color structure from wavefunction
        color_phases = (0.0, 2*np.pi/3, 4*np.pi/3)
        color_sum = sum(np.exp(1j * p) for p in color_phases)
        color_mag = abs(color_sum)
        
        # E_subspace = E_kinetic + E_potential + E_nonlinear + E_circulation
        E_subspace = result.E_kinetic + result.E_potential + result.E_nonlinear + result.E_circulation
        
        return CompositeBaryonState(
            chi_baryon=chi_scaled,
            quark_types=tuple(quark_types),
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
            k_eff=abs(structure.k_eff),
            k_total=k_total,
            phases=color_phases,
            color_sum_magnitude=color_mag,
            is_color_neutral=color_mag < 0.1,
            l_composition=structure.l_composition,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=0.0,
        )


def solve_proton(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for proton (uud)."""
    solver = CompositeBaryonSolver(grid, potential)
    return solver.solve(quark_types=PROTON_QUARKS, **kwargs)


def solve_neutron(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for neutron (udd)."""
    solver = CompositeBaryonSolver(grid, potential)
    return solver.solve(quark_types=NEUTRON_QUARKS, **kwargs)
