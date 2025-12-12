"""
Composite Baryon Solver for SFM - Pure First-Principles Implementation.

CRITICAL PHYSICS:
- Baryons are THREE-QUARK composites
- Quantum numbers: k_total = sum of quark windings
- Binding comes primarily from NONLINEAR term (∫|χ₁+χ₂+χ₃|⁴)
- Three quarks with color phases create constructive interference
- Coupling is weaker than leptons (different physics)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer
from sfm_solver.potentials.three_well import ThreeWellPotential


# Quark winding numbers (SIGNED)
QUARK_WINDING = {
    'u': +2,  # Up quark: +2/3 charge → k = +2
    'd': -1,  # Down quark: -1/3 charge → k = -1
}

PROTON_QUARKS = ['u', 'u', 'd']  # k_total = 2+2-1 = 3
NEUTRON_QUARKS = ['u', 'd', 'd']  # k_total = 2-1-1 = 0


@dataclass
class CompositeBaryonState:
    """Result of composite baryon solver."""
    chi_baryon: NDArray[np.complexfloating]
    quark_types: Tuple[str, str, str]
    
    # Input quantum numbers
    n_spatial: int
    k_total: int
    
    # From minimizer
    amplitude: float
    amplitude_squared: float
    delta_x: float
    delta_sigma: float
    k_eff: float  # For compatibility
    
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
    
    # Color structure
    phases: Tuple[float, float, float]
    color_sum_magnitude: float
    is_color_neutral: bool
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeBaryonSolver:
    """
    Baryon solver using quantum numbers as inputs.
    
    For baryons (composite particles):
    - k_total = sum of quark windings (determines electric charge)
    - n_spatial = 1 for ground state (u,d quarks)
    - Binding from NONLINEAR term: constructive interference of 3 quarks
    - Coupling is weaker than leptons
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
    
    def _compute_k_total(self, quark_types: List[str]) -> int:
        """Compute net winding from quark content (for charge)."""
        return sum(QUARK_WINDING.get(q, 0) for q in quark_types)
    
    def _compute_k_coupling(self, quark_types: List[str]) -> int:
        """
        Compute coupling strength from quark content.
        
        For composites, the coupling depends on the SUM OF MAGNITUDES,
        not the net winding. This ensures neutral particles still couple!
        """
        return sum(abs(QUARK_WINDING.get(q, 0)) for q in quark_types)
    
    def _initialize_baryon(
        self,
        quark_types: List[str],
        initial_amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Initialize baryon as THREE-PEAK composite wavefunction.
        
        Three peaks at wells with:
        - Color phases (0, 2π/3, 4π/3) for color neutrality
        - Each peak has its quark's winding
        """
        sigma = self.grid.sigma
        N = len(sigma)
        
        chi = np.zeros(N, dtype=complex)
        width = 0.5
        
        for i, (well_pos, quark) in enumerate(zip(self.WELL_POSITIONS, quark_types)):
            k = QUARK_WINDING.get(quark, 0)
            color_phase = i * 2 * np.pi / 3
            
            dist = np.angle(np.exp(1j * (sigma - well_pos)))
            envelope = np.exp(-0.5 * (dist / width)**2)
            phase = k * sigma + color_phase
            chi += envelope * np.exp(1j * phase)
        
        # Scale to initial amplitude
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        if current_amp_sq > 1e-10:
            chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def _extract_color_phases(self, chi: NDArray[np.complexfloating]) -> Tuple[Tuple[float, ...], float]:
        """Extract phases at wells for color neutrality check."""
        sigma = self.grid.sigma
        phases = []
        
        for well_pos in self.WELL_POSITIONS:
            idx = np.argmin(np.abs(sigma - well_pos))
            phases.append(float(np.angle(chi[idx])))
        
        color_sum = sum(np.exp(1j * phi) for phi in phases)
        return tuple(phases), abs(color_sum)
    
    def solve(
        self,
        quark_types: List[str] = None,
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 0.01,
        verbose: bool = False
    ) -> CompositeBaryonState:
        """
        Solve for baryon with quark content as input.
        """
        if quark_types is None:
            quark_types = PROTON_QUARKS
        
        if len(quark_types) != 3:
            raise ValueError("quark_types must have exactly 3 elements")
        
        # Compute quantum numbers
        k_total = self._compute_k_total(quark_types)
        n_spatial = 1  # Ground state for u,d quarks
        
        if verbose:
            print("=" * 60)
            print(f"BARYON SOLVER: {''.join(quark_types)}")
            print(f"  Quantum numbers: n={n_spatial}, k_total={k_total}")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}")
            print("=" * 60)
        
        # Initialize THREE-PEAK composite wavefunction
        chi_initial = self._initialize_baryon(quark_types, initial_amplitude)
        
        # Minimize with (n, k) as inputs
        # is_lepton=False means weaker coupling, binding from nonlinear
        result = self.minimizer.minimize(
            chi_initial=chi_initial,
            n_spatial=n_spatial,
            k_winding=abs(k_total),  # Use magnitude for coupling strength
            is_lepton=False,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        
        # Extract color structure
        phases, color_mag = self._extract_color_phases(result.chi)
        
        if verbose:
            mass_mev = self.beta * result.A_squared * 1000
            print(f"\n  Results:")
            print(f"    A² = {result.A_squared:.6g}")
            print(f"    Mass = {mass_mev:.2f} MeV")
            print(f"    Converged: {result.converged}")
        
        return CompositeBaryonState(
            chi_baryon=result.chi,
            quark_types=tuple(quark_types),
            n_spatial=n_spatial,
            k_total=k_total,
            amplitude=result.A,
            amplitude_squared=result.A_squared,
            delta_x=result.delta_x,
            delta_sigma=result.delta_sigma,
            k_eff=float(abs(k_total)),
            energy_total=result.energy.E_total,
            energy_subspace=result.energy.E_subspace,
            energy_spatial=result.energy.E_spatial,
            energy_coupling=result.energy.E_coupling,
            energy_curvature=result.energy.E_curvature,
            energy_kinetic=result.energy.E_kinetic,
            energy_potential=result.energy.E_potential,
            energy_nonlinear=result.energy.E_nonlinear,
            energy_circulation=result.energy.E_circulation,
            phases=phases,
            color_sum_magnitude=color_mag,
            is_color_neutral=color_mag < 0.1,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=result.final_residual,
        )


def solve_proton(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for proton (uud)."""
    solver = CompositeBaryonSolver(grid, potential)
    return solver.solve(quark_types=PROTON_QUARKS, **kwargs)


def solve_neutron(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for neutron (udd)."""
    solver = CompositeBaryonSolver(grid, potential)
    return solver.solve(quark_types=NEUTRON_QUARKS, **kwargs)
