"""
Composite Baryon Solver for SFM - Pure First-Principles Implementation.

CRITICAL: Uses WAVEFUNCTION-BASED energy minimization.

The actual wavefunction structure matters:
- Three peaks at wells with color phases
- Actual ∫V(σ)|χ|² captures three-well interaction
- Actual ∫|χ|⁴ captures nonlinear self-interaction
- k_eff emerges from ∫|∂χ/∂σ|²

This is fundamentally different from parameterized approximations!
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
    'u': +5,
    'd': -3,
}

PROTON_QUARKS = ['u', 'u', 'd']
NEUTRON_QUARKS = ['u', 'd', 'd']


@dataclass
class CompositeBaryonState:
    """Result of composite baryon solver."""
    chi_baryon: NDArray[np.complexfloating]
    quark_types: Tuple[str, str, str]
    
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
    
    # Emergent k_eff
    k_eff: float
    
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
    Baryon solver using WAVEFUNCTION-BASED energy minimization.
    
    The key difference: energy is computed from ACTUAL INTEGRALS
    over the three-peak baryon wavefunction, not approximations.
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
        
        # Get/derive fundamental parameters
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
    
    def _initialize_baryon(
        self,
        quark_types: List[str],
        initial_amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Initialize baryon wavefunction with THREE peaks at wells.
        
        This creates the distinctive baryon structure:
        - Peak at each well (σ = 0, 2π/3, 4π/3)
        - Each peak has quark-specific winding
        - Color phases (0, 2π/3, 4π/3) for neutrality
        """
        sigma = self.grid.sigma
        N = len(sigma)
        
        chi = np.zeros(N, dtype=complex)
        width = 0.5  # Initial width (structure will evolve)
        
        for i, (well_pos, quark) in enumerate(zip(self.WELL_POSITIONS, quark_types)):
            k = QUARK_WINDING.get(quark, 3)
            color_phase = i * 2 * np.pi / 3
            
            # Gaussian at well
            dist = np.angle(np.exp(1j * (sigma - well_pos)))
            envelope = np.exp(-0.5 * (dist / width)**2)
            
            # Winding + color phase
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
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> CompositeBaryonState:
        """
        Solve for baryon using wavefunction-based minimization.
        
        The THREE-PEAK structure is crucial - it interacts with
        the three-well potential in a specific way that determines
        the baryon mass!
        """
        if quark_types is None:
            quark_types = PROTON_QUARKS
        
        if len(quark_types) != 3:
            raise ValueError("quark_types must have exactly 3 elements")
        
        if verbose:
            print("=" * 60)
            print(f"BARYON SOLVER (Wavefunction-Based): {''.join(quark_types)}")
            print(f"  THREE-PEAK structure at well positions")
            print(f"  Energy from ACTUAL INTEGRALS over χ")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}, κ={self.kappa:.6g}")
            print("=" * 60)
        
        # Initialize THREE-PEAK baryon wavefunction
        chi_initial = self._initialize_baryon(quark_types, initial_amplitude)
        
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
        
        # Extract color structure
        phases, color_mag = self._extract_color_phases(result.chi)
        
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
        
        return CompositeBaryonState(
            chi_baryon=result.chi,
            quark_types=tuple(quark_types),
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
            k_eff=result.k_eff,
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
