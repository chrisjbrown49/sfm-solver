"""
Composite Baryon Solver for SFM - Pure First-Principles Implementation.

CRITICAL PHYSICS:
- E_coupling computed from ACTUAL WAVEFUNCTIONS:
  E_coupling = -α ∫∫ (∇φ · ∂χ/∂σ) φ χ d³x dσ

- χ(σ) = χ₁ + χ₂ + χ₃ is the THREE-PEAK composite wavefunction
- ∂χ/∂σ captures the interference of all three quarks
- Coupling emerges from wavefunction structure, not input quantum numbers!
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer
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
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeBaryonSolver:
    """
    Baryon solver using actual wavefunctions for coupling.
    
    Key physics:
    - χ = χ₁ + χ₂ + χ₃ (three quarks with color phases)
    - ∂χ/∂σ captures the interference pattern
    - φ_n(r) has n-1 radial nodes based on quark generation
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
    
    def _initialize_baryon(
        self,
        quark_types: List[str],
        initial_amplitude: float = 0.01
    ) -> NDArray[np.complexfloating]:
        """
        Initialize baryon as THREE-PEAK composite wavefunction.
        
        Each quark χ_i contributes:
        - Its winding number k_i (encoded in phase)
        - Its radial structure f_n(σ) based on quark generation!
        - Color phase (0, 2π/3, 4π/3) for color neutrality
        
        CRITICAL: Heavier quarks (c, b) have more radial nodes in their
        subspace wavefunction, just like muon/tau have more nodes than electron!
        """
        sigma = self.grid.sigma
        N = len(sigma)
        
        chi = np.zeros(N, dtype=complex)
        width = 0.5
        
        for i, (well_pos, quark) in enumerate(zip(self.WELL_POSITIONS, quark_types)):
            k = QUARK_WINDING.get(quark, 0)
            n_quark = QUARK_GENERATION.get(quark, 1)  # Radial structure!
            color_phase = i * 2 * np.pi / 3
            
            # Distance from well (periodic)
            dist = np.angle(np.exp(1j * (sigma - well_pos)))
            
            # Gaussian envelope
            envelope = np.exp(-0.5 * (dist / width)**2)
            
            # Radial structure f_n based on quark generation (like leptons!)
            # Using SMOOTH OSCILLATORY envelopes that NEVER go to zero:
            # n=1: constant (no oscillation)
            # n=2: one oscillation period → larger gradients
            # n=3: two oscillation periods → even larger gradients
            # 
            # Form: f_n = 1 + modulation * cos((n-1) * pi * x / width)
            # This stays positive (between 0.5 and 1.5), smooth, differentiable
            x = dist / width
            modulation = 0.5  # Keeps envelope between 0.5 and 1.5
            if n_quark == 1:
                radial = np.ones_like(dist)  # No oscillation
            elif n_quark == 2:
                radial = 1.0 + modulation * np.cos(np.pi * x)  # One period
            else:  # n_quark >= 3
                radial = 1.0 + modulation * np.cos(2 * np.pi * x)  # Two periods
            
            # Full quark wavefunction: envelope × radial × winding × color
            phase = k * sigma + color_phase
            chi_quark = envelope * radial * np.exp(1j * phase)
            chi += chi_quark
        
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
        n_radial: int = 1,
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 0.01,
        verbose: bool = False
    ) -> CompositeBaryonState:
        """
        Solve for baryon.
        
        The composite wavefunction χ = χ₁ + χ₂ + χ₃ captures all the physics:
        - Each quark's winding is encoded in exp(i k σ)
        - Color phases create interference
        - ∂χ/∂σ naturally captures the effective winding
        
        Args:
            quark_types: List of quark types ['u', 'u', 'd'] etc.
            n_radial: Radial excitation quantum number (1 = ground state)
                      This is INDEPENDENT of quark generation!
        """
        if quark_types is None:
            quark_types = PROTON_QUARKS
        
        if len(quark_types) != 3:
            raise ValueError("quark_types must have exactly 3 elements")
        
        # Spatial mode = radial excitation (NOT quark generation!)
        # Ground state baryons (proton, neutron, etc.) have n_spatial = 1
        # Excited baryons (resonances) have n_spatial = 2, 3, etc.
        n_spatial = n_radial
        k_total = sum(QUARK_WINDING.get(q, 0) for q in quark_types)
        
        if verbose:
            print("=" * 60)
            print(f"BARYON SOLVER: {''.join(quark_types)}")
            print(f"  THREE-PEAK composite wavefunction")
            print(f"  Spatial mode n={n_spatial} (from quark generations)")
            print(f"  E_coupling from actual wavefunction integrals")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}")
            print("=" * 60)
        
        # Initialize THREE-PEAK composite wavefunction
        chi_initial = self._initialize_baryon(quark_types, initial_amplitude)
        
        # Minimize
        result = self.minimizer.minimize(
            chi_initial=chi_initial,
            n_spatial=n_spatial,
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
            print(f"    spatial_factor = {result.energy.spatial_factor:.4g}")
            print(f"    subspace_factor = {result.energy.subspace_factor:.4g}")
            print(f"    Converged: {result.converged}")
        
        return CompositeBaryonState(
            chi_baryon=result.chi,
            quark_types=tuple(quark_types),
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
            k_eff=abs(result.energy.subspace_factor),
            k_total=k_total,
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
