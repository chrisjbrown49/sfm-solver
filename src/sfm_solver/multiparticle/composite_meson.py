"""
Composite Meson Solver for SFM - Pure First-Principles Implementation.

CRITICAL REQUIREMENTS (from Tier2-2b_Hadron_Solvers_Fix_Plan.md):
=================================================================
A. ALL predictions must EMERGE from first principles of the Single-Field Model.
   NO phenomenological parameters are permitted.

B. Uses ONLY fundamental parameters derived from first principles:
   - β, κ = 1/β², α = C × β, g₁, g₂, V₀
   These are shared across ALL solvers (lepton, baryon, meson).

C. Uses the UNIVERSAL energy minimizer to minimize E_total(A, Δx, Δσ).

PHYSICS:
========
- Single composite wavefunction: χ = χ_q + χ_qbar
- Quark and antiquark have different windings creating interference
- k_eff emerges from the composite wavefunction gradient
- Mass from m = β × A²

REMOVED PHENOMENOLOGICAL PARAMETERS:
===================================
- A_LEPTON, B_LEPTON (fitted exponents)
- DELTA_X_EXPONENT (WKB)
- C_RAD, GENERATION_DILUTION (calibrated)
- spatial_mode_power, DELTA_V_POWER (tuned)
- MESON_OVERLAP_FACTOR, GEN2_BOOST, GEN23_POWER (fitted)
- EM_MASS_FRACTION, P_COUPLING (calibrated)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.core.energy_minimizer import UniversalEnergyMinimizer, MinimizationResult
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


# Quark winding numbers (SIGNED)
QUARK_WINDING = {
    'u': +5, 'd': -3,
    'c': +5, 's': -3,
    'b': +5, 't': -3,
}

# Quark spatial mode (generation)
QUARK_SPATIAL_MODE = {
    'u': 1, 'd': 1,  # Generation 1
    'c': 2, 's': 2,  # Generation 2
    'b': 3, 't': 3,  # Generation 3
}

# Meson configurations
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
    
    # From universal minimizer
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
    k_net: int        # Net winding (for charge)
    k_eff: float      # EMERGENT from wavefunction
    
    # Generation
    generation: int
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeMesonSolver:
    """
    Pure first-principles meson solver using universal energy minimizer.
    
    NO PHENOMENOLOGICAL PARAMETERS - all mass ratios must EMERGE.
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
        """
        Initialize with ONLY fundamental parameters.
        
        NO solver-specific calibration allowed!
        """
        self.grid = grid
        self.potential = potential
        
        if use_physical is None:
            use_physical = SFM_CONSTANTS.use_physical
        self.use_physical = use_physical
        
        # Get fundamental parameters
        self.beta = beta if beta is not None else SFM_CONSTANTS.beta_physical
        
        # Pure first-principles derivations
        alpha_em = 1.0 / 137.036
        electron_mass_gev = 0.000511
        
        self.kappa = kappa if kappa is not None else (1.0 / (self.beta ** 2))
        self.alpha = alpha if alpha is not None else (0.5 * self.beta)
        self.g1 = g1 if g1 is not None else (alpha_em * self.beta / electron_mass_gev)
        self.g2 = g2 if g2 is not None else SFM_CONSTANTS.g2_alpha
        
        self.m_eff = m_eff
        self.hbar = hbar
        
        # Create universal minimizer
        self.minimizer = UniversalEnergyMinimizer(
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            g1=self.g1,
            g2=self.g2,
            V0=1.0,
            m_eff=m_eff,
            hbar=hbar,
        )
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
    
    def _get_quark_windings(self, quark: str, antiquark: str) -> Tuple[int, int, int]:
        """
        Get signed winding numbers for quark and antiquark.
        
        Antiquarks have OPPOSITE winding sign.
        
        Returns: (k_q, k_qbar, k_net)
        """
        k_q = QUARK_WINDING.get(quark, -3)
        
        # Antiquark has opposite sign
        antiquark_base = antiquark.replace('_bar', '')
        k_antiquark_base = QUARK_WINDING.get(antiquark_base, -3)
        k_qbar = -k_antiquark_base
        
        k_net = k_q + k_qbar
        
        return k_q, k_qbar, k_net
    
    def _initialize_meson(
        self,
        quark: str,
        antiquark: str,
        initial_amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Initialize meson as composite wavefunction: χ = χ_q + χ_qbar
        
        The interference between different windings creates the physics.
        """
        sigma = self.grid.sigma
        N = len(sigma)
        
        k_q, k_qbar, _ = self._get_quark_windings(quark, antiquark)
        
        # Initial width (will be optimized via delta_sigma)
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
        
        # Composite wavefunction
        chi = chi_q + chi_qbar
        
        # Scale to initial amplitude
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        if current_amp_sq > 1e-10:
            chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def _compute_k_eff(self, chi: NDArray[np.complexfloating]) -> float:
        """
        Compute effective winding from wavefunction gradient.
        
        k²_eff = ∫|∂χ/∂σ|² dσ / ∫|χ|² dσ
        
        This captures the quark-antiquark interference!
        """
        dchi = self.grid.first_derivative(chi)
        numerator = np.sum(np.abs(dchi)**2) * self.grid.dsigma
        denominator = np.sum(np.abs(chi)**2) * self.grid.dsigma
        
        if denominator < 1e-10:
            return 1.0
        
        return float(np.sqrt(numerator / denominator))
    
    def solve(
        self,
        meson_type: str = 'pion_plus',
        max_iter: int = 20000,
        tol: float = 1e-10,
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> CompositeMesonState:
        """
        Solve for meson using UNIVERSAL energy minimizer.
        
        Steps:
        1. Initialize meson wavefunction (meson-specific)
        2. Compute k_eff from interference (meson-specific)
        3. Use UNIVERSAL minimizer (same for all particles!)
        """
        config = MESON_CONFIGS.get(meson_type, MESON_CONFIGS['pion_plus'])
        quark = config['quark']
        antiquark = config['antiquark']
        
        k_q, k_qbar, k_net = self._get_quark_windings(quark, antiquark)
        generation = max(QUARK_SPATIAL_MODE.get(quark, 1), 
                        QUARK_SPATIAL_MODE.get(antiquark, 1))
        
        if verbose:
            print("=" * 60)
            print(f"MESON SOLVER (First-Principles): {meson_type.upper()}")
            print(f"  Quark: {quark} (k={k_q})")
            print(f"  Antiquark: {antiquark}-bar (k={k_qbar})")
            print(f"  k_net = {k_net}, generation = {generation}")
            print(f"  Using UNIVERSAL energy minimizer")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}, κ={self.kappa:.6g}")
            print("  NO phenomenological parameters!")
            print("=" * 60)
        
        # Step 1: Initialize meson wavefunction
        chi = self._initialize_meson(quark, antiquark, initial_amplitude)
        
        # Step 2: Compute k_eff from wavefunction (EMERGENT!)
        k_eff = self._compute_k_eff(chi)
        
        if verbose:
            print(f"  Emergent k_eff from interference: {k_eff:.4f}")
        
        # Step 3: Use UNIVERSAL minimizer
        result = self.minimizer.minimize(
            k_eff=k_eff,
            initial_A=np.sqrt(initial_amplitude),
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        
        if verbose:
            mass_mev = self.beta * result.A_squared * 1000
            print(f"\n  Results:")
            print(f"    A² = {result.A_squared:.6f}")
            print(f"    Δx = {result.delta_x:.6g}")
            print(f"    Δσ = {result.delta_sigma:.4f}")
            print(f"    Mass = {mass_mev:.2f} MeV (exp: {config['mass_mev']:.2f} MeV)")
            print(f"    Converged: {result.converged}")
        
        return CompositeMesonState(
            chi_meson=chi,
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
            k_eff=k_eff,
            generation=generation,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=result.final_residual,
        )
    
    def solve_pion(self, **kwargs) -> CompositeMesonState:
        """Solve for pion (π⁺ = ud̄)."""
        return self.solve(meson_type='pion_plus', **kwargs)
    
    def solve_jpsi(self, **kwargs) -> CompositeMesonState:
        """Solve for J/ψ (cc̄)."""
        return self.solve(meson_type='jpsi', **kwargs)
    
    def solve_upsilon(self, **kwargs) -> CompositeMesonState:
        """Solve for Υ(1S) (bb̄)."""
        return self.solve(meson_type='upsilon_1s', **kwargs)
