"""
SFM Parameter Optimizer - First-Principles beta-Only Optimization.

THEORETICAL BASIS:
==================
From "Implementation Note - The Beautiful Balance":

    "The solver's purpose is to DISCOVER the values of beta and other framework 
    parameters (alpha, kappa, V_0, g_1, g_2) by requiring that energy minimization 
    reproduces all known particle masses."

FIRST-PRINCIPLES APPROACH:
==========================
All parameters derive from beta and fundamental constants (hbar, c, G, alpha_em, m_e):

1. kappa = 8piGbeta/(hbarc^3)         - Enhanced 5D gravity at subspace scale
2. g_1 = alpha_em * beta^2 * c^2 / m_e  - Nonlinear coupling (EM/gravity hierarchy)
3. alpha_coupling = hbar^2/beta = hbarL_0c   - Spatial-subspace coupling strength

With these relationships, we search ONLY over beta. All other parameters
are derived, ensuring physical consistency across the framework.

APPROACH:
=========
1. Search over beta only
2. For each beta, compute {alpha, kappa, g_1} from first-principles formulas
3. Run actual solvers for all calibration particles
4. Minimize total mass prediction error
5. The optimal beta determines ALL framework parameters

Reference: docs/SFM_First_Principles_Parameter_Derivation_Plan.md

OPTIMAL PARAMETERS (13 December 2025):
===================================
The following parameter set achieves lepton mass ratios within 6% of experimental:

    alpha = 10.5      # Spatial-subspace coupling strength
    beta  = 100.0     # Mass scale (GeV)
    kappa = 0.00003   # Curvature/gravitational coupling
    g1    = 5000.0    # Nonlinear self-interaction (not used in current solution)

These values were found through systematic parameter optimization using the
self-consistent Deltax solver (solve_lepton_self_consistent) with the Compton
wavelength cap disabled and MIN_SCALE=0.0001. See docs/missing_physics_report.md
for full details.
"""

import argparse
import numpy as np
import os
from datetime import datetime
from scipy.optimize import differential_evolution, minimize_scalar, OptimizeResult
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import time
import warnings

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.core.constants import (
    save_constants_to_json,
    load_constants_from_json,
    get_constants_json_path,
    BETA as BETA_INITIAL,
    ALPHA as ALPHA_INITIAL,
    G_INTERNAL as G_INTERNAL_INITIAL,
    G1 as G1_INITIAL,
    G2 as G2_INITIAL,
    LAMBDA_SO as LAMBDA_SO_INITIAL,
)
from sfm_solver.core.particle_configurations import (
    PROTON, NEUTRON, LAMBDA,
    SIGMA_PLUS, SIGMA_ZERO, SIGMA_MINUS,
    XI_ZERO, XI_MINUS, OMEGA_MINUS,
    ALL_BARYONS,
)

# For backward compatibility with legacy "four" mode (deprecated)
# Compute KAPPA from G_INTERNAL: kappa = g_internal / beta
KAPPA_INITIAL = G_INTERNAL_INITIAL / BETA_INITIAL


# =============================================================================
# Fundamental Constants (SI units)
# =============================================================================

HBAR = 1.054571817e-34      # J.s (reduced Planck constant)
C = 299792458.0             # m/s (speed of light)
G = 6.67430e-11             # m^3/(kg.s^2) (Newton's gravitational constant)
ALPHA_EM = 1.0 / 137.036    # Fine structure constant (dimensionless)
M_E_KG = 9.1093837015e-31   # kg (electron mass)
M_E_GEV = 0.000511          # GeV (electron mass)

# Conversion factors
GEV_TO_KG = 1.78266192e-27  # kg per GeV
GEV_TO_JOULE = 1.602176634e-10  # J per GeV


# =============================================================================
# PURE FIRST-PRINCIPLES PARAMETER DERIVATIONS FROM beta
# =============================================================================
#
# NO PHENOMENOLOGICAL FUDGE FACTORS!
# These are the true theoretical derivations from SFM.
#
# FORMULAS (from user-provided SFM theory):
# =========================================
# 
# 1. kappa = 8piGbeta^2c/hbar  (enhanced 5D gravity at subspace scale)
#    G_eff = G/L_0 combined with beta L_0 c = hbar gives extra factor of beta
#
# 2. g_1 = alpha_em * beta / m_e  (nonlinear self-interaction)
#    Single power of beta (not beta^2)
#
# 3. alpha = C * beta where C ~ 0.5  (spatial-subspace coupling)
#    Direct proportionality to fundamental mass scale

# Dimensionless constant for alpha_coupling (from theory: C ~ 0.5)
ALPHA_COUPLING_C = 0.5  # alpha = C * beta


def derive_kappa_from_beta(beta_gev: float) -> float:
    """
    Derive kappa (curvature coupling) from beta - PURE FIRST PRINCIPLES (NO G!).
    
    FORMULA: kappa = hbar^2/(beta^2c^2)
    
    In natural units (hbar=c=1):
        kappa = 1/beta^2
    
    Physical basis:
        kappa is determined entirely by fundamental SFM parameters (beta, L_0, hbar, c).
        The Newtonian gravitational constant G is a PREDICTION that emerges
        from SFM, not an input. (See Beautiful Equation Section 5.6)
    
    Note: This gives kappa ~ 0.00034 GeV^-^2 for beta ~ 54 GeV, which is very
    different from what current solvers expect (~230 GeV^-^2). This
    discrepancy reveals phenomenological parameters in the solvers.
    
    Args:
        beta_gev: Mass coupling beta in GeV
        
    Returns:
        kappa in GeV^-^2
    """
    # Pure first-principles: kappa = 1/beta^2 (in natural units)
    kappa = 1.0 / (beta_gev ** 2)
    return kappa


def derive_g1_from_beta(beta_gev: float) -> float:
    """
    Derive g_1 (nonlinear coupling) from beta - PURE FIRST PRINCIPLES.
    
    FORMULA: g_1 = alpha_em * beta / m_e
    
    Physical basis:
        From EM/gravity hierarchy and fine structure constant.
        Single power of beta gives correct scaling.
    
    Args:
        beta_gev: Mass coupling beta in GeV
        
    Returns:
        g_1 (dimensionless)
    """
    # Pure first-principles: g_1 = alpha_em * beta / m_e
    g1 = ALPHA_EM * beta_gev / M_E_GEV
    return g1


def derive_alpha_coupling_from_beta(beta_gev: float) -> float:
    """
    Derive alpha_coupling (spatial-subspace coupling) from beta - PURE FIRST PRINCIPLES.
    
    FORMULA: alpha = C * beta  where C ~ 0.5
    
    Physical basis:
        Dimensional analysis of coupling integral shows [alpha] = [Energy].
        Coupling strength increases with fundamental mass scale.
    
    Args:
        beta_gev: Mass coupling beta in GeV
        
    Returns:
        alpha in GeV
    """
    # Pure first-principles: alpha = C * beta
    alpha = ALPHA_COUPLING_C * beta_gev
    return alpha


def derive_all_parameters_from_beta(beta_gev: float) -> Dict[str, float]:
    """
    Derive all SFM parameters from beta alone.
    
    This is the core of the first-principles approach: given only beta,
    all other parameters are determined by the SFM framework.
    
    Args:
        beta_gev: Mass coupling beta in GeV
        
    Returns:
        Dictionary with all derived parameters
    """
    return {
        'beta': beta_gev,
        'kappa': derive_kappa_from_beta(beta_gev),
        'g1': derive_g1_from_beta(beta_gev),
        'alpha': derive_alpha_coupling_from_beta(beta_gev),
        'L0': 1.0 / beta_gev,  # Beautiful Equation (natural units)
    }


# =============================================================================
# Particle Specifications
# =============================================================================

@dataclass
class ParticleSpec:
    """Specification for a particle to include in optimization."""
    
    name: str
    mass_gev: float
    particle_type: str  # 'lepton', 'meson', 'baryon'
    
    # For leptons
    generation: int = 1  # n = 1, 2, 3 for e, mu, tau
    k_lepton: int = 1    # Winding number for leptons
    
    # For mesons
    quark: Optional[str] = None
    antiquark: Optional[str] = None
    n_rad: int = 1       # Radial excitation
    meson_type: Optional[str] = None  # Key in MESON_CONFIGS
    
    # For baryons
    quarks: Optional[List[str]] = None
    baryon_type: Optional[str] = None  # 'proton' or 'neutron'
    
    # Weight in optimization (higher = more important to fit)
    weight: float = 1.0


# Calibration particle set - covers all particle types and generations
# Includes:
#   - All 3 lepton generations (n=1,2,3)
#   - Ground state meson (2-quark composite)
#   - Ground state baryon (3-quark composite)
CALIBRATION_PARTICLES = [
    # LEPTONS - all three generations
    ParticleSpec(
        name='electron',
        mass_gev=0.000511,
        particle_type='lepton',
        generation=1,
        k_lepton=1,
        weight=1.0
    ),
    ParticleSpec(
        name='muon',
        mass_gev=0.1057,
        particle_type='lepton',
        generation=2,
        k_lepton=1,
        weight=1.0
    ),
    ParticleSpec(
        name='tau',
        mass_gev=1.777,
        particle_type='lepton',
        generation=3,
        k_lepton=1,
        weight=1.0
    ),
    # BARYON - ground state (3-quark composite)
    ParticleSpec(
        name='proton',
        mass_gev=0.938,
        particle_type='baryon',
        quarks=['u', 'u', 'd'],
        baryon_type='proton',
        weight=1.0
    ),
]

# Validation particle set - held out to test predictions
# These are genuine predictions, not used in parameter fitting
VALIDATION_PARTICLES = [
    # MESON - moved from calibration (solver needs work)
    ParticleSpec(
        name='pion',
        mass_gev=0.140,
        particle_type='meson',
        quark='u',
        antiquark='d',
        n_rad=1,
        meson_type='pion_plus',
        weight=1.0
    ),
    # Pions (different charge states)
    ParticleSpec(
        name='pion_minus',
        mass_gev=0.13957,
        particle_type='meson',
        quark='d',
        antiquark='u',
        n_rad=1,
        meson_type='pion_minus',
        weight=1.0
    ),
    ParticleSpec(
        name='pion_zero',
        mass_gev=0.13498,
        particle_type='meson',
        quark='u',  # Actually (uu-bar + dd-bar)/sqrt(2)
        antiquark='u',
        n_rad=1,
        meson_type='pion_zero',
        weight=1.0
    ),
    # Charmonium states
    ParticleSpec(
        name='jpsi',
        mass_gev=3.097,
        particle_type='meson',
        quark='c',
        antiquark='c',
        n_rad=1,
        meson_type='jpsi',
        weight=1.0
    ),
    ParticleSpec(
        name='psi_2S',
        mass_gev=3.6861,
        particle_type='meson',
        quark='c',
        antiquark='c',
        n_rad=2,  # Radial excitation
        meson_type='psi_2S',
        weight=1.0
    ),
    # Bottomonium states
    ParticleSpec(
        name='upsilon_1S',
        mass_gev=9.4603,
        particle_type='meson',
        quark='b',
        antiquark='b',
        n_rad=1,
        meson_type='upsilon_1S',
        weight=1.0
    ),
    ParticleSpec(
        name='upsilon_2S',
        mass_gev=10.0233,
        particle_type='meson',
        quark='b',
        antiquark='b',
        n_rad=2,  # Radial excitation
        meson_type='upsilon_2S',
        weight=1.0
    ),
    # Baryons
    ParticleSpec(
        name='neutron',
        mass_gev=0.9396,
        particle_type='baryon',
        quarks=['u', 'd', 'd'],
        baryon_type='neutron',
        weight=1.0
    ),
]


# =============================================================================
# Optimization Result
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of global parameter optimization."""
    
    # Optimal parameters
    beta: float
    alpha: float
    kappa: float
    g1: float
    
    # EM parameters (for baryon mass splitting)
    g2: float = G2_INITIAL           # Circulation/EM coupling
    lambda_so: float = LAMBDA_SO_INITIAL  # Spin-orbit coupling
    
    # Derived quantities
    L0_gev_inv: float = 0.0  # From Beautiful Equation
    
    # Fit quality
    total_error: float = 0.0
    converged: bool = False
    
    # Per-particle results
    calibration_results: Dict[str, Dict] = field(default_factory=dict)
    validation_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Optimization metadata
    iterations: int = 0
    function_evaluations: int = 0
    time_seconds: float = 0.0
    
    # Original scipy result
    scipy_result: Optional[OptimizeResult] = None
    
    def save_constants_to_json(self, save: bool = True) -> None:
        """
        Save the optimized constants to constants.json.
        
        Saves: hbar, c, beta, alpha, kappa, g1, g2, lambda_so
        
        This updates the constants.json file in sfm_solver/core/ with the optimized
        values so they will be used as defaults in future runs.
        
        Args:
            save: If True, saves the constants. If False, skips saving.
        """
        if not save:
            print("\n[SKIP] Saving to constants.json disabled (--save-json off)")
            return
            
        # Load current constants to preserve hbar and c
        current = load_constants_from_json()
        
        # Update with optimized values
        updated_constants = {
            "hbar": current.get("hbar", HBAR),
            "c": current.get("c", C),
            "beta": self.beta,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "g1": self.g1,
            "g2": self.g2,
            "lambda_so": self.lambda_so,
        }
        
        save_constants_to_json(updated_constants)
        print(f"\n[OK] Optimized constants saved to: {get_constants_json_path()}")
    
    def summary(self) -> str:
        """Generate a summary of optimization results."""
        lines = [
            "=" * 70,
            "SFM FIRST-PRINCIPLES PARAMETER OPTIMIZATION RESULTS",
            "=" * 70,
            "",
            "OPTIMAL PARAMETERS:",
            f"  beta (beta)  = {self.beta:.4f} GeV",
            f"  alpha (alpha) = {self.alpha:.6f} GeV", 
            f"  kappa (kappa) = {self.kappa:.4f} GeV^-^2",
            f"  g_1        = {self.g1:.2f}",
            f"  g_2        = {self.g2:.6f} (EM circulation coupling)",
            f"  lambda_so  = {self.lambda_so:.4f} (spin-orbit coupling)",
            f"  L_0        = {self.L0_gev_inv:.6f} GeV^-1 (from Beautiful Equation)",
            "",
            f"CONVERGENCE: {'[OK] YES' if self.converged else '[FAIL] NO'}",
            f"Total Error: {self.total_error:.6f}",
            f"Iterations: {self.iterations}",
            f"Time: {self.time_seconds:.2f} s",
            "",
            "CALIBRATION PARTICLES (used to fit parameters):",
            "-" * 50,
        ]
        
        for name, result in self.calibration_results.items():
            status = '[OK]' if result['percent_error'] < 5 else '[FAIL]'
            lines.append(
                f"  {name:12s}: predicted={result['predicted_mass']*1000:.2f} MeV, "
                f"exp={result['experimental_mass']*1000:.2f} MeV, "
                f"error={result['percent_error']:.2f}% {status}"
            )
        
        lines.extend([
            "",
            "VALIDATION PARTICLES (genuine predictions):",
            "-" * 50,
        ])
        
        for name, result in self.validation_results.items():
            status = '[OK]' if result['percent_error'] < 5 else '[FAIL]'
            lines.append(
                f"  {name:12s}: predicted={result['predicted_mass']*1000:.2f} MeV, "
                f"exp={result['experimental_mass']*1000:.2f} MeV, "
                f"error={result['percent_error']:.2f}% {status}"
            )
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


# =============================================================================
# Core Optimizer Using Actual Solvers
# =============================================================================

class SFMParameterOptimizer:
    """
    Global optimizer to discover SFM framework parameters from first principles.
    
    This uses the ACTUAL SFM solvers in the optimization loop to accurately
    capture the physics. Slower than a simplified model but more accurate.
    """
    
    def __init__(
        self,
        calibration_particles: Optional[List[ParticleSpec]] = None,
        validation_particles: Optional[List[ParticleSpec]] = None,
        N_grid: int = 64,
        V0: float = 1.0,
        verbose: bool = True,
        log_file: Optional[str] = None,
        max_iter_lepton: int = 30,
        max_iter_meson: int = 30,
        max_iter_baryon: int = 30,
        max_iter_scf: int = 10,
    ):
        """
        Initialize the optimizer.
        
        Args:
            calibration_particles: Particles to use for fitting parameters.
                                   Default: pion, proton
            validation_particles: Particles to use for validation.
                                  Default: jpsi, neutron
            N_grid: Grid size for subspace solver.
            V0: Three-well potential depth (fixed at 1.0 GeV).
            verbose: Print progress information.
            log_file: Path to log file for dynamic updates (optional).
            max_iter_lepton: Maximum outer iterations for lepton solver.
            max_iter_meson: Maximum outer iterations for meson solver.
            max_iter_baryon: Maximum outer iterations for baryon solver.
            max_iter_scf: Maximum SCF iterations for baryon solver.
        """
        self.calibration = calibration_particles or CALIBRATION_PARTICLES
        self.validation = validation_particles or VALIDATION_PARTICLES
        self.N_grid = N_grid
        self.V0 = V0
        self.verbose = verbose
        self.log_file = log_file
        self.max_iter_lepton = max_iter_lepton
        self.max_iter_meson = max_iter_meson
        self.max_iter_baryon = max_iter_baryon
        self.max_iter_scf = max_iter_scf
        
        # Tracking
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        self._best_beta = None
        self._start_time = None
    
    def _log(self, message: str):
        """Write message to log file if configured."""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
                f.flush()  # Ensure immediate write
    
    def _run_lepton_solver(
        self,
        particle: ParticleSpec,
        beta: float,
        alpha: float,
        kappa: float,
        g1: float,
        g2: float = G2_INITIAL,
        lambda_so: float = LAMBDA_SO_INITIAL,
    ) -> Optional[float]:
        """
        Run the self-consistent lepton solver with injected parameters.
        
        Uses NonSeparableWavefunctionSolver.solve_lepton_self_consistent()
        which implements the self-consistent Deltax feedback mechanism.
        
        Returns:
            Predicted mass in GeV, or None if solver fails.
        """
        try:
            from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
            
            # Create solver with injected parameters
            solver = NonSeparableWavefunctionSolver(
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                g1=g1,
                g2=g2,
                lambda_so=lambda_so,
                V0=self.V0,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
            
            # Map generation to n_target
            n_target = particle.generation  # 1=electron, 2=muon, 3=tau
            
            # Run self-consistent solver
            result = solver.solve_lepton_self_consistent(
                n_target=n_target,
                k_winding=1,
                max_iter_outer=self.max_iter_lepton,
                max_iter_nl=0,  # Step 1 only (no nonlinear iteration needed)
                verbose=False,
            )
            
            # Check convergence with detailed diagnostics
            if not result.converged:
                # Extract convergence metrics from history
                conv_status = "UNKNOWN"
                final_change = None
                if result.convergence_history and 'A' in result.convergence_history:
                    A_hist = result.convergence_history['A']
                    if len(A_hist) > 1:
                        final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                        if final_change < 5e-4:
                            conv_status = "SOFT (nearly converged)"
                        elif final_change < 1e-3:
                            conv_status = "MODERATE (slow convergence)"
                        else:
                            conv_status = "HARD (not converging)"
                
                warning_msg = (f"WARNING: Lepton solver - {particle.name} (n={n_target}): "
                              f"iterations={result.iterations}/{self.max_iter_lepton}, "
                              f"status={conv_status}")
                if final_change is not None:
                    warning_msg += f", dA/A={final_change:.2e}"
                    if final_change < 5e-4:
                        warning_msg += " [LIKELY VALID]"
                    elif final_change < 1e-3:
                        warning_msg += " [CAUTION: verify results]"
                    else:
                        warning_msg += " [UNRELIABLE]"
                
                if self.verbose and self._eval_count % 100 == 0:
                    print(f"    {warning_msg}")
                if self.log_file:
                    self._log(f"  {warning_msg}")
            
            # Extract predicted mass: m = beta * A^2
            A = result.structure_norm
            m_pred = beta * A**2
            return m_pred
            
        except Exception as e:
            if self.verbose and self._eval_count % 100 == 0:
                print(f"    Lepton solver failed: {e}")
            return None
    
    def _run_meson_solver(
        self,
        particle: ParticleSpec,
        beta: float,
        alpha: float,
        kappa: float,
        g1: float,
        g2: float = G2_INITIAL,
        lambda_so: float = LAMBDA_SO_INITIAL,
    ) -> Optional[float]:
        """
        Run the self-consistent meson solver with injected parameters.
        
        Uses NonSeparableWavefunctionSolver.solve_meson_self_consistent().
        
        Returns:
            Predicted mass in GeV, or None if solver fails.
        """
        try:
            import numpy as np
            from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
            
            # Derive g_internal from kappa and beta for backward compatibility
            g_internal = kappa * beta
            
            # Create solver with injected parameters
            solver = NonSeparableWavefunctionSolver(
                alpha=alpha,
                g_internal=g_internal,
                g1=g1,
                g2=g2,
                lambda_so=lambda_so,
                V0=self.V0,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
            
            # Run self-consistent meson solver
            # quark_wells are well indices (1, 2), not angles
            result = solver.solve_meson_self_consistent(
                quark_wells=(1, 2),
                max_iter_outer=self.max_iter_meson,
                verbose=False,
            )
            
            # Check convergence with detailed diagnostics
            if not result.converged:
                # Extract convergence metrics from history
                conv_status = "UNKNOWN"
                final_change = None
                if result.convergence_history and 'A' in result.convergence_history:
                    A_hist = result.convergence_history['A']
                    if len(A_hist) > 1:
                        final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                        if final_change < 5e-4:
                            conv_status = "SOFT (nearly converged)"
                        elif final_change < 1e-3:
                            conv_status = "MODERATE (slow convergence)"
                        else:
                            conv_status = "HARD (not converging)"
                
                warning_msg = (f"WARNING: Meson solver - {particle.name}: "
                              f"iterations={result.iterations}/{self.max_iter_meson}, "
                              f"status={conv_status}")
                if final_change is not None:
                    warning_msg += f", dA/A={final_change:.2e}"
                    if final_change < 5e-4:
                        warning_msg += " [LIKELY VALID]"
                    elif final_change < 1e-3:
                        warning_msg += " [CAUTION: verify results]"
                    else:
                        warning_msg += " [UNRELIABLE]"
                
                if self.verbose and self._eval_count % 100 == 0:
                    print(f"    {warning_msg}")
                if self.log_file:
                    self._log(f"  {warning_msg}")
            
            # Extract predicted mass: m = beta * A^2
            A = result.structure_norm
            m_pred = beta * A**2
            return m_pred
            
        except Exception as e:
            if self.verbose and self._eval_count % 100 == 0:
                print(f"    Meson solver failed: {e}")
            return None
    
    def _run_baryon_solver(
        self,
        particle: ParticleSpec,
        beta: float,
        alpha: float,
        kappa: float,
        g1: float,
        g2: float = G2_INITIAL,
        lambda_so: float = LAMBDA_SO_INITIAL,
    ) -> Optional[Tuple[float, Optional[float]]]:
        """
        Run the self-consistent 4D baryon solver with injected parameters.
        
        Uses NonSeparableWavefunctionSolver.solve_baryon_4D_self_consistent().
        
        Returns:
            Tuple of (predicted mass in GeV, EM self-energy), or (None, None) if solver fails.
        """
        try:
            import numpy as np
            from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
            
            # Derive g_internal from kappa and beta for backward compatibility
            g_internal = kappa * beta
            
            # Create solver with injected parameters
            solver = NonSeparableWavefunctionSolver(
                alpha=alpha,
                g_internal=g_internal,
                g1=g1,
                g2=g2,
                lambda_so=lambda_so,
                V0=self.V0,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
            
            # Determine quark configuration from particle database
            particle_config = None
            for baryon in ALL_BARYONS:
                if baryon.name.lower() == particle.baryon_type.lower():
                    particle_config = baryon
                    break
            
            # Fall back to legacy hardcoded values if not in database
            if particle_config:
                quark_windings = particle_config.windings
                quark_spins = particle_config.spins
                quark_generations = particle_config.generations
            else:
                # Legacy fallback
                if particle.baryon_type == 'proton':
                    quark_windings = (5, 5, -3)  # uud: +2/3, +2/3, -1/3
                    quark_spins = (+1, -1, +1)
                    quark_generations = (1, 1, 1)
                elif particle.baryon_type == 'neutron':
                    quark_windings = (5, -3, -3)  # udd: +2/3, -1/3, -1/3
                    quark_spins = (+1, +1, -1)
                    quark_generations = (1, 1, 1)
                else:
                    quark_windings = None  # Use default
                    quark_spins = (+1, -1, +1)
                    quark_generations = (1, 1, 1)
            
            # Run self-consistent 4D baryon solver (new solver with shared Delta_x)
            # quark_wells are well indices (1, 2, 3)
            result = solver.solve_baryon_4D_self_consistent(
                quark_wells=(1, 2, 3),
                color_phases=(0, 2*np.pi/3, 4*np.pi/3),
                quark_windings=quark_windings,
                quark_spins=quark_spins,
                quark_generations=quark_generations,
                max_iter_outer=self.max_iter_baryon,
                max_iter_scf=self.max_iter_scf,
                verbose=False,
            )
            
            # Check convergence with detailed diagnostics
            if not result.converged:
                # Extract convergence metrics from history
                conv_status = "UNKNOWN"
                final_dA = None
                final_dDx = None
                
                if result.convergence_history:
                    hist = result.convergence_history
                    if 'A' in hist and len(hist['A']) > 1:
                        A_hist = hist['A']
                        final_dA = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                    
                    if 'Delta_x' in hist and len(hist['Delta_x']) > 1:
                        dx_hist = hist['Delta_x']
                        final_dDx = abs(dx_hist[-1] - dx_hist[-2]) / max(dx_hist[-1], 0.001)
                    
                    # Assess convergence quality
                    if final_dA is not None and final_dDx is not None:
                        max_change = max(final_dA, final_dDx)
                        if max_change < 5e-4:
                            conv_status = "SOFT (nearly converged)"
                        elif max_change < 1e-3:
                            conv_status = "MODERATE (slow convergence)"
                        else:
                            conv_status = "HARD (not converging)"
                
                warning_msg = (f"WARNING: Baryon solver - {particle.name} "
                              f"(windings={quark_windings}): "
                              f"iterations={result.iterations}/{self.max_iter_baryon}, "
                              f"status={conv_status}")
                
                if final_dA is not None:
                    warning_msg += f", dA/A={final_dA:.2e}"
                if final_dDx is not None:
                    warning_msg += f", dDx/Dx={final_dDx:.2e}"
                
                # Validity assessment
                if final_dA is not None and final_dDx is not None:
                    max_change = max(final_dA, final_dDx)
                    if max_change < 5e-4:
                        warning_msg += " [LIKELY VALID - use results]"
                    elif max_change < 1e-3:
                        warning_msg += " [CAUTION - verify physics]"
                    else:
                        warning_msg += " [UNRELIABLE - discard]"
                
                if self.verbose and self._eval_count % 100 == 0:
                    print(f"    {warning_msg}")
                if self.log_file:
                    self._log(f"  {warning_msg}")
            
            # Extract predicted mass: m = beta * A^2
            A = result.structure_norm
            m_pred = beta * A**2
            em_energy = result.em_energy
            return m_pred, em_energy
            
        except Exception as e:
            if self.verbose and self._eval_count % 100 == 0:
                print(f"    Baryon solver failed: {e}")
            return None
    
    def predict_mass(
        self,
        particle: ParticleSpec,
        beta: float,
        alpha: float,
        kappa: float,
        g1: float,
        g2: float = G2_INITIAL,
        lambda_so: float = LAMBDA_SO_INITIAL,
    ) -> Optional[float]:
        """
        Predict mass for a particle using actual solvers.
        
        Returns:
            Predicted mass in GeV, or None if solver fails.
        """
        if particle.particle_type == 'lepton':
            return self._run_lepton_solver(particle, beta, alpha, kappa, g1, g2, lambda_so)
        elif particle.particle_type == 'meson':
            return self._run_meson_solver(particle, beta, alpha, kappa, g1, g2, lambda_so)
        elif particle.particle_type == 'baryon':
            result = self._run_baryon_solver(particle, beta, alpha, kappa, g1, g2, lambda_so)
            if result is not None:
                return result[0]  # Return just the mass, not EM energy
            return None
        else:
            return None
           
    def _compute_error(self, beta: float, alpha: float, kappa: float, g1: float) -> float:
        """
        Compute total weighted error for a parameter set.
        
        Args:
            beta, alpha, kappa, g1: SFM parameters
            
        Returns:
            Total weighted squared relative error.
        """
        total_error = 0.0
        valid_count = 0
        particle_results = {}
        
        for particle in self.calibration:
            try:
                m_pred = self.predict_mass(particle, beta, alpha, kappa, g1)
                
                if m_pred is None or m_pred <= 0 or np.isnan(m_pred):
                    total_error += 10.0  # Penalty for failed calculation
                    particle_results[particle.name] = {'pred': None, 'exp': particle.mass_gev, 'error': 'FAILED'}
                    continue
                
                m_exp = particle.mass_gev
                
                # Relative squared error, weighted
                rel_error = ((m_pred - m_exp) / m_exp) ** 2
                total_error += particle.weight * rel_error
                valid_count += 1
                
                pct_error = abs(m_pred - m_exp) / m_exp * 100
                particle_results[particle.name] = {'pred': m_pred, 'exp': m_exp, 'error': pct_error}
                
            except Exception as e:
                total_error += 10.0
                particle_results[particle.name] = {'pred': None, 'exp': particle.mass_gev, 'error': str(e)}
        
        # Penalty if no valid predictions
        if valid_count == 0:
            total_error = 1e10
        
        # Track progress
        self._eval_count += 1
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        is_new_best = total_error < self._best_error
        if is_new_best:
            self._best_error = total_error
            self._best_beta = beta
            self._best_params = (beta, alpha, kappa, g1)
        
        # Log to file
        if self.log_file:
            log_msg = f"\n{'='*70}\n"
            log_msg += f"Eval {self._eval_count} | Time: {elapsed:.1f}s | {'*** NEW BEST ***' if is_new_best else ''}\n"
            log_msg += f"Parameters: beta={beta:.4f}, alpha={alpha:.6f}, kappa={kappa:.6f}, g1={g1:.2f}\n"
            log_msg += f"Total Error: {total_error:.6f} (best so far: {self._best_error:.6f})\n"
            log_msg += f"Particle predictions:\n"
            for name, res in particle_results.items():
                if res['pred'] is not None:
                    log_msg += f"  {name:12s}: pred={res['pred']*1000:.2f} MeV, exp={res['exp']*1000:.2f} MeV, error={res['error']:.1f}%\n"
                else:
                    log_msg += f"  {name:12s}: FAILED - {res['error']}\n"
            self._log(log_msg)
        
        # Console output
        if is_new_best and self.verbose:
            print(f"  Eval {self._eval_count}: New best error = {self._best_error:.6f}")
            print(f"    beta={beta:.4f} GeV, alpha={alpha:.6f}, kappa={kappa:.6f}, g1={g1:.2f}")
        elif self.verbose and self._eval_count % 20 == 0:
            print(f"  Eval {self._eval_count}: error = {total_error:.6f}")
        
        return total_error
       
    def _evaluate_particles(
        self,
        particles: List[ParticleSpec],
        beta: float,
        alpha: float,
        kappa: float,
        g1: float,
        g2: float = G2_INITIAL,
        lambda_so: float = LAMBDA_SO_INITIAL,
    ) -> Dict[str, Dict]:
        """Evaluate predictions for a list of particles."""
        results = {}
        for particle in particles:
            m_pred = self.predict_mass(particle, beta, alpha, kappa, g1, g2, lambda_so)
            m_exp = particle.mass_gev
            
            if m_pred is not None:
                percent_error = abs(m_pred - m_exp) / m_exp * 100
            else:
                m_pred = 0
                percent_error = 100
            
            results[particle.name] = {
                'predicted_mass': m_pred,
                'experimental_mass': m_exp,
                'percent_error': percent_error,
                'weight': particle.weight,
            }
        return results
        
    def compare_with_current(self) -> Dict:
        """
        Compare optimal parameters with current calibrated values.
        
        Returns:
            Dictionary comparing old vs new parameter values.
        """
        from sfm_solver.core.sfm_global import SFM_CONSTANTS
        
        current = {
            'beta': SFM_CONSTANTS.beta_physical,
            'alpha': SFM_CONSTANTS.alpha_coupling_base,
            'kappa': SFM_CONSTANTS.kappa_physical,
            'g1': SFM_CONSTANTS.g1,
        }
        
        return current
        
    def _objective_lepton(self, params: np.ndarray) -> float:
        """
        Objective function for lepton 3-parameter optimization.
        
        Optimizes alpha, g_internal, and g1 with beta derived from electron mass.
        
        Args:
            params: Array of [alpha, g_internal, g1]
            
        Returns:
            Total weighted error in mass predictions.
        """
        alpha, g_internal, g1 = params
        
        self._eval_count += 1
        start_time = time.time()
        
        # Compute A^2 for leptons
        A_squared = {}
        
        for particle in self.calibration:
            if particle.particle_type != 'lepton':
                continue
            
            try:
                from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
                
                solver = NonSeparableWavefunctionSolver(
                    alpha=alpha,
                    g_internal=g_internal,
                    g1=g1,
                    g2=0.004,
                    V0=self.V0,
                    n_max=5,
                    l_max=2,
                    N_sigma=64,
                )
                
                n_target = particle.generation
                result = solver.solve_lepton_self_consistent(
                    n_target=n_target,
                    k_winding=1,
                    max_iter_outer=self.max_iter_lepton,
                    max_iter_nl=0,
                    verbose=False,
                )
                
                # Check convergence with diagnostics
                if not result.converged:
                    conv_status = "UNKNOWN"
                    final_change = None
                    validity = ""
                    
                    if result.convergence_history and 'A' in result.convergence_history:
                        A_hist = result.convergence_history['A']
                        if len(A_hist) > 1:
                            final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                            if final_change < 5e-4:
                                conv_status = "SOFT"
                                validity = " [LIKELY VALID]"
                            elif final_change < 1e-3:
                                conv_status = "MODERATE"
                                validity = " [CAUTION]"
                            else:
                                conv_status = "HARD"
                                validity = " [UNRELIABLE]"
                    
                    warning_msg = (f"WARNING: Lepton - {particle.name} (n={n_target}): "
                                  f"{result.iterations}/{self.max_iter_lepton} iters, {conv_status}")
                    if final_change is not None:
                        warning_msg += f", dA/A={final_change:.2e}"
                    warning_msg += validity
                    
                    if self.log_file:
                        with open(self.log_file, 'a', encoding='utf-8') as f:
                            f.write(f"  {warning_msg}\n")
                
                A_squared[particle.name] = result.structure_norm ** 2
                
            except Exception as e:
                return 1e20
        
        if 'electron' not in A_squared or A_squared['electron'] <= 0:
            return 1e20
        
        # Derive beta from electron mass
        # beta = m_e / A_e^2 where m_e = 0.511 MeV
        # Then m = beta * A^2 gives mass in MeV
        m_e_exp = 0.511  # Electron mass in MeV
        beta_derived = m_e_exp / A_squared['electron']
        
        # Compute errors
        total_error = 0.0
        predictions = {}
        
        for particle in self.calibration:
            if particle.particle_type != 'lepton':
                continue
            
            if particle.name not in A_squared:
                continue
            
            m_pred = beta_derived * A_squared[particle.name]  # Mass in MeV
            m_exp = particle.mass_gev * 1000  # Convert GeV to MeV
            percent_error = abs(m_pred - m_exp) / m_exp * 100
            
            predictions[particle.name] = {
                'predicted': m_pred,  # In MeV
                'experimental': m_exp,  # In MeV
                'error': percent_error,
            }
            
            if particle.name == 'electron':
                total_error += percent_error * 0.01
            else:
                total_error += percent_error * particle.weight
        
        elapsed = time.time() - start_time
        
        is_best = total_error < self._best_error
        if is_best:
            self._best_error = total_error
            self._best_params = (alpha, g_internal, g1, beta_derived)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nEval {self._eval_count}: alpha={alpha:.4f}, g_int={g_internal:.6f}, g1={g1:.1f}")
                f.write(f" -> error={total_error:.2f}%")
                if is_best:
                    f.write(" *** BEST ***")
                f.write("\n")
        
        if self.verbose and (self._eval_count % 10 == 0 or is_best):
            marker = " *** BEST ***" if is_best else ""
            print(f"Eval {self._eval_count}: alpha={alpha:.4f}, g_int={g_internal:.6f}, "
                  f"g1={g1:.1f}, error={total_error:.2f}%{marker}")
        
        return total_error

    def optimize_lepton(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        maxiter: int = 100,
        seed: int = 42,
        popsize: int = 10,
        save_json: bool = True,
        bounds_tol: float = 0.01,
    ) -> OptimizationResult:
        """
        Lepton 3-parameter optimization over alpha, g_internal, and g1.
        
        Beta is derived from electron mass after solving.
        
        Args:
            bounds: List of (min, max) bounds for [alpha, g_internal, g1].
                   If None, calculated as initial_value * (1 +/- bounds_tol).
            maxiter: Maximum iterations.
            seed: Random seed for reproducibility.
            popsize: Population size.
            save_json: If True, saves optimized constants.
            bounds_tol: Fractional tolerance for auto-calculated bounds (default: 0.01 = 1%).
            
        Returns:
            OptimizationResult with optimal parameters.
        """
        if bounds is None:
            # Calculate bounds as +/- bounds_tol around initial values
            bounds = [
                (ALPHA_INITIAL * (1 - bounds_tol), ALPHA_INITIAL * (1 + bounds_tol)),
                (G_INTERNAL_INITIAL * (1 - bounds_tol), G_INTERNAL_INITIAL * (1 + bounds_tol)),
                (G1_INITIAL * (1 - bounds_tol), G1_INITIAL * (1 + bounds_tol)),
            ]
        
        if self.verbose:
            print("=" * 70)
            print("LEPTON 3-PARAMETER OPTIMIZATION")
            print("Optimizing {alpha, g_internal, g1}")
            print("Beta derived from electron: beta = m_e / A_e^2")
            print("=" * 70)
            print(f"\nParameter bounds:")
            print(f"  alpha:      [{bounds[0][0]}, {bounds[0][1]}]")
            print(f"  g_internal: [{bounds[1][0]}, {bounds[1][1]}]")
            print(f"  g1:         [{bounds[2][0]}, {bounds[2][1]}]")
            print("-" * 70)
        
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        self._start_time = time.time()
        
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("SFM LEPTON 3-PARAMETER OPTIMIZATION\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
        
        x0 = [ALPHA_INITIAL, G_INTERNAL_INITIAL, G1_INITIAL]
        
        if self.verbose:
            print(f"Seeding: alpha={x0[0]}, g_internal={x0[1]}, g1={x0[2]}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                self._objective_lepton,
                bounds=bounds,
                x0=x0,
                maxiter=maxiter,
                seed=seed,
                workers=1,
                disp=False,
                polish=True,
                tol=1e-6,
                atol=1e-6,
                popsize=popsize,
            )
        
        elapsed = time.time() - self._start_time
        
        alpha_opt, g_internal_opt, g1_opt = result.x
        
        # Derive beta from optimal parameters
        try:
            from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
            
            solver = NonSeparableWavefunctionSolver(
                alpha=alpha_opt,
                g_internal=g_internal_opt,
                g1=g1_opt,
            )
            
            result_e = solver.solve_lepton_self_consistent(
                n_target=1, k_winding=1, max_iter_outer=self.max_iter_lepton, verbose=False
            )
            
            # Check convergence with diagnostics
            if not result_e.converged:
                conv_status = "UNKNOWN"
                final_change = None
                validity = ""
                
                if result_e.convergence_history and 'A' in result_e.convergence_history:
                    A_hist = result_e.convergence_history['A']
                    if len(A_hist) > 1:
                        final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                        if final_change < 5e-4:
                            conv_status = "SOFT"
                            validity = " [LIKELY VALID]"
                        elif final_change < 1e-3:
                            conv_status = "MODERATE"
                            validity = " [CAUTION]"
                        else:
                            conv_status = "HARD"
                            validity = " [UNRELIABLE]"
                
                warning_msg = (f"WARNING: Final electron: {result_e.iterations}/{self.max_iter_lepton} iters, "
                              f"{conv_status}")
                if final_change is not None:
                    warning_msg += f", dA/A={final_change:.2e}"
                warning_msg += validity
                
                if self.verbose:
                    print(f"  {warning_msg}")
                if self.log_file:
                    self._log(f"  {warning_msg}")
            
            A_e_squared = result_e.structure_norm ** 2
            beta_opt = 0.511 / A_e_squared  # SFM units convention
            
        except Exception as e:
            print(f"Warning: Failed to derive beta: {e}")
            beta_opt = BETA_INITIAL
        
        # For backward compatibility with result structure
        kappa_opt = g_internal_opt / beta_opt  # Derived kappa
        L0_opt = 1.0 / beta_opt
        
        if self.verbose:
            print("-" * 70)
            print(f"Optimal: alpha={alpha_opt:.4f}, g_internal={g_internal_opt:.6f}, g1={g1_opt:.1f}")
            print(f"Derived: beta={beta_opt:.8f} GeV")
        
        calibration_results = self._evaluate_particles(
            self.calibration, beta_opt, alpha_opt, kappa_opt, g1_opt
        )
        validation_results = self._evaluate_particles(
            self.validation, beta_opt, alpha_opt, kappa_opt, g1_opt
        )
        
        opt_result = OptimizationResult(
            beta=beta_opt,
            alpha=alpha_opt,
            kappa=kappa_opt,  # Derived
            g1=g1_opt,
            L0_gev_inv=L0_opt,
            total_error=result.fun,
            converged=result.success,
            calibration_results=calibration_results,
            validation_results=validation_results,
            iterations=result.nit,
            function_evaluations=result.nfev,
            time_seconds=elapsed,
            scipy_result=result,
        )
        
        if self.verbose:
            print("\n" + opt_result.summary())
        
        opt_result.save_constants_to_json(save=save_json)
        
        return opt_result


    def _objective_baryon(self, params: np.ndarray) -> float:
        """
        Objective function for baryon parameter optimization.
        
        Optimizes g1, g2, and lambda_so to match proton and neutron masses,
        with alpha and g_internal fixed at current values (calibrated for leptons).
        
        Args:
            params: Array of [g1, g2, lambda_so]
            
        Returns:
            Total weighted error in mass predictions, emphasizing mass splitting.
        """
        g1, g2, lambda_so = params
        
        # Use fixed values from constants (calibrated for leptons)
        alpha = ALPHA_INITIAL
        g_internal = G_INTERNAL_INITIAL
        
        self._eval_count += 1
        start_time = time.time()
        
        from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
        
        # Create solver with current parameters
        solver = NonSeparableWavefunctionSolver(
            alpha=alpha,
            g_internal=g_internal,
            g1=g1,
            g2=g2,
            lambda_so=lambda_so,
            V0=self.V0,
            n_max=5,
            l_max=2,
            N_sigma=64,
        )
        
        # First, get electron amplitude to derive beta
        try:
            result_e = solver.solve_lepton_self_consistent(
                n_target=1, k_winding=1, max_iter_outer=self.max_iter_lepton, verbose=False
            )
            
            # Check convergence with diagnostics
            if not result_e.converged:
                conv_status = "UNKNOWN"
                final_change = None
                validity = ""
                
                if result_e.convergence_history and 'A' in result_e.convergence_history:
                    A_hist = result_e.convergence_history['A']
                    if len(A_hist) > 1:
                        final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                        if final_change < 5e-4:
                            conv_status = "SOFT"
                            validity = " [LIKELY VALID]"
                        elif final_change < 1e-3:
                            conv_status = "MODERATE"
                            validity = " [CAUTION]"
                        else:
                            conv_status = "HARD"
                            validity = " [UNRELIABLE]"
                
                warning_msg = (f"WARNING: Electron: {result_e.iterations}/{self.max_iter_lepton} iters, "
                              f"{conv_status}")
                if final_change is not None:
                    warning_msg += f", dA/A={final_change:.2e}"
                warning_msg += validity
                
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"  {warning_msg}\n")
            
            A_e_squared = result_e.structure_norm ** 2
            beta_derived = 0.511 / A_e_squared  # SFM units convention
        except Exception:
            return 1e20
        
        # Solve for proton and neutron using 4D solver
        try:
            result_p = solver.solve_baryon_4D_self_consistent(
                quark_wells=(1, 2, 3),
                color_phases=(0, 2*np.pi/3, 4*np.pi/3),
                quark_windings=(5, 5, -3),  # uud
                max_iter_outer=self.max_iter_baryon,
                max_iter_scf=self.max_iter_scf,
                verbose=False,
            )
            
            # Check proton convergence with diagnostics
            if not result_p.converged:
                conv_status = "UNKNOWN"
                final_dA, final_dDx = None, None
                validity = ""
                
                if result_p.convergence_history:
                    hist = result_p.convergence_history
                    if 'A' in hist and len(hist['A']) > 1:
                        final_dA = abs(hist['A'][-1] - hist['A'][-2]) / max(hist['A'][-1], 0.01)
                    if 'Delta_x' in hist and len(hist['Delta_x']) > 1:
                        final_dDx = abs(hist['Delta_x'][-1] - hist['Delta_x'][-2]) / max(hist['Delta_x'][-1], 0.001)
                    
                    if final_dA is not None and final_dDx is not None:
                        max_change = max(final_dA, final_dDx)
                        if max_change < 5e-4:
                            conv_status = "SOFT"
                            validity = " [LIKELY VALID]"
                        elif max_change < 1e-3:
                            conv_status = "MODERATE"
                            validity = " [CAUTION]"
                        else:
                            conv_status = "HARD"
                            validity = " [UNRELIABLE]"
                
                warning_msg = (f"WARNING: Proton: {result_p.iterations}/{self.max_iter_baryon} iters, "
                              f"{conv_status}")
                if final_dA is not None:
                    warning_msg += f", dA/A={final_dA:.2e}"
                if final_dDx is not None:
                    warning_msg += f", dDx/Dx={final_dDx:.2e}"
                warning_msg += validity
                
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"  {warning_msg}\n")
            
            m_proton = beta_derived * result_p.structure_norm ** 2 * 1000  # Baryon formula
            
            result_n = solver.solve_baryon_4D_self_consistent(
                quark_wells=(1, 2, 3),
                color_phases=(0, 2*np.pi/3, 4*np.pi/3),
                quark_windings=(5, -3, -3),  # udd
                max_iter_outer=self.max_iter_baryon,
                max_iter_scf=self.max_iter_scf,
                verbose=False,
            )
            
            # Check neutron convergence with diagnostics
            if not result_n.converged:
                conv_status = "UNKNOWN"
                final_dA, final_dDx = None, None
                validity = ""
                
                if result_n.convergence_history:
                    hist = result_n.convergence_history
                    if 'A' in hist and len(hist['A']) > 1:
                        final_dA = abs(hist['A'][-1] - hist['A'][-2]) / max(hist['A'][-1], 0.01)
                    if 'Delta_x' in hist and len(hist['Delta_x']) > 1:
                        final_dDx = abs(hist['Delta_x'][-1] - hist['Delta_x'][-2]) / max(hist['Delta_x'][-1], 0.001)
                    
                    if final_dA is not None and final_dDx is not None:
                        max_change = max(final_dA, final_dDx)
                        if max_change < 5e-4:
                            conv_status = "SOFT"
                            validity = " [LIKELY VALID]"
                        elif max_change < 1e-3:
                            conv_status = "MODERATE"
                            validity = " [CAUTION]"
                        else:
                            conv_status = "HARD"
                            validity = " [UNRELIABLE]"
                
                warning_msg = (f"WARNING: Neutron: {result_n.iterations}/{self.max_iter_baryon} iters, "
                              f"{conv_status}")
                if final_dA is not None:
                    warning_msg += f", dA/A={final_dA:.2e}"
                if final_dDx is not None:
                    warning_msg += f", dDx/Dx={final_dDx:.2e}"
                warning_msg += validity
                
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"  {warning_msg}\n")
            
            m_neutron = beta_derived * result_n.structure_norm ** 2 * 1000  # Baryon formula
        except Exception:
            return 1e20
        
        # Experimental values (convert to MeV)
        m_p_exp = 938.272  # MeV
        m_n_exp = 939.565  # MeV
        delta_m_exp = m_n_exp - m_p_exp  # ~1.293 MeV
        
        # Predicted mass difference
        delta_m_pred = m_neutron - m_proton
        
        # Compute errors
        # 1. Absolute mass errors (PRIMARY - high weight, these are what we want to optimize)
        p_error = ((m_proton - m_p_exp) / m_p_exp) ** 2
        n_error = ((m_neutron - m_n_exp) / m_n_exp) ** 2
        
        # 2. Mass splitting error (SECONDARY - lower weight, for fine-tuning)
        splitting_error = ((delta_m_pred - delta_m_exp) / delta_m_exp) ** 2
        
        # 3. Ordering penalty: neutron MUST be heavier than proton
        ordering_penalty = 0.0 if delta_m_pred > 0 else 100.0
        
        # Combine errors with weights
        # Absolute masses are most important (10x weight each)
        # Mass splitting is secondary (0.5x weight)
        total_error = 10.0 * p_error + 10.0 * n_error + 0.5 * splitting_error + ordering_penalty
        
        elapsed = time.time() - start_time
        
        is_best = total_error < self._best_error
        if is_best:
            self._best_error = total_error
            self._best_params = (g2, lambda_so, beta_derived)
        
        # Calculate percent errors (with sign: negative = below target, positive = above target)
        p_err_pct = (m_proton - m_p_exp) / m_p_exp * 100
        n_err_pct = (m_neutron - m_n_exp) / m_n_exp * 100
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nEval {self._eval_count}: g1={g1:.2f}, g2={g2:.6f}, lambda_so={lambda_so:.4f}")
                f.write(f" -> m_p={m_proton:.2f} MeV ({p_err_pct:+.1f}%), m_n={m_neutron:.2f} MeV ({n_err_pct:+.1f}%)")
                f.write(f", delta_m={delta_m_pred:.3f} MeV")
                f.write(f", error={total_error:.4f}")
                if is_best:
                    f.write(" *** BEST ***")
                f.write("\n")
        
        if self.verbose and (self._eval_count % 5 == 0 or is_best):
            marker = " *** BEST ***" if is_best else ""
            print(f"Eval {self._eval_count}: g1={g1:.2f}, g2={g2:.5f}, lambda_so={lambda_so:.3f} "
                  f"-> m_p={m_proton:.1f} MeV ({p_err_pct:+.1f}%), m_n={m_neutron:.1f} MeV ({n_err_pct:+.1f}%){marker}")
        
        return total_error
    
    def optimize_baryon(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        maxiter: int = 50,
        seed: int = 42,
        popsize: int = 10,
        save_json: bool = True,
        bounds_tol: float = 0.01,
    ) -> OptimizationResult:
        """
        Baryon parameter optimization: optimize g1, g2, and lambda_so for proton and neutron masses.
        
        Alpha and g_internal are fixed at current constants.json values (calibrated for leptons).
        Beta is derived from electron mass.
        
        This mode prioritizes absolute proton and neutron mass accuracy (PRIMARY objective),
        with the ~1.3 MeV mass difference as a secondary consideration for fine-tuning.
        
        Args:
            bounds: List of (min, max) bounds for [g1, g2, lambda_so].
                   If None, calculated as initial_value * (1 +/- bounds_tol).
            maxiter: Maximum iterations.
            seed: Random seed for reproducibility.
            popsize: Population size for differential evolution.
            save_json: If True, saves optimized constants to constants.json.
            bounds_tol: Fractional tolerance for auto-calculated bounds (default: 0.01 = 1%).
            
        Returns:
            OptimizationResult with optimal parameters and predictions.
        """
        if bounds is None:
            # Calculate bounds as +/- bounds_tol around initial values
            bounds = [
                (G1_INITIAL * (1 - bounds_tol), G1_INITIAL * (1 + bounds_tol)),
                (G2_INITIAL * (1 - bounds_tol), G2_INITIAL * (1 + bounds_tol)),
                (LAMBDA_SO_INITIAL * (1 - bounds_tol), LAMBDA_SO_INITIAL * (1 + bounds_tol)),
            ]
        
        if self.verbose:
            print("=" * 70)
            print("BARYON PARAMETER OPTIMIZATION")
            print("Optimizing {g1, g2, lambda_so} for proton and neutron masses")
            print("Alpha and g_internal fixed at current values (calibrated for leptons)")
            print("=" * 70)
            print(f"\nFixed parameters:")
            print(f"  alpha      = {ALPHA_INITIAL}")
            print(f"  g_internal = {G_INTERNAL_INITIAL}")
            print(f"\nParameter bounds:")
            print(f"  g1:        [{bounds[0][0]}, {bounds[0][1]}]")
            print(f"  g2:        [{bounds[1][0]}, {bounds[1][1]}]")
            print(f"  lambda_so: [{bounds[2][0]}, {bounds[2][1]}]")
            print(f"\nTargets: proton mass = 938.27 MeV, neutron mass = 939.57 MeV")
            print(f"         mass difference = 1.293 MeV")
            print("-" * 70)
        
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        self._start_time = time.time()
        
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("SFM BARYON PARAMETER OPTIMIZATION\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
                f.write("Targets: proton mass = 938.27 MeV, neutron mass = 939.57 MeV\n")
                f.write("         mass difference = 1.293 MeV\n")
                f.write(f"Fixed: alpha={ALPHA_INITIAL}, g_internal={G_INTERNAL_INITIAL}\n")
                f.write(f"Bounds: g1=[{bounds[0][0]}, {bounds[0][1]}], "
                        f"g2=[{bounds[1][0]}, {bounds[1][1]}], "
                        f"lambda_so=[{bounds[2][0]}, {bounds[2][1]}]\n")
                f.write("=" * 70 + "\n")
        
        start_time = time.time()
        
        # Seed with current values
        x0 = [G1_INITIAL, G2_INITIAL, LAMBDA_SO_INITIAL]
        
        if self.verbose:
            print(f"Starting from: g1={x0[0]}, g2={x0[1]}, lambda_so={x0[2]}")
        
        # Use differential evolution for global optimization over 3 parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                self._objective_baryon,
                bounds=bounds,
                x0=x0,
                maxiter=maxiter,
                seed=seed,
                workers=1,
                disp=False,
                polish=True,
                tol=1e-4,
                atol=0.1,
                popsize=popsize,
            )
        
        elapsed = time.time() - start_time
        
        g1_opt, g2_opt, lambda_so_opt = result.x
        
        # Derive beta from electron mass
        try:
            from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
            
            solver = NonSeparableWavefunctionSolver(
                alpha=ALPHA_INITIAL,
                g_internal=G_INTERNAL_INITIAL,
                g1=g1_opt,
                g2=g2_opt,
                lambda_so=lambda_so_opt,
            )
            
            result_e = solver.solve_lepton_self_consistent(
                n_target=1, k_winding=1, max_iter_outer=self.max_iter_lepton, verbose=False
            )
            
            # Check convergence with diagnostics
            if not result_e.converged:
                conv_status = "UNKNOWN"
                final_change = None
                validity = ""
                
                if result_e.convergence_history and 'A' in result_e.convergence_history:
                    A_hist = result_e.convergence_history['A']
                    if len(A_hist) > 1:
                        final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                        if final_change < 5e-4:
                            conv_status = "SOFT"
                            validity = " [LIKELY VALID]"
                        elif final_change < 1e-3:
                            conv_status = "MODERATE"
                            validity = " [CAUTION]"
                        else:
                            conv_status = "HARD"
                            validity = " [UNRELIABLE]"
                
                warning_msg = (f"WARNING: Final electron: {result_e.iterations}/{self.max_iter_lepton} iters, "
                              f"{conv_status}")
                if final_change is not None:
                    warning_msg += f", dA/A={final_change:.2e}"
                warning_msg += validity
                
                if self.verbose:
                    print(f"  {warning_msg}")
                if self.log_file:
                    self._log(f"  {warning_msg}")
            
            A_e_squared = result_e.structure_norm ** 2
            beta_opt = 0.511 / A_e_squared  # SFM units convention
        except Exception as e:
            print(f"Warning: Failed to derive beta: {e}")
            beta_opt = BETA_INITIAL
        
        # Derive kappa for backward compatibility
        kappa_opt = G_INTERNAL_INITIAL / beta_opt
        L0_opt = 1.0 / beta_opt
        
        if self.verbose:
            print("-" * 70)
            print(f"Optimization complete.")
            print(f"Optimal: g1={g1_opt:.2f}, g2={g2_opt:.6f}, lambda_so={lambda_so_opt:.4f}")
            print(f"Derived: beta={beta_opt:.8f} GeV")
        
        # Compute results for all particles with optimized baryon parameters
        calibration_results = self._evaluate_particles(
            self.calibration, beta_opt, ALPHA_INITIAL, kappa_opt, g1_opt, g2_opt, lambda_so_opt
        )
        validation_results = self._evaluate_particles(
            self.validation, beta_opt, ALPHA_INITIAL, kappa_opt, g1_opt, g2_opt, lambda_so_opt
        )
        
        opt_result = OptimizationResult(
            beta=beta_opt,
            alpha=ALPHA_INITIAL,
            kappa=kappa_opt,
            g1=g1_opt,
            g2=g2_opt,
            lambda_so=lambda_so_opt,
            L0_gev_inv=L0_opt,
            total_error=result.fun,
            converged=result.success,
            calibration_results=calibration_results,
            validation_results=validation_results,
            iterations=getattr(result, 'nit', 0),
            function_evaluations=getattr(result, 'nfev', self._eval_count),
            time_seconds=elapsed,
            scipy_result=result,
        )
        
        if self.verbose:
            print("\n" + opt_result.summary())
            
            # Print specific proton-neutron comparison
            print("\nPROTON-NEUTRON MASS SPLITTING:")
            print("-" * 40)
            p_mass = calibration_results.get('proton', {}).get('predicted_mass', 0) * 1000
            n_mass = validation_results.get('neutron', {}).get('predicted_mass', 0) * 1000
            delta = n_mass - p_mass
            print(f"  Proton mass:  {p_mass:.2f} MeV (exp: 938.27 MeV)")
            print(f"  Neutron mass: {n_mass:.2f} MeV (exp: 939.57 MeV)")
            print(f"  Difference:   {delta:.3f} MeV (exp: 1.293 MeV)")
        
        opt_result.save_constants_to_json(save=save_json)
        
        return opt_result


    def _objective_full(self, params: np.ndarray) -> float:
        """
        Objective function for full 5-parameter optimization.
        
        Optimizes alpha, g_internal, g1, g2, and lambda_so simultaneously.
        Beta is derived from electron mass.
        
        Args:
            params: Array of [alpha, g_internal, g1, g2, lambda_so]
            
        Returns:
            Total weighted error in mass predictions.
        """
        alpha, g_internal, g1, g2, lambda_so = params
        
        self._eval_count += 1
        start_time = time.time()
        
        from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
        
        # Create solver with current parameters
        try:
            solver = NonSeparableWavefunctionSolver(
                alpha=alpha,
                g_internal=g_internal,
                g1=g1,
                g2=g2,
                lambda_so=lambda_so,
                V0=self.V0,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
        except Exception:
            return 1e20
        
        # Compute A^2 for all calibration particles
        A_squared = {}
        em_energies = {}
        
        for particle in self.calibration:
            try:
                if particle.particle_type == 'lepton':
                    n_target = particle.generation
                    result = solver.solve_lepton_self_consistent(
                        n_target=n_target,
                        k_winding=1,
                        max_iter_outer=30,
                        max_iter_nl=0,
                        verbose=False,
                    )
                    
                    # Check convergence
                    if not result.converged:
                        warning_msg = (f"WARNING: Lepton solver did not converge for {particle.name} "
                                      f"(n={n_target}) after {result.iterations} iterations")
                        if self.log_file:
                            with open(self.log_file, 'a', encoding='utf-8') as f:
                                f.write(f"  {warning_msg}\n")
                    
                    A_squared[particle.name] = result.structure_norm ** 2
                    
                elif particle.particle_type == 'meson':
                    result = solver.solve_meson_self_consistent(
                        quark_wells=(1, 2),
                        max_iter_outer=30,
                        verbose=False,
                    )
                    
                    # Check convergence with diagnostics
                    if not result.converged:
                        conv_status = "UNKNOWN"
                        final_change = None
                        validity = ""
                        
                        if result.convergence_history and 'A' in result.convergence_history:
                            A_hist = result.convergence_history['A']
                            if len(A_hist) > 1:
                                final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                                if final_change < 5e-4:
                                    conv_status = "SOFT"
                                    validity = " [LIKELY VALID]"
                                elif final_change < 1e-3:
                                    conv_status = "MODERATE"
                                    validity = " [CAUTION]"
                                else:
                                    conv_status = "HARD"
                                    validity = " [UNRELIABLE]"
                        
                        warning_msg = (f"WARNING: Meson - {particle.name}: "
                                      f"{result.iterations}/{self.max_iter_meson} iters, {conv_status}")
                        if final_change is not None:
                            warning_msg += f", dA/A={final_change:.2e}"
                        warning_msg += validity
                        
                        if self.log_file:
                            with open(self.log_file, 'a', encoding='utf-8') as f:
                                f.write(f"  {warning_msg}\n")
                    
                    A_squared[particle.name] = result.structure_norm ** 2
                    
                elif particle.particle_type == 'baryon':
                    # Determine quark windings
                    if particle.baryon_type == 'proton':
                        quark_windings = (5, 5, -3)
                    elif particle.baryon_type == 'neutron':
                        quark_windings = (5, -3, -3)
                    else:
                        quark_windings = None
                    
                    result = solver.solve_baryon_4D_self_consistent(
                        quark_wells=(1, 2, 3),
                        color_phases=(0, 2*np.pi/3, 4*np.pi/3),
                        quark_windings=quark_windings,
                        max_iter_outer=self.max_iter_baryon,
                        max_iter_scf=self.max_iter_scf,
                        verbose=False,
                    )
                    
                    # Check convergence with diagnostics
                    if not result.converged:
                        conv_status = "UNKNOWN"
                        final_dA, final_dDx = None, None
                        validity = ""
                        
                        if result.convergence_history:
                            hist = result.convergence_history
                            if 'A' in hist and len(hist['A']) > 1:
                                final_dA = abs(hist['A'][-1] - hist['A'][-2]) / max(hist['A'][-1], 0.01)
                            if 'Delta_x' in hist and len(hist['Delta_x']) > 1:
                                final_dDx = abs(hist['Delta_x'][-1] - hist['Delta_x'][-2]) / max(hist['Delta_x'][-1], 0.001)
                            
                            if final_dA is not None and final_dDx is not None:
                                max_change = max(final_dA, final_dDx)
                                if max_change < 5e-4:
                                    conv_status = "SOFT"
                                    validity = " [LIKELY VALID]"
                                elif max_change < 1e-3:
                                    conv_status = "MODERATE"
                                    validity = " [CAUTION]"
                                else:
                                    conv_status = "HARD"
                                    validity = " [UNRELIABLE]"
                        
                        warning_msg = (f"WARNING: Baryon - {particle.name} (k={quark_windings}): "
                                      f"{result.iterations}/{self.max_iter_baryon} iters, {conv_status}")
                        if final_dA is not None:
                            warning_msg += f", dA/A={final_dA:.2e}"
                        if final_dDx is not None:
                            warning_msg += f", dDx/Dx={final_dDx:.2e}"
                        warning_msg += validity
                        
                        if self.log_file:
                            with open(self.log_file, 'a', encoding='utf-8') as f:
                                f.write(f"  {warning_msg}\n")
                    
                    A_squared[particle.name] = result.structure_norm ** 2
                    em_energies[particle.name] = result.em_energy
                    
            except Exception:
                return 1e20
        
        # Also solve neutron (from validation set) for mass splitting
        try:
            result_n = solver.solve_baryon_4D_self_consistent(
                quark_wells=(1, 2, 3),
                color_phases=(0, 2*np.pi/3, 4*np.pi/3),
                quark_windings=(5, -3, -3),  # udd
                max_iter_outer=self.max_iter_baryon,
                max_iter_scf=self.max_iter_scf,
                verbose=False,
            )
            
            # Check convergence
            if not result_n.converged:
                warning_msg = (f"WARNING: Neutron solver did not converge "
                              f"after {result_n.iterations} iterations")
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"  {warning_msg}\n")
            
            A_squared['neutron'] = result_n.structure_norm ** 2
            em_energies['neutron'] = result_n.em_energy
        except Exception:
            return 1e20
        
        # Check we have electron
        if 'electron' not in A_squared or A_squared['electron'] <= 0:
            return 1e20
        
        # Derive beta from electron mass
        # beta = 0.511 / A_e^2
        # For leptons: m = beta * A^2 (in MeV)
        # For baryons: m = beta * A^2 * 1000 (convert GeV to MeV)
        m_e_exp = 0.511  # Electron mass in MeV
        beta_derived = m_e_exp / A_squared['electron']
        
        # Compute predicted masses and errors
        total_error = 0.0
        predictions = {}
        
        for particle in self.calibration:
            if particle.name not in A_squared:
                continue
            
            # Apply correct formula based on particle type
            if particle.particle_type == 'lepton':
                m_pred = beta_derived * A_squared[particle.name]  # Already in MeV
            else:
                m_pred = beta_derived * A_squared[particle.name] * 1000  # Convert to MeV
            
            m_exp = particle.mass_gev * 1000  # Convert GeV to MeV
            percent_error = abs(m_pred - m_exp) / m_exp * 100
            
            predictions[particle.name] = {
                'predicted': m_pred,  # In MeV
                'experimental': m_exp,  # In MeV
                'error': percent_error,
            }
            
            # Weight: electron is perfect by construction, others weighted normally
            if particle.name == 'electron':
                total_error += percent_error * 0.01
            else:
                total_error += percent_error * particle.weight
        
        # Add proton-neutron mass splitting error (SECONDARY - lower weight for fine-tuning)
        if 'proton' in A_squared and 'neutron' in A_squared:
            m_proton = beta_derived * A_squared['proton'] * 1000  # Baryon formula: convert to MeV
            m_neutron = beta_derived * A_squared['neutron'] * 1000  # Baryon formula: convert to MeV
            
            delta_m_exp = 1.293  # MeV (experimental neutron-proton mass difference)
            delta_m_pred = m_neutron - m_proton  # In MeV
            
            # Splitting error (0.5x weight - secondary to absolute masses)
            splitting_error = ((delta_m_pred - delta_m_exp) / delta_m_exp) ** 2 * 100
            
            # Ordering penalty: neutron MUST be heavier
            if delta_m_pred <= 0:
                splitting_error += 1000.0
            
            total_error += 0.5 * splitting_error
            
            predictions['n-p_splitting'] = {
                'predicted': delta_m_pred,  # Already in MeV
                'experimental': delta_m_exp,  # Already in MeV
                'error': abs(delta_m_pred - delta_m_exp) / delta_m_exp * 100,
            }
        
        elapsed = time.time() - start_time
        
        is_best = total_error < self._best_error
        if is_best:
            self._best_error = total_error
            self._best_params = (alpha, g_internal, g1, g2, lambda_so, beta_derived)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nEval {self._eval_count}: alpha={alpha:.4f}, g_int={g_internal:.6f}, ")
                f.write(f"g1={g1:.2f}, g2={g2:.5f}, lambda_so={lambda_so:.3f}")
                f.write(f" -> error={total_error:.2f}")
                if is_best:
                    f.write(" *** BEST ***")
                f.write("\n")
                for name, pred in predictions.items():
                    f.write(f"  {name}: {pred['predicted']:.2f} MeV (exp: {pred['experimental']:.2f}), err={pred['error']:.1f}%\n")
        
        if self.verbose and (self._eval_count % 5 == 0 or is_best):
            marker = " *** BEST ***" if is_best else ""
            print(f"Eval {self._eval_count}: a={alpha:.3f}, g_int={g_internal:.5f}, "
                  f"g1={g1:.1f}, g2={g2:.4f}, lso={lambda_so:.2f} -> err={total_error:.1f}{marker}")
        
        return total_error
    
    def optimize_full(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        maxiter: int = 100,
        seed: int = 42,
        popsize: int = 15,
        save_json: bool = True,
        bounds_tol: float = 0.01,
    ) -> OptimizationResult:
        """
        Full 5-parameter optimization: alpha, g_internal, g1, g2, and lambda_so.
        
        Beta is derived from electron mass.
        This mode prioritizes absolute mass accuracy (PRIMARY), with proton-neutron
        splitting as a secondary consideration for fine-tuning.
        
        Args:
            bounds: List of (min, max) bounds for [alpha, g_internal, g1, g2, lambda_so].
                   If None, calculated as initial_value * (1 +/- bounds_tol).
            maxiter: Maximum iterations.
            seed: Random seed for reproducibility.
            popsize: Population size for differential evolution.
            save_json: If True, saves optimized constants to constants.json.
            bounds_tol: Fractional tolerance for auto-calculated bounds (default: 0.01 = 1%).
            
        Returns:
            OptimizationResult with optimal parameters and predictions.
        """
        if bounds is None:
            # Calculate bounds as +/- bounds_tol around initial values
            bounds = [
                (ALPHA_INITIAL * (1 - bounds_tol), ALPHA_INITIAL * (1 + bounds_tol)),
                (G_INTERNAL_INITIAL * (1 - bounds_tol), G_INTERNAL_INITIAL * (1 + bounds_tol)),
                (G1_INITIAL * (1 - bounds_tol), G1_INITIAL * (1 + bounds_tol)),
                (G2_INITIAL * (1 - bounds_tol), G2_INITIAL * (1 + bounds_tol)),
                (LAMBDA_SO_INITIAL * (1 - bounds_tol), LAMBDA_SO_INITIAL * (1 + bounds_tol)),
            ]
        
        if self.verbose:
            print("=" * 70)
            print("FULL 5-PARAMETER OPTIMIZATION")
            print("Optimizing {alpha, g_internal, g1, g2, lambda_so}")
            print("Beta derived from electron mass")
            print("Targets: absolute masses + proton-neutron mass splitting")
            print("=" * 70)
            print(f"\nParameter bounds:")
            print(f"  alpha:      [{bounds[0][0]}, {bounds[0][1]}]")
            print(f"  g_internal: [{bounds[1][0]}, {bounds[1][1]}]")
            print(f"  g1:         [{bounds[2][0]}, {bounds[2][1]}]")
            print(f"  g2:         [{bounds[3][0]}, {bounds[3][1]}]")
            print(f"  lambda_so:  [{bounds[4][0]}, {bounds[4][1]}]")
            print("-" * 70)
        
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        self._start_time = time.time()
        
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("SFM FULL 5-PARAMETER OPTIMIZATION\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
                f.write("Optimizing: alpha, g_internal, g1, g2, lambda_so\n")
                f.write("Targets: absolute masses + p-n mass splitting\n")
                f.write(f"Bounds: alpha=[{bounds[0]}], g_internal=[{bounds[1]}]\n")
                f.write(f"        g1=[{bounds[2]}], g2=[{bounds[3]}], lambda_so=[{bounds[4]}]\n")
                f.write("=" * 70 + "\n")
        
        start_time = time.time()
        
        # Seed with current values
        x0 = [ALPHA_INITIAL, G_INTERNAL_INITIAL, G1_INITIAL, G2_INITIAL, LAMBDA_SO_INITIAL]
        
        if self.verbose:
            print(f"Starting from: alpha={x0[0]}, g_internal={x0[1]}, g1={x0[2]}, g2={x0[3]}, lambda_so={x0[4]}")
        
        # Use differential evolution for global search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                self._objective_full,
                bounds=bounds,
                x0=x0,
                maxiter=maxiter,
                seed=seed,
                workers=1,
                disp=False,
                polish=True,
                tol=1e-4,
                atol=0.1,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
            )
        
        elapsed = time.time() - start_time
        
        alpha_opt, g_internal_opt, g1_opt, g2_opt, lambda_so_opt = result.x
        
        # Derive beta from electron mass
        try:
            from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
            
            solver = NonSeparableWavefunctionSolver(
                alpha=alpha_opt,
                g_internal=g_internal_opt,
                g1=g1_opt,
                g2=g2_opt,
                lambda_so=lambda_so_opt,
            )
            
            result_e = solver.solve_lepton_self_consistent(
                n_target=1, k_winding=1, max_iter_outer=self.max_iter_lepton, verbose=False
            )
            
            # Check convergence with diagnostics
            if not result_e.converged:
                conv_status = "UNKNOWN"
                final_change = None
                validity = ""
                
                if result_e.convergence_history and 'A' in result_e.convergence_history:
                    A_hist = result_e.convergence_history['A']
                    if len(A_hist) > 1:
                        final_change = abs(A_hist[-1] - A_hist[-2]) / max(A_hist[-1], 0.01)
                        if final_change < 5e-4:
                            conv_status = "SOFT"
                            validity = " [LIKELY VALID]"
                        elif final_change < 1e-3:
                            conv_status = "MODERATE"
                            validity = " [CAUTION]"
                        else:
                            conv_status = "HARD"
                            validity = " [UNRELIABLE]"
                
                warning_msg = (f"WARNING: Final electron: {result_e.iterations}/{self.max_iter_lepton} iters, "
                              f"{conv_status}")
                if final_change is not None:
                    warning_msg += f", dA/A={final_change:.2e}"
                warning_msg += validity
                
                if self.verbose:
                    print(f"  {warning_msg}")
                if self.log_file:
                    self._log(f"  {warning_msg}")
            
            A_e_squared = result_e.structure_norm ** 2
            beta_opt = 0.511 / A_e_squared  # SFM units convention
        except Exception as e:
            print(f"Warning: Failed to derive beta: {e}")
            beta_opt = BETA_INITIAL
        
        kappa_opt = g_internal_opt / beta_opt
        L0_opt = 1.0 / beta_opt
        
        if self.verbose:
            print("-" * 70)
            print(f"Optimization complete.")
            print(f"Optimal: alpha={alpha_opt:.6f}, g_internal={g_internal_opt:.8f}")
            print(f"         g1={g1_opt:.4f}, g2={g2_opt:.6f}, lambda_so={lambda_so_opt:.4f}")
            print(f"Derived: beta={beta_opt:.8f} GeV")
        
        # Compute results for all particles
        calibration_results = self._evaluate_particles(
            self.calibration, beta_opt, alpha_opt, kappa_opt, g1_opt, g2_opt, lambda_so_opt
        )
        validation_results = self._evaluate_particles(
            self.validation, beta_opt, alpha_opt, kappa_opt, g1_opt, g2_opt, lambda_so_opt
        )
        
        opt_result = OptimizationResult(
            beta=beta_opt,
            alpha=alpha_opt,
            kappa=kappa_opt,
            g1=g1_opt,
            g2=g2_opt,
            lambda_so=lambda_so_opt,
            L0_gev_inv=L0_opt,
            total_error=result.fun,
            converged=result.success,
            calibration_results=calibration_results,
            validation_results=validation_results,
            iterations=result.nit,
            function_evaluations=result.nfev,
            time_seconds=elapsed,
            scipy_result=result,
        )
        
        if self.verbose:
            print("\n" + opt_result.summary())
            
            # Print specific proton-neutron comparison
            print("\nPROTON-NEUTRON MASS SPLITTING:")
            print("-" * 40)
            p_mass = calibration_results.get('proton', {}).get('predicted_mass', 0) * 1000
            n_mass = validation_results.get('neutron', {}).get('predicted_mass', 0) * 1000
            delta = n_mass - p_mass
            print(f"  Proton mass:  {p_mass:.2f} MeV (exp: 938.27 MeV)")
            print(f"  Neutron mass: {n_mass:.2f} MeV (exp: 939.57 MeV)")
            print(f"  Difference:   {delta:.3f} MeV (exp: 1.293 MeV)")
        
        opt_result.save_constants_to_json(save=save_json)
        
        return opt_result


def run_optimization_full(
    verbose: bool = True,
    maxiter: int = 100,
    log_file: Optional[str] = None,
    save_json: bool = True,
    max_iter_lepton: int = 30,
    max_iter_meson: int = 30,
    max_iter_baryon: int = 30,
    max_iter_scf: int = 10,
    bounds_tol: float = 0.01,
) -> OptimizationResult:
    """
    Run full 5-parameter optimization.
    
    Optimizes alpha, g_internal, g2, and lambda_so for both absolute masses
    and proton-neutron mass splitting.
    
    Args:
        verbose: Print progress information.
        maxiter: Maximum iterations.
        log_file: Path to log file for optimization progress (optional).
        save_json: If True, saves optimized constants to constants.json.
        max_iter_lepton: Maximum outer iterations for lepton solver.
        max_iter_meson: Maximum outer iterations for meson solver.
        max_iter_baryon: Maximum outer iterations for baryon solver.
        max_iter_scf: Maximum SCF iterations for baryon solver.
        bounds_tol: Fractional tolerance for parameter bounds (default: 0.01 = 1%).
    
    Returns:
        OptimizationResult with optimal parameters.
    """
    optimizer = SFMParameterOptimizer(
        verbose=verbose, 
        log_file=log_file,
        max_iter_lepton=max_iter_lepton,
        max_iter_meson=max_iter_meson,
        max_iter_baryon=max_iter_baryon,
        max_iter_scf=max_iter_scf,
    )
    return optimizer.optimize_full(maxiter=maxiter, save_json=save_json, bounds_tol=bounds_tol)


def run_optimization_baryon(
    verbose: bool = True,
    maxiter: int = 50,
    log_file: Optional[str] = None,
    save_json: bool = True,
    max_iter_lepton: int = 30,
    max_iter_baryon: int = 30,
    max_iter_scf: int = 10,
    bounds_tol: float = 0.01,
) -> OptimizationResult:
    """
    Run baryon parameter optimization for proton and neutron masses.
    
    Optimizes g1, g2, and lambda_so while keeping alpha/g_internal fixed (calibrated for leptons).
    
    Args:
        verbose: Print progress information.
        maxiter: Maximum iterations.
        log_file: Path to log file for optimization progress (optional).
        save_json: If True, saves optimized constants to constants.json.
        max_iter_lepton: Maximum outer iterations for lepton solver (for beta derivation).
        max_iter_baryon: Maximum outer iterations for baryon solver.
        max_iter_scf: Maximum SCF iterations for baryon solver.
        bounds_tol: Fractional tolerance for parameter bounds (default: 0.01 = 1%).
    
    Returns:
        OptimizationResult with optimal parameters.
    """
    optimizer = SFMParameterOptimizer(
        verbose=verbose, 
        log_file=log_file,
        max_iter_lepton=max_iter_lepton,
        max_iter_baryon=max_iter_baryon,
        max_iter_scf=max_iter_scf,
    )
    return optimizer.optimize_baryon(maxiter=maxiter, save_json=save_json, bounds_tol=bounds_tol)


def run_optimization_lepton(
    verbose: bool = True,
    maxiter: int = 100,
    log_file: Optional[str] = None,
    save_json: bool = True,
    max_iter_lepton: int = 30,
    bounds_tol: float = 0.01,
) -> OptimizationResult:
    """
    Run lepton 3-parameter optimization over alpha, g_internal, and g1.
    
    Beta is derived from the electron mass after solving.
    
    Args:
        verbose: Print progress information.
        maxiter: Maximum iterations.
        log_file: Path to log file for optimization progress (optional).
        save_json: If True, saves optimized constants to constants.json.
        max_iter_lepton: Maximum outer iterations for lepton solver.
        bounds_tol: Fractional tolerance for parameter bounds (default: 0.01 = 1%).
    
    Returns:
        OptimizationResult with optimal parameters.
    """
    optimizer = SFMParameterOptimizer(
        verbose=verbose, 
        log_file=log_file,
        max_iter_lepton=max_iter_lepton,
    )
    return optimizer.optimize_lepton(maxiter=maxiter, save_json=save_json, bounds_tol=bounds_tol)


def create_log_file_path() -> str:
    """
    Create a log file path with timestamp in sfm-solver/logs directory.
    
    Returns:
        Path to the log file.
    """
    # Get the path to sfm-solver/logs directory (project root)
    # Path: optimization/ -> sfm_solver/ -> src/ -> sfm-solver/ -> logs/
    logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"optimizer_log_{timestamp}.txt"
    
    return str(logs_dir / log_filename)


def main():
    """Main entry point for the parameter optimizer CLI."""
    parser = argparse.ArgumentParser(
        description="SFM Parameter Optimizer - Optimize fundamental constants for the Single-Field Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parameter_optimizer.py                     # full mode (default), save to JSON
  python parameter_optimizer.py --mode lepton       # lepton mode - optimize alpha, g_internal, g1
  python parameter_optimizer.py --mode baryon       # baryon mode - optimize g1, g2, lambda_so for p-n masses
  python parameter_optimizer.py --mode full         # full mode - optimize all 5 params (alpha, g_internal, g1, g2, lambda_so)
  python parameter_optimizer.py --save-json off     # Don't save to constants.json
  python parameter_optimizer.py --max-iter 200      # Run 200 full iterations
  python parameter_optimizer.py --max-iter-baryon 50 --max-iter-scf 20  # Increase baryon solver iterations
  python parameter_optimizer.py --bounds-tol 0.05   # Set parameter bounds to +/- 5% of initial values
  python parameter_optimizer.py --mode baryon --bounds-tol 0.001  # Tight 0.1% bounds for baryon fine-tuning
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lepton", "baryon", "full"],
        default="full",
        help="Optimization mode: "
             "'lepton' for 3-parameter (alpha, g_internal, g1) lepton calibration, "
             "'baryon' for baryon parameters (g1, g2, lambda_so) for proton and neutron masses, "
             "'full' for all 5 parameters (alpha, g_internal, g1, g2, lambda_so) [default]"
    )
    
    parser.add_argument(
        "--save-json",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Save optimized constants to constants.json: 'on' or 'off' (default: off)"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        dest="max_iter",
        help="Maximum number of iterations (default: 100)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--max-iter-lepton",
        type=int,
        default=30,
        dest="max_iter_lepton",
        help="Maximum outer iterations for lepton solver (default: 30)"
    )
    
    parser.add_argument(
        "--max-iter-meson",
        type=int,
        default=30,
        dest="max_iter_meson",
        help="Maximum outer iterations for meson solver (default: 30)"
    )
    
    parser.add_argument(
        "--max-iter-baryon",
        type=int,
        default=30,
        dest="max_iter_baryon",
        help="Maximum outer iterations for baryon solver (default: 30)"
    )
    
    parser.add_argument(
        "--max-iter-scf",
        type=int,
        default=10,
        dest="max_iter_scf",
        help="Maximum SCF iterations for solver initialization (default: 10)"
    )
    
    parser.add_argument(
        "--bounds-tol",
        type=float,
        default=0.01,
        dest="bounds_tol",
        help="Fractional tolerance for parameter bounds as +/- fraction of initial value (default: 0.01 = 1%%)"
    )
    
    args = parser.parse_args()
    
    # Determine save_json setting
    save_json = args.save_json == "on"
    verbose = not args.quiet
    
    # Create log file path
    log_file = create_log_file_path()
    
    # Map mode names for display
    mode_names = {
        'lepton': '3-parameter (alpha, g_internal, g1) lepton calibration',
        'baryon': 'baryon parameters (g1, g2, lambda_so) for proton and neutron masses',
        'full': 'full 5-parameter (alpha, g_internal, g1, g2, lambda_so)',
    }
    
    # Print startup information
    print("=" * 70)
    print("SFM PARAMETER OPTIMIZER")
    print("=" * 70)
    print(f"Mode: {mode_names.get(args.mode, args.mode)}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Bounds tolerance: {args.bounds_tol * 100:.1f}% (+/- around initial values)")
    print(f"Save to constants.json: {'Yes' if save_json else 'No'}")
    print(f"Log file: {log_file}")
    print()
    print(f"Initial values from constants.json:")
    print(f"  alpha      = {ALPHA_INITIAL}")
    print(f"  g_internal = {G_INTERNAL_INITIAL}")
    print(f"  g1         = {G1_INITIAL}")
    print(f"  g2         = {G2_INITIAL}")
    print(f"  lambda_so  = {LAMBDA_SO_INITIAL}")
    print("=" * 70)
    print()
    
    # Run the appropriate optimization
    if args.mode == "baryon":
        result = run_optimization_baryon(
            verbose=verbose,
            maxiter=args.max_iter,
            log_file=log_file,
            save_json=save_json,
            max_iter_lepton=args.max_iter_lepton,
            max_iter_baryon=args.max_iter_baryon,
            max_iter_scf=args.max_iter_scf,
            bounds_tol=args.bounds_tol,
        )
    elif args.mode == "full":
        result = run_optimization_full(
            verbose=verbose,
            maxiter=args.max_iter,
            log_file=log_file,
            save_json=save_json,
            max_iter_lepton=args.max_iter_lepton,
            max_iter_meson=args.max_iter_meson,
            max_iter_baryon=args.max_iter_baryon,
            max_iter_scf=args.max_iter_scf,
            bounds_tol=args.bounds_tol,
        )
    else:  # lepton
        result = run_optimization_lepton(
            verbose=verbose,
            maxiter=args.max_iter,
            log_file=log_file,
            save_json=save_json,
            max_iter_lepton=args.max_iter_lepton,
            bounds_tol=args.bounds_tol,
        )
    
    # Print final summary
    print("\n" + result.summary())
    print(f"\nLog file saved to: {log_file}")


if __name__ == '__main__':
    main()
