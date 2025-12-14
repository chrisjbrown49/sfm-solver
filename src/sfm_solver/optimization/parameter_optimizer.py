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
    # MESON - ground state (2-quark composite)
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
    
    # Derived quantities
    L0_gev_inv: float  # From Beautiful Equation
    
    # Fit quality
    total_error: float
    converged: bool
    
    # Per-particle results
    calibration_results: Dict[str, Dict]
    validation_results: Dict[str, Dict]
    
    # Optimization metadata
    iterations: int
    function_evaluations: int
    time_seconds: float
    
    # Original scipy result
    scipy_result: Optional[OptimizeResult] = None
    
    def save_constants_to_json(self, save: bool = True) -> None:
        """
        Save the optimized constants (hbar, c, beta, alpha, kappa, g1) to constants.json.
        
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
        log_file: Optional[str] = None
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
        """
        self.calibration = calibration_particles or CALIBRATION_PARTICLES
        self.validation = validation_particles or VALIDATION_PARTICLES
        self.N_grid = N_grid
        self.V0 = V0
        self.verbose = verbose
        self.log_file = log_file
        
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
        g1: float
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
                g2=0.004,
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
                max_iter_outer=30,
                max_iter_nl=0,  # Step 1 only (no nonlinear iteration needed)
                verbose=False,
            )
            
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
        g1: float
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
                g2=0.004,
                V0=self.V0,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
            
            # Run self-consistent meson solver
            # quark_wells are well indices (1, 2), not angles
            result = solver.solve_meson_self_consistent(
                quark_wells=(1, 2),
                max_iter_outer=30,
                verbose=False,
            )
            
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
        g1: float
    ) -> Optional[float]:
        """
        Run the self-consistent baryon solver with injected parameters.
        
        Uses NonSeparableWavefunctionSolver.solve_baryon_self_consistent().
        
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
                g2=0.004,
                V0=self.V0,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
            
            # Run self-consistent baryon solver
            # quark_wells are well indices (1, 2, 3)
            result = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3),
                max_iter_outer=30,
                verbose=False,
            )
            
            # Extract predicted mass: m = beta * A^2
            A = result.structure_norm
            m_pred = beta * A**2
            return m_pred
            
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
        g1: float
    ) -> Optional[float]:
        """
        Predict mass for a particle using actual solvers.
        
        Returns:
            Predicted mass in GeV, or None if solver fails.
        """
        if particle.particle_type == 'lepton':
            return self._run_lepton_solver(particle, beta, alpha, kappa, g1)
        elif particle.particle_type == 'meson':
            return self._run_meson_solver(particle, beta, alpha, kappa, g1)
        elif particle.particle_type == 'baryon':
            return self._run_baryon_solver(particle, beta, alpha, kappa, g1)
        else:
            return None
    
    def _objective_beta_only(self, beta: float) -> float:
        """
        FIRST-PRINCIPLES objective function for beta-only optimization.
        
        All other parameters are DERIVED from beta using SFM relationships:
            kappa = 8piGbeta/(hbarc^3)
            g_1 = alpha_em * beta^2 / m_e
            alpha_coupling = 1/beta
        
        Args:
            beta: Mass coupling constant in GeV
            
        Returns:
            Total weighted error.
        """
        if beta <= 0:
            return 1e10
        
        # DERIVE all parameters from beta (first principles!)
        params = derive_all_parameters_from_beta(beta)
        alpha = params['alpha']
        kappa = params['kappa']
        g1 = params['g1']
        
        return self._compute_error(beta, alpha, kappa, g1)
    
    def _objective(self, params: np.ndarray) -> float:
        """
        Legacy objective function for 4-parameter optimization.
        
        DEPRECATED: Use _objective_beta_only for first-principles approach.
        
        Args:
            params: [beta, alpha, kappa, g1]
            
        Returns:
            Total weighted error.
        """
        beta, alpha, kappa, g1 = params
        
        # Enforce positivity
        if any(p <= 0 for p in [beta, alpha, kappa, g1]):
            return 1e10
        
        return self._compute_error(beta, alpha, kappa, g1)
    
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
    
    def optimize_beta_only(
        self,
        beta_bounds: Optional[Tuple[float, float]] = None,
        maxiter: int = 100,
        save_json: bool = True,
    ) -> OptimizationResult:
        """
        FIRST-PRINCIPLES optimization: search ONLY over beta.
        
        All other parameters are DERIVED from beta using SFM relationships:
            kappa = 8piGbeta/(hbarc^3)           - Enhanced 5D gravity
            g_1 = alpha_em * beta^2 / m_e     - Nonlinear coupling (EM/gravity hierarchy)
            alpha_coupling = 1/beta         - Spatial-subspace coupling
        
        Args:
            beta_bounds: (min, max) bounds for beta in GeV. Default is (10.0, 500.0).
            maxiter: Maximum iterations for optimization.
            save_json: If True, saves optimized constants to constants.json.
            
        Returns:
            OptimizationResult with optimal beta and derived parameters.
            
        Raises:
            ValueError: If the initial beta value from constants.json is outside bounds.
        """
        # Use fixed default bounds
        if beta_bounds is None:
            beta_bounds = (10.0, 500.0)
        
        # Validate that loaded initial value is within bounds
        if not (beta_bounds[0] <= BETA_INITIAL <= beta_bounds[1]):
            raise ValueError(
                f"Initial beta value from constants.json ({BETA_INITIAL}) is outside "
                f"the optimization bounds [{beta_bounds[0]}, {beta_bounds[1]}]. "
                f"Please update constants.json with a beta value within the valid range, "
                f"or specify different bounds using the beta_bounds parameter."
            )
        
        if self.verbose:
            print("=" * 70)
            print("SFM FIRST-PRINCIPLES BETA-ONLY OPTIMIZATION")
            print("=" * 70)
            print("\nDERIVATION FORMULAS (all from beta):")
            print("  kappa = 1/beta^2         [enhanced 5D gravity]")
            print("  g1 = alpha_em * beta / m_e   [EM/gravity hierarchy]")
            print("  alpha = C * beta         [spatial-subspace coupling]")
            print(f"\nCalibration particles: {[p.name for p in self.calibration]}")
            print(f"Validation particles: {[p.name for p in self.validation]}")
            print(f"\nSearching beta in [{beta_bounds[0]}, {beta_bounds[1]}] GeV")
            print("-" * 70)
        
        # Reset tracking
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_beta = None
        
        start_time = time.time()
        
        # Use scalar optimization (1D search over beta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize_scalar(
                self._objective_beta_only,
                bounds=beta_bounds,
                method='bounded',
                options={'maxiter': maxiter, 'xatol': 0.01}
            )
        
        elapsed = time.time() - start_time
        
        # Extract optimal beta and derive all other parameters
        beta_opt = result.x
        params = derive_all_parameters_from_beta(beta_opt)
        alpha_opt = params['alpha']
        kappa_opt = params['kappa']
        g1_opt = params['g1']
        L0_opt = params['L0']
        
        if self.verbose:
            print("-" * 70)
            print(f"Optimization complete. Computing final predictions...")
        
        # Compute results for all particles
        calibration_results = self._evaluate_particles(
            self.calibration, beta_opt, alpha_opt, kappa_opt, g1_opt
        )
        validation_results = self._evaluate_particles(
            self.validation, beta_opt, alpha_opt, kappa_opt, g1_opt
        )
        
        # Create result object
        opt_result = OptimizationResult(
            beta=beta_opt,
            alpha=alpha_opt,
            kappa=kappa_opt,
            g1=g1_opt,
            L0_gev_inv=L0_opt,
            total_error=result.fun,
            converged=result.success if hasattr(result, 'success') else True,
            calibration_results=calibration_results,
            validation_results=validation_results,
            iterations=result.nfev,
            function_evaluations=result.nfev,
            time_seconds=elapsed,
            scipy_result=None,
        )
        
        if self.verbose:
            print("\n" + opt_result.summary())
            print("\nDERIVED PARAMETER VERIFICATION:")
            print(f"  From beta = {beta_opt:.4f} GeV:")
            print(f"    kappa = 8piGbeta/(hbarc^3) = {kappa_opt:.6e} GeV^-^2")
            print(f"    g_1 = alpha_em * beta^2 / m_e = {g1_opt:.2f}")
            print(f"    alpha = 1/beta = {alpha_opt:.6f} GeV")
        
        # Save optimized constants to JSON file (if enabled)
        opt_result.save_constants_to_json(save=save_json)
        
        return opt_result
    
    def _evaluate_particles(
        self,
        particles: List[ParticleSpec],
        beta: float,
        alpha: float,
        kappa: float,
        g1: float
    ) -> Dict[str, Dict]:
        """Evaluate predictions for a list of particles."""
        results = {}
        for particle in particles:
            m_pred = self.predict_mass(particle, beta, alpha, kappa, g1)
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
    
    def optimize(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        maxiter: int = 100,
        seed: int = 42,
        popsize: int = 10,
        save_json: bool = True,
    ) -> OptimizationResult:
        """
        DEPRECATED: Use optimize_beta_only() for first-principles approach.
        
        This method searches over all 4 parameters independently, which
        does not guarantee physical consistency.
        
        Args:
            bounds: List of (min, max) bounds for [beta, alpha, kappa, g1].
            maxiter: Maximum iterations.
            seed: Random seed for reproducibility.
            popsize: Population size for differential evolution.
            save_json: If True, saves optimized constants to constants.json.
            
        Returns:
            OptimizationResult with optimal parameters and predictions.
        """
        if bounds is None:
            # Fixed default bounds for [beta, alpha, kappa, g1]
            # Wide enough to cover both empirical and first-principles parameter regimes
            bounds = [
                (0.0001, 0.01),       # beta (GeV) - around 0.0005
                (1.0, 50.0),          # alpha (GeV) - around 10.5
                (0.000001, 0.001),    # kappa (GeV^-2) - around 0.00003
                (1000.0, 10000.0),    # g1 (dimensionless) - around 5000
            ]
        
        # Validate that loaded initial values are within bounds
        initial_values = [
            ('beta', BETA_INITIAL, bounds[0]),
            ('alpha', ALPHA_INITIAL, bounds[1]),
            ('kappa', KAPPA_INITIAL, bounds[2]),
            ('g1', G1_INITIAL, bounds[3]),
        ]
        
        errors = []
        for name, value, (low, high) in initial_values:
            if not (low <= value <= high):
                errors.append(
                    f"  {name}: {value} is outside bounds [{low}, {high}]"
                )
        
        if errors:
            error_msg = (
                "Initial values from constants.json are outside optimization bounds:\n"
                + "\n".join(errors)
                + "\n\nPlease update constants.json with values within the valid ranges, "
                "or specify different bounds using the bounds parameter."
            )
            raise ValueError(error_msg)
        
        if self.verbose:
            print("=" * 60)
            print("4-PARAMETER OPTIMIZATION")
            print("Searching over {beta, alpha, kappa, g1} independently")
            print("=" * 60)
            print(f"\nCalibration particles: {[p.name for p in self.calibration]}")
            print(f"Validation particles: {[p.name for p in self.validation]}")
            print(f"\nParameter bounds:")
            print(f"  beta:  [{bounds[0][0]}, {bounds[0][1]}] GeV")
            print(f"  alpha: [{bounds[1][0]}, {bounds[1][1]}] GeV")
            print(f"  kappa: [{bounds[2][0]}, {bounds[2][1]}] GeV^-2")
            print(f"  g1:    [{bounds[3][0]}, {bounds[3][1]}]")
            print(f"\nStarting optimization (maxiter={maxiter}, popsize={popsize})...")
            print("-" * 60)
        
        # Reset tracking
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        self._start_time = time.time()
        
        # Initialize log file
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("SFM 4-PARAMETER OPTIMIZATION LOG\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
                f.write(f"Calibration particles: {[p.name for p in self.calibration]}\n")
                f.write(f"Validation particles: {[p.name for p in self.validation]}\n")
                f.write(f"Parameter bounds:\n")
                f.write(f"  beta:  [{bounds[0][0]}, {bounds[0][1]}] GeV\n")
                f.write(f"  alpha: [{bounds[1][0]}, {bounds[1][1]}] GeV\n")
                f.write(f"  kappa: [{bounds[2][0]}, {bounds[2][1]}] GeV^-2\n")
                f.write(f"  g1:    [{bounds[3][0]}, {bounds[3][1]}]\n")
                f.write(f"maxiter={maxiter}, popsize={popsize}\n")
                f.write(f"Initial seed (x0) from constants.json:\n")
                f.write(f"  beta={BETA_INITIAL}, alpha={ALPHA_INITIAL}, kappa={KAPPA_INITIAL}, g1={G1_INITIAL}\n")
                f.write("=" * 70 + "\n\n")
        
        start_time = time.time()
        
        # Seed initial population with known good values from constants.json
        # This ensures the optimizer starts from a reasonable baseline
        x0 = [BETA_INITIAL, ALPHA_INITIAL, KAPPA_INITIAL, G1_INITIAL]
        
        if self.verbose:
            print(f"Seeding initial population with constants.json values:")
            print(f"  x0 = [beta={x0[0]}, alpha={x0[1]}, kappa={x0[2]}, g1={x0[3]}]")
        
        # Run differential evolution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                x0=x0,  # Seed with known good values
                maxiter=maxiter,
                seed=seed,
                workers=1,  # Serial for now (solvers may not be thread-safe)
                disp=False,
                polish=True,
                tol=1e-6,
                atol=1e-6,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
            )
        
        elapsed = time.time() - start_time
        
        # Extract optimal parameters
        beta_opt, alpha_opt, kappa_opt, g1_opt = result.x
        
        # Compute L_0 from Beautiful Equation
        L0_opt = 1.0 / beta_opt
        
        if self.verbose:
            print("-" * 60)
            print(f"Optimization complete. Computing final predictions...")
        
        # Compute results for all particles
        calibration_results = {}
        for particle in self.calibration:
            m_pred = self.predict_mass(particle, beta_opt, alpha_opt, kappa_opt, g1_opt)
            m_exp = particle.mass_gev
            
            if m_pred is not None:
                percent_error = abs(m_pred - m_exp) / m_exp * 100
            else:
                m_pred = 0
                percent_error = 100
            
            calibration_results[particle.name] = {
                'predicted_mass': m_pred,
                'experimental_mass': m_exp,
                'percent_error': percent_error,
                'weight': particle.weight,
            }
        
        validation_results = {}
        for particle in self.validation:
            m_pred = self.predict_mass(particle, beta_opt, alpha_opt, kappa_opt, g1_opt)
            m_exp = particle.mass_gev
            
            if m_pred is not None:
                percent_error = abs(m_pred - m_exp) / m_exp * 100
            else:
                m_pred = 0
                percent_error = 100
            
            validation_results[particle.name] = {
                'predicted_mass': m_pred,
                'experimental_mass': m_exp,
                'percent_error': percent_error,
                'weight': particle.weight,
            }
        
        # Create result object
        opt_result = OptimizationResult(
            beta=beta_opt,
            alpha=alpha_opt,
            kappa=kappa_opt,
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
        
        # Save optimized constants to JSON file (if enabled)
        opt_result.save_constants_to_json(save=save_json)
        
        return opt_result
    
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
    
    def _objective_ratio(self, params: np.ndarray) -> float:
        """
        Objective function for ratio-based optimization.
        
        Optimizes alpha and g_internal to match masses, with beta
        computed analytically to fix the electron mass exactly.
        
        Includes leptons, mesons (pion), and baryons (proton) in calibration.
        
        Args:
            params: Array of [alpha, g_internal]
            
        Returns:
            Total weighted error in mass predictions.
        """
        alpha, g_internal = params
        g1 = G1_INITIAL  # Keep g1 fixed
        
        self._eval_count += 1
        start_time = time.time()
        
        # Compute A^2 for all calibration particles
        A_squared = {}
        
        # The solver is now completely beta-independent!
        # G_internal controls self-confinement: Δx = 1/(G_internal × A⁶)^(1/3)
        # Beta is only used to convert amplitude to physical mass at the end.
        
        from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
        
        # Create solver once (parameters are the same for all particles)
        solver = NonSeparableWavefunctionSolver(
            alpha=alpha,
            g_internal=g_internal,  # Fundamental parameter
            g1=g1,
            g2=0.004,
            V0=self.V0,
            n_max=5,
            l_max=2,
            N_sigma=64,
        )
        
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
                    A_squared[particle.name] = result.structure_norm ** 2
                    
                elif particle.particle_type == 'meson':
                    # Use self-consistent meson solver
                    # quark_wells are well indices (1, 2), not angles
                    result = solver.solve_meson_self_consistent(
                        quark_wells=(1, 2),
                        max_iter_outer=30,
                        verbose=False,
                    )
                    A_squared[particle.name] = result.structure_norm ** 2
                    
                elif particle.particle_type == 'baryon':
                    # Use self-consistent baryon solver
                    # quark_wells are well indices (1, 2, 3)
                    result = solver.solve_baryon_self_consistent(
                        quark_wells=(1, 2, 3),
                        max_iter_outer=30,
                        verbose=False,
                    )
                    A_squared[particle.name] = result.structure_norm ** 2
                    
            except Exception as e:
                # Solver failed - return large error
                return 1e20
        
        # Check we have electron
        if 'electron' not in A_squared or A_squared['electron'] <= 0:
            return 1e20
        
        # Derive beta from electron mass: beta = m_e / A_e^2
        m_e_exp = 0.000511  # GeV
        beta_derived = m_e_exp / A_squared['electron']
        
        # Now compute predicted masses and errors for all particles
        total_error = 0.0
        predictions = {}
        
        for particle in self.calibration:
            if particle.name not in A_squared:
                continue
            
            m_pred = beta_derived * A_squared[particle.name]
            m_exp = particle.mass_gev
            
            # For electron, error should be ~0 by construction
            percent_error = abs(m_pred - m_exp) / m_exp * 100
            
            predictions[particle.name] = {
                'predicted': m_pred * 1000,  # MeV
                'experimental': m_exp * 1000,  # MeV
                'error': percent_error,
            }
            
            # Weight the error (electron should be perfect by construction)
            if particle.name == 'electron':
                total_error += percent_error * 0.01  # Low weight
            else:
                total_error += percent_error * particle.weight
        
        elapsed = time.time() - start_time
        
        # Track best result
        is_best = total_error < self._best_error
        if is_best:
            self._best_error = total_error
            self._best_params = (alpha, g_internal, beta_derived, g1)
        
        # Log progress
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"Eval {self._eval_count} | Time: {elapsed:.1f}s | ")
                f.write("*** NEW BEST ***\n" if is_best else "\n")
                f.write(f"Search params: alpha={alpha:.6f}, g_internal={g_internal:.8f}\n")
                f.write(f"Derived beta: {beta_derived:.8f} GeV\n")
                f.write(f"Total Error: {total_error:.6f} (best so far: {self._best_error:.6f})\n")
                f.write(f"Particle predictions:\n")
                for name, pred in predictions.items():
                    f.write(f"  {name:12}: pred={pred['predicted']:.4f} MeV, ")
                    f.write(f"exp={pred['experimental']:.2f} MeV, ")
                    f.write(f"error={pred['error']:.2f}%\n")
        
        if self.verbose and (self._eval_count % 10 == 0 or is_best):
            marker = " *** BEST ***" if is_best else ""
            print(f"Eval {self._eval_count}: alpha={alpha:.4f}, g_int={g_internal:.6f}, "
                  f"beta={beta_derived:.6e}, error={total_error:.2f}%{marker}")
        
        return total_error
    
    def optimize_ratio(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        maxiter: int = 100,
        seed: int = 42,
        popsize: int = 10,
        save_json: bool = True,
    ) -> OptimizationResult:
        """
        Ratio-based optimization: optimize alpha and kappa to match mass RATIOS.
        
        Beta is derived analytically at each step to fix electron mass exactly:
            beta = m_e_experimental / A_electron^2
        
        This reduces the problem from 4 free parameters to 2, focusing on
        what matters: the mass hierarchy between generations.
        
        Args:
            bounds: List of (min, max) bounds for [alpha, kappa].
            maxiter: Maximum iterations.
            seed: Random seed for reproducibility.
            popsize: Population size for differential evolution.
            save_json: If True, saves optimized constants to constants.json.
            
        Returns:
            OptimizationResult with optimal parameters and predictions.
        """
        if bounds is None:
            # Bounds for [alpha, g_internal] only
            # TIGHTENED around known good values: alpha=10.5, g_internal=0.003
            # The solver is highly sensitive to g_internal, so keep bounds narrow
            bounds = [
                (8.0, 15.0),          # alpha: ±50% around 10.5
                (0.001, 0.01),        # g_internal: ±3× around 0.003
            ]
        
        if self.verbose:
            print("=" * 70)
            print("RATIO-BASED OPTIMIZATION")
            print("Optimizing {alpha, g_internal} to match mass RATIOS")
            print("Beta derived from electron: beta = m_e / A_e^2")
            print("g_internal is FUNDAMENTAL (works with amplitude, not mass)")
            print("Solver is completely beta-independent!")
            print("=" * 70)
            print(f"\nCalibration particles: {[p.name for p in self.calibration]}")
            print(f"Validation particles: {[p.name for p in self.validation]}")
            print(f"\nParameter bounds:")
            print(f"  alpha: [{bounds[0][0]}, {bounds[0][1]}] GeV")
            print(f"  g_internal: [{bounds[1][0]}, {bounds[1][1]}]")
            print(f"\nFixed parameters:")
            print(f"  g1: {G1_INITIAL}")
            print(f"\nStarting optimization (maxiter={maxiter}, popsize={popsize})...")
            print("-" * 70)
        
        # Reset tracking
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        self._start_time = time.time()
        
        # Initialize log file
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("SFM RATIO-BASED OPTIMIZATION LOG\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
                f.write("Strategy: Optimize alpha/g_internal for mass RATIOS\n")
                f.write("Beta derived from electron: beta = m_e / A_e^2\n")
                f.write("g_internal is FUNDAMENTAL (works with amplitude, not mass)\n")
                f.write("Solver is completely beta-independent!\n")
                f.write("=" * 70 + "\n")
                f.write(f"Calibration particles: {[p.name for p in self.calibration]}\n")
                f.write(f"Validation particles: {[p.name for p in self.validation]}\n")
                f.write(f"Parameter bounds:\n")
                f.write(f"  alpha: [{bounds[0][0]}, {bounds[0][1]}] GeV\n")
                f.write(f"  g_internal: [{bounds[1][0]}, {bounds[1][1]}]\n")
                f.write(f"Fixed: g1={G1_INITIAL}\n")
                f.write(f"maxiter={maxiter}, popsize={popsize}\n")
                f.write("=" * 70 + "\n\n")
        
        start_time = time.time()
        
        # Seed initial population with known good values
        x0 = [ALPHA_INITIAL, G_INTERNAL_INITIAL]
        
        if self.verbose:
            print(f"Starting from: alpha={x0[0]}, g_internal={x0[1]}")
        
        # Use LOCAL optimizer (Nelder-Mead) since we already have a good starting point
        # This avoids the wild exploration that differential_evolution does
        from scipy.optimize import minimize
        
        # maxfev = max function evaluations (more intuitive for user)
        # Each user "iteration" = ~5 function evaluations for Nelder-Mead
        max_func_evals = maxiter * 5
        
        if self.verbose:
            print(f"Running Nelder-Mead local optimizer (max {max_func_evals} evaluations)...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._objective_ratio,
                x0=x0,
                method='Nelder-Mead',
                options={
                    'maxfev': max_func_evals,  # Limit function evaluations
                    'xatol': 1e-4,
                    'fatol': 0.1,  # Stop if error changes by < 0.1%
                    'disp': False,
                },
            )
        
        elapsed = time.time() - start_time
        
        # Extract optimal parameters
        alpha_opt, g_internal_opt = result.x
        g1_opt = G1_INITIAL
        
        # Derive beta from electron mass using optimal parameters
        try:
            from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
            
            solver = NonSeparableWavefunctionSolver(
                alpha=alpha_opt,
                g_internal=g_internal_opt,
                g1=g1_opt,
                g2=0.004,
                V0=self.V0,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
            
            result_e = solver.solve_lepton_self_consistent(
                n_target=1,  # electron
                k_winding=1,
                max_iter_outer=30,
                max_iter_nl=0,
                verbose=False,
            )
            
            A_e_squared = result_e.structure_norm ** 2
            beta_opt = 0.000511 / A_e_squared  # m_e in GeV
            
        except Exception as e:
            print(f"Warning: Failed to derive beta: {e}")
            beta_opt = BETA_INITIAL
        
        L0_opt = 1.0 / beta_opt
        
        # Derive kappa for backward compatibility: kappa = g_internal / beta
        kappa_opt = g_internal_opt / beta_opt
        
        if self.verbose:
            print("-" * 70)
            print(f"Optimization complete. Computing final predictions...")
            print(f"Optimal: alpha={alpha_opt:.6f}, g_internal={g_internal_opt:.8f}")
            print(f"Derived: beta={beta_opt:.8f} GeV, kappa={kappa_opt:.8f}")
        
        # Compute results for all particles
        calibration_results = self._evaluate_particles(
            self.calibration, beta_opt, alpha_opt, kappa_opt, g1_opt
        )
        validation_results = self._evaluate_particles(
            self.validation, beta_opt, alpha_opt, kappa_opt, g1_opt
        )
        
        # Create result object
        opt_result = OptimizationResult(
            beta=beta_opt,
            alpha=alpha_opt,
            kappa=kappa_opt,  # Derived for backward compatibility
            g1=g1_opt,
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
        
        # Save optimized constants to JSON file (if enabled)
        opt_result.save_constants_to_json(save=save_json)
        
        return opt_result

    def _objective_free(self, params: np.ndarray) -> float:
        """
        Objective function for free 3-parameter optimization.
        
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
                    max_iter_outer=30,
                    max_iter_nl=0,
                    verbose=False,
                )
                
                A_squared[particle.name] = result.structure_norm ** 2
                
            except Exception as e:
                return 1e20
        
        if 'electron' not in A_squared or A_squared['electron'] <= 0:
            return 1e20
        
        # Derive beta from electron mass
        m_e_exp = 0.000511  # GeV
        beta_derived = m_e_exp / A_squared['electron']
        
        # Compute errors
        total_error = 0.0
        predictions = {}
        
        for particle in self.calibration:
            if particle.particle_type != 'lepton':
                continue
            
            if particle.name not in A_squared:
                continue
            
            m_pred = beta_derived * A_squared[particle.name]
            m_exp = particle.mass_gev
            percent_error = abs(m_pred - m_exp) / m_exp * 100
            
            predictions[particle.name] = {
                'predicted': m_pred * 1000,
                'experimental': m_exp * 1000,
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

    def optimize_free(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        maxiter: int = 100,
        seed: int = 42,
        popsize: int = 10,
        save_json: bool = True,
    ) -> OptimizationResult:
        """
        Free 3-parameter optimization over alpha, g_internal, and g1.
        
        Beta is derived from electron mass after solving.
        
        Args:
            bounds: List of (min, max) bounds for [alpha, g_internal, g1].
            maxiter: Maximum iterations.
            seed: Random seed for reproducibility.
            popsize: Population size.
            save_json: If True, saves optimized constants.
            
        Returns:
            OptimizationResult with optimal parameters.
        """
        if bounds is None:
            bounds = [
                (5.0, 30.0),          # alpha
                (0.0001, 0.1),        # g_internal
                (100.0, 20000.0),     # g1
            ]
        
        if self.verbose:
            print("=" * 70)
            print("FREE 3-PARAMETER OPTIMIZATION")
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
                f.write("SFM FREE 3-PARAMETER OPTIMIZATION\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
        
        x0 = [ALPHA_INITIAL, G_INTERNAL_INITIAL, G1_INITIAL]
        
        if self.verbose:
            print(f"Seeding: alpha={x0[0]}, g_internal={x0[1]}, g1={x0[2]}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                self._objective_free,
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
                n_target=1, k_winding=1, max_iter_outer=30, verbose=False
            )
            
            A_e_squared = result_e.structure_norm ** 2
            beta_opt = 0.000511 / A_e_squared
            
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


def run_optimization_ratio(
    verbose: bool = True,
    maxiter: int = 100,
    log_file: Optional[str] = None,
    save_json: bool = True,
) -> OptimizationResult:
    """
    Run ratio-based optimization over alpha and kappa.
    
    Beta is derived analytically to fix electron mass exactly.
    This focuses optimization on matching mass RATIOS (m_mu/m_e, m_tau/m_e).
    
    Args:
        verbose: Print progress information.
        maxiter: Maximum iterations.
        log_file: Path to log file for optimization progress (optional).
        save_json: If True, saves optimized constants to constants.json.
    
    Returns:
        OptimizationResult with optimal parameters.
    """
    optimizer = SFMParameterOptimizer(verbose=verbose, log_file=log_file)
    return optimizer.optimize_ratio(maxiter=maxiter, save_json=save_json)


def run_optimization_beta_only(
    verbose: bool = True,
    maxiter: int = 100,
    beta_bounds: Tuple[float, float] = (10.0, 500.0),
    log_file: Optional[str] = None,
    save_json: bool = True,
) -> OptimizationResult:
    """
    Run FIRST-PRINCIPLES beta-only optimization.
    
    All parameters derived from beta:
        kappa = 8piGbeta/(hbarc^3)
        g_1 = alpha_em * beta^2 / m_e
        alpha = 1/beta
    
    Args:
        verbose: Print progress information.
        maxiter: Maximum iterations.
        beta_bounds: Search range for beta in GeV. Default is (10.0, 500.0).
        log_file: Path to log file for optimization progress (optional).
        save_json: If True, saves optimized constants to constants.json.
        
    Returns:
        OptimizationResult with optimal beta and derived parameters.
        
    Raises:
        ValueError: If the initial beta from constants.json is outside bounds.
    """
    optimizer = SFMParameterOptimizer(verbose=verbose, log_file=log_file)
    return optimizer.optimize_beta_only(beta_bounds=beta_bounds, maxiter=maxiter, save_json=save_json)


def run_optimization_free(
    verbose: bool = True,
    maxiter: int = 100,
    log_file: Optional[str] = None,
    save_json: bool = True,
) -> OptimizationResult:
    """
    Run 3-parameter optimization over alpha, g_internal, and g1.
    
    Beta is derived from the electron mass after solving.
    
    Args:
        verbose: Print progress information.
        maxiter: Maximum iterations.
        log_file: Path to log file for optimization progress (optional).
        save_json: If True, saves optimized constants to constants.json.
    
    Returns:
        OptimizationResult with optimal parameters.
    """
    optimizer = SFMParameterOptimizer(verbose=verbose, log_file=log_file)
    return optimizer.optimize_free(maxiter=maxiter, save_json=save_json)


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
  python parameter_optimizer.py                     # ratio mode (default), save to JSON
  python parameter_optimizer.py --mode ratio        # ratio mode - optimize alpha/g_internal for mass ratios
  python parameter_optimizer.py --mode free         # free mode - optimize alpha, g_internal, g1
  python parameter_optimizer.py --save-json off     # Don't save to constants.json
  python parameter_optimizer.py --max-iter 200      # Run 200 iterations
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ratio", "free"],
        default="ratio",
        help="Optimization mode: 'ratio' (default) optimizes alpha/g_internal for mass ratios, "
             "'free' for 3-parameter (alpha, g_internal, g1)"
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
    
    args = parser.parse_args()
    
    # Determine save_json setting
    save_json = args.save_json == "on"
    verbose = not args.quiet
    
    # Create log file path
    log_file = create_log_file_path()
    
    # Map mode names for display
    mode_names = {
        'ratio': 'ratio-based (alpha/g_internal)',
        'free': '3-parameter (alpha, g_internal, g1)',
    }
    
    # Print startup information
    print("=" * 70)
    print("SFM PARAMETER OPTIMIZER")
    print("=" * 70)
    print(f"Mode: {mode_names.get(args.mode, args.mode)}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Save to constants.json: {'Yes' if save_json else 'No'}")
    print(f"Log file: {log_file}")
    print()
    print(f"Initial values from constants.json:")
    print(f"  alpha      = {ALPHA_INITIAL}")
    print(f"  g_internal = {G_INTERNAL_INITIAL}")
    print(f"  g1         = {G1_INITIAL}")
    print("=" * 70)
    print()
    
    # Run the appropriate optimization
    if args.mode == "ratio":
        result = run_optimization_ratio(
            verbose=verbose,
            maxiter=args.max_iter,
            log_file=log_file,
            save_json=save_json,
        )
    else:  # free
        result = run_optimization_free(
            verbose=verbose,
            maxiter=args.max_iter,
            log_file=log_file,
            save_json=save_json,
        )
    
    # Print final summary
    print("\n" + result.summary())
    print(f"\nLog file saved to: {log_file}")


if __name__ == '__main__':
    main()
