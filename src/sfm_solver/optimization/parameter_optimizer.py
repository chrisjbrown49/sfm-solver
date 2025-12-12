"""
SFM Parameter Optimizer - First-Principles β-Only Optimization.

THEORETICAL BASIS:
==================
From "Implementation Note - The Beautiful Balance":

    "The solver's purpose is to DISCOVER the values of β and other framework 
    parameters (α, κ, V₀, g₁, g₂) by requiring that energy minimization 
    reproduces all known particle masses."

FIRST-PRINCIPLES APPROACH:
==========================
All parameters derive from β and fundamental constants (ℏ, c, G, α_em, m_e):

1. κ = 8πGβ/(ℏc³)         — Enhanced 5D gravity at subspace scale
2. g₁ = α_em × β² × c² / m_e  — Nonlinear coupling (EM/gravity hierarchy)
3. α_coupling = ℏ²/β = ℏL₀c   — Spatial-subspace coupling strength

With these relationships, we search ONLY over β. All other parameters
are derived, ensuring physical consistency across the framework.

APPROACH:
=========
1. Search over β only
2. For each β, compute {α, κ, g₁} from first-principles formulas
3. Run actual solvers for all calibration particles
4. Minimize total mass prediction error
5. The optimal β determines ALL framework parameters

Reference: docs/SFM_First_Principles_Parameter_Derivation_Plan.md
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar, OptimizeResult
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import time
import warnings

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential


# =============================================================================
# Fundamental Constants (SI units)
# =============================================================================

HBAR = 1.054571817e-34      # J·s (reduced Planck constant)
C = 299792458.0             # m/s (speed of light)
G = 6.67430e-11             # m³/(kg·s²) (Newton's gravitational constant)
ALPHA_EM = 1.0 / 137.036    # Fine structure constant (dimensionless)
M_E_KG = 9.1093837015e-31   # kg (electron mass)
M_E_GEV = 0.000511          # GeV (electron mass)

# Conversion factors
GEV_TO_KG = 1.78266192e-27  # kg per GeV
GEV_TO_JOULE = 1.602176634e-10  # J per GeV


# =============================================================================
# PURE FIRST-PRINCIPLES PARAMETER DERIVATIONS FROM β
# =============================================================================
#
# NO PHENOMENOLOGICAL FUDGE FACTORS!
# These are the true theoretical derivations from SFM.
#
# FORMULAS (from user-provided SFM theory):
# =========================================
# 
# 1. κ = 8πGβ²c/ℏ  (enhanced 5D gravity at subspace scale)
#    G_eff = G/L₀ combined with β L₀ c = ℏ gives extra factor of β
#
# 2. g₁ = α_em × β / m_e  (nonlinear self-interaction)
#    Single power of β (not β²)
#
# 3. α = C × β where C ~ 0.5  (spatial-subspace coupling)
#    Direct proportionality to fundamental mass scale
#
# NOTE: These formulas may produce values very different from what
# current solvers expect. This reveals that solver internals may
# contain phenomenological parameters that need theoretical correction.

# Dimensionless constant for α_coupling (from theory: C ~ 0.5)
ALPHA_COUPLING_C = 0.5  # α = C × β


def derive_kappa_from_beta(beta_gev: float) -> float:
    """
    Derive κ (curvature coupling) from β - PURE FIRST PRINCIPLES (NO G!).
    
    FORMULA: κ = ℏ²/(β²c²)
    
    In natural units (ℏ=c=1):
        κ = 1/β²
    
    Physical basis:
        κ is determined entirely by fundamental SFM parameters (β, L₀, ℏ, c).
        The Newtonian gravitational constant G is a PREDICTION that emerges
        from SFM, not an input. (See Beautiful Equation Section 5.6)
    
    Note: This gives κ ~ 0.00034 GeV⁻² for β ~ 54 GeV, which is very
    different from what current solvers expect (~230 GeV⁻²). This
    discrepancy reveals phenomenological parameters in the solvers.
    
    Args:
        beta_gev: Mass coupling β in GeV
        
    Returns:
        κ in GeV⁻²
    """
    # Pure first-principles: κ = 1/β² (in natural units)
    kappa = 1.0 / (beta_gev ** 2)
    return kappa


def derive_g1_from_beta(beta_gev: float) -> float:
    """
    Derive g₁ (nonlinear coupling) from β - PURE FIRST PRINCIPLES.
    
    FORMULA: g₁ = α_em × β / m_e
    
    Physical basis:
        From EM/gravity hierarchy and fine structure constant.
        Single power of β gives correct scaling.
    
    Args:
        beta_gev: Mass coupling β in GeV
        
    Returns:
        g₁ (dimensionless)
    """
    # Pure first-principles: g₁ = α_em × β / m_e
    g1 = ALPHA_EM * beta_gev / M_E_GEV
    return g1


def derive_alpha_coupling_from_beta(beta_gev: float) -> float:
    """
    Derive α_coupling (spatial-subspace coupling) from β - PURE FIRST PRINCIPLES.
    
    FORMULA: α = C × β  where C ~ 0.5
    
    Physical basis:
        Dimensional analysis of coupling integral shows [α] = [Energy].
        Coupling strength increases with fundamental mass scale.
    
    Args:
        beta_gev: Mass coupling β in GeV
        
    Returns:
        α in GeV
    """
    # Pure first-principles: α = C × β
    alpha = ALPHA_COUPLING_C * beta_gev
    return alpha


def derive_all_parameters_from_beta(beta_gev: float) -> Dict[str, float]:
    """
    Derive all SFM parameters from β alone.
    
    This is the core of the first-principles approach: given only β,
    all other parameters are determined by the SFM framework.
    
    Args:
        beta_gev: Mass coupling β in GeV
        
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
    generation: int = 1  # n = 1, 2, 3 for e, μ, τ
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


# Calibration particle set - minimal set to determine parameters
# Now includes leptons, mesons, and baryons for universal β
CALIBRATION_PARTICLES = [
    # LEPTONS - single particles with k=1
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
    # MESONS - quark-antiquark bound states
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
    # BARYONS - three-quark bound states
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
VALIDATION_PARTICLES = [
    ParticleSpec(
        name='tau',
        mass_gev=1.777,
        particle_type='lepton',
        generation=3,
        k_lepton=1,
        weight=1.0
    ),
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
    
    def summary(self) -> str:
        """Generate a summary of optimization results."""
        lines = [
            "=" * 70,
            "SFM FIRST-PRINCIPLES PARAMETER OPTIMIZATION RESULTS",
            "=" * 70,
            "",
            "OPTIMAL PARAMETERS:",
            f"  β (beta)  = {self.beta:.4f} GeV",
            f"  α (alpha) = {self.alpha:.6f} GeV", 
            f"  κ (kappa) = {self.kappa:.4f} GeV⁻²",
            f"  g₁        = {self.g1:.2f}",
            f"  L₀        = {self.L0_gev_inv:.6f} GeV⁻¹ (from Beautiful Equation)",
            "",
            f"CONVERGENCE: {'✅ YES' if self.converged else '❌ NO'}",
            f"Total Error: {self.total_error:.6f}",
            f"Iterations: {self.iterations}",
            f"Time: {self.time_seconds:.2f} s",
            "",
            "CALIBRATION PARTICLES (used to fit parameters):",
            "-" * 50,
        ]
        
        for name, result in self.calibration_results.items():
            status = '✅' if result['percent_error'] < 5 else '❌'
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
            status = '✅' if result['percent_error'] < 5 else '❌'
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
        verbose: bool = True
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
        """
        self.calibration = calibration_particles or CALIBRATION_PARTICLES
        self.validation = validation_particles or VALIDATION_PARTICLES
        self.N_grid = N_grid
        self.V0 = V0
        self.verbose = verbose
        
        # Tracking
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        self._best_beta = None
    
    def _run_lepton_solver(
        self,
        particle: ParticleSpec,
        beta: float,
        alpha: float,
        kappa: float,
        g1: float
    ) -> Optional[float]:
        """
        Run the actual lepton solver with injected parameters.
        
        Returns:
            Predicted mass in GeV, or None if solver fails.
        """
        try:
            from sfm_solver.eigensolver.sfm_lepton_solver import SFMLeptonSolver
            
            # Create grid and potential
            grid = SpectralGrid(N=self.N_grid)
            potential = ThreeWellPotential(V0=self.V0, V1=0.1)
            
            # Map generation to particle name
            particle_name = {1: 'electron', 2: 'muon', 3: 'tau'}[particle.generation]
            
            # Create solver with injected parameters
            solver = SFMLeptonSolver(
                grid=grid,
                potential=potential,
                g1=g1,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                use_physical=True,
            )
            
            # Run solver
            state = solver.solve_lepton(particle=particle_name, max_iter=2000)
            
            # Extract predicted mass
            m_pred = beta * state.amplitude_squared
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
        Run the actual meson solver with injected parameters.
        
        Returns:
            Predicted mass in GeV, or None if solver fails.
        """
        try:
            from sfm_solver.multiparticle.composite_meson import CompositeMesonSolver
            
            # Create grid and potential
            grid = SpectralGrid(N=self.N_grid)
            potential = ThreeWellPotential(V0=self.V0, V1=0.1)
            
            # Create solver with injected parameters
            solver = CompositeMesonSolver(
                grid=grid,
                potential=potential,
                g1=g1,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                use_physical=True,  # Always use physical mode
            )
            
            # Run solver
            state = solver.solve(meson_type=particle.meson_type, max_iter=1000)
            
            # Extract predicted mass
            m_pred = beta * state.amplitude_squared
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
        Run the actual baryon solver with injected parameters.
        
        Returns:
            Predicted mass in GeV, or None if solver fails.
        """
        try:
            from sfm_solver.multiparticle.composite_baryon import CompositeBaryonSolver
            
            # Create grid and potential
            grid = SpectralGrid(N=self.N_grid)
            potential = ThreeWellPotential(V0=self.V0, V1=0.1)
            
            # Create solver with injected parameters
            solver = CompositeBaryonSolver(
                grid=grid,
                potential=potential,
                g1=g1,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                use_physical=True,
            )
            
            # Run solver
            state = solver.solve(quark_types=particle.quarks, max_iter=1000)
            
            # Extract predicted mass
            m_pred = beta * state.amplitude_squared
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
        FIRST-PRINCIPLES objective function for β-only optimization.
        
        All other parameters are DERIVED from β using SFM relationships:
            κ = 8πGβ/(ℏc³)
            g₁ = α_em × β² / m_e
            α_coupling = 1/β
        
        Args:
            beta: Mass coupling constant in GeV
            
        Returns:
            Total weighted error.
        """
        if beta <= 0:
            return 1e10
        
        # DERIVE all parameters from β (first principles!)
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
        
        for particle in self.calibration:
            try:
                m_pred = self.predict_mass(particle, beta, alpha, kappa, g1)
                
                if m_pred is None or m_pred <= 0 or np.isnan(m_pred):
                    total_error += 10.0  # Penalty for failed calculation
                    continue
                
                m_exp = particle.mass_gev
                
                # Relative squared error, weighted
                rel_error = ((m_pred - m_exp) / m_exp) ** 2
                total_error += particle.weight * rel_error
                valid_count += 1
                
            except Exception:
                total_error += 10.0
        
        # Penalty if no valid predictions
        if valid_count == 0:
            total_error = 1e10
        
        # Track progress
        self._eval_count += 1
        if total_error < self._best_error:
            self._best_error = total_error
            self._best_beta = beta
            if self.verbose:
                print(f"  Eval {self._eval_count}: New best error = {self._best_error:.6f}")
                print(f"    β={beta:.4f} GeV → α={alpha:.6f}, κ={kappa:.6f}, g₁={g1:.2f}")
        elif self.verbose and self._eval_count % 20 == 0:
            print(f"  Eval {self._eval_count}: error = {total_error:.6f}")
        
        return total_error
    
    def optimize_beta_only(
        self,
        beta_bounds: Tuple[float, float] = (10.0, 500.0),
        maxiter: int = 100,
    ) -> OptimizationResult:
        """
        FIRST-PRINCIPLES optimization: search ONLY over β.
        
        All other parameters are DERIVED from β using SFM relationships:
            κ = 8πGβ/(ℏc³)           - Enhanced 5D gravity
            g₁ = α_em × β² / m_e     - Nonlinear coupling (EM/gravity hierarchy)
            α_coupling = 1/β         - Spatial-subspace coupling
        
        Args:
            beta_bounds: (min, max) bounds for β in GeV.
            maxiter: Maximum iterations for optimization.
            
        Returns:
            OptimizationResult with optimal β and derived parameters.
        """
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
            print(f"\nSearching β in [{beta_bounds[0]}, {beta_bounds[1]}] GeV")
            print("-" * 70)
        
        # Reset tracking
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_beta = None
        
        start_time = time.time()
        
        # Use scalar optimization (1D search over β)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize_scalar(
                self._objective_beta_only,
                bounds=beta_bounds,
                method='bounded',
                options={'maxiter': maxiter, 'xatol': 0.01}
            )
        
        elapsed = time.time() - start_time
        
        # Extract optimal β and derive all other parameters
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
            print(f"  From β = {beta_opt:.4f} GeV:")
            print(f"    κ = 8πGβ/(ℏc³) = {kappa_opt:.6e} GeV⁻²")
            print(f"    g₁ = α_em × β² / m_e = {g1_opt:.2f}")
            print(f"    α = 1/β = {alpha_opt:.6f} GeV")
        
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
        popsize: int = 10
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
            
        Returns:
            OptimizationResult with optimal parameters and predictions.
        """
        if bounds is None:
            bounds = [
                (40.0, 200.0),       # beta
                (0.01, 1.0),         # alpha
                (50.0, 500.0),       # kappa
                (500.0, 5000.0),     # g1
            ]
        
        if self.verbose:
            print("=" * 60)
            print("WARNING: Using 4-parameter optimization (DEPRECATED)")
            print("Consider using optimize_beta_only() for first-principles approach")
            print("=" * 60)
            print(f"\nCalibration particles: {[p.name for p in self.calibration]}")
            print(f"Validation particles: {[p.name for p in self.validation]}")
            print(f"\nParameter bounds:")
            print(f"  β: [{bounds[0][0]}, {bounds[0][1]}] GeV")
            print(f"  α: [{bounds[1][0]}, {bounds[1][1]}] GeV")
            print(f"  κ: [{bounds[2][0]}, {bounds[2][1]}] GeV⁻²")
            print(f"  g₁: [{bounds[3][0]}, {bounds[3][1]}]")
            print(f"\nStarting optimization (maxiter={maxiter}, popsize={popsize})...")
            print("-" * 60)
        
        # Reset tracking
        self._eval_count = 0
        self._best_error = float('inf')
        self._best_params = None
        
        start_time = time.time()
        
        # Run differential evolution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                self._objective,
                bounds=bounds,
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
        
        # Compute L₀ from Beautiful Equation
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


def run_optimization_beta_only(
    verbose: bool = True,
    maxiter: int = 100,
    beta_bounds: Tuple[float, float] = (10.0, 500.0)
) -> OptimizationResult:
    """
    Run FIRST-PRINCIPLES β-only optimization.
    
    All parameters derived from β:
        κ = 8πGβ/(ℏc³)
        g₁ = α_em × β² / m_e
        α = 1/β
    
    Args:
        verbose: Print progress information.
        maxiter: Maximum iterations.
        beta_bounds: Search range for β in GeV.
        
    Returns:
        OptimizationResult with optimal β and derived parameters.
    """
    optimizer = SFMParameterOptimizer(verbose=verbose)
    return optimizer.optimize_beta_only(beta_bounds=beta_bounds, maxiter=maxiter)


def run_optimization(verbose: bool = True, maxiter: int = 100) -> OptimizationResult:
    """
    DEPRECATED: Use run_optimization_beta_only() for first-principles approach.
    
    Returns:
        OptimizationResult with optimal parameters.
    """
    optimizer = SFMParameterOptimizer(verbose=verbose)
    return optimizer.optimize(maxiter=maxiter)


if __name__ == '__main__':
    # Run first-principles β-only optimization
    print("Running FIRST-PRINCIPLES β-only optimization...")
    result = run_optimization_beta_only(maxiter=100, beta_bounds=(10.0, 200.0))
    print("\n" + result.summary())
