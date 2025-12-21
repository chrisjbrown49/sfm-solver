"""
Unified SFM Solver - Two-Stage Architecture.

This module provides the main interface for the new two-stage solver:
    Stage 1: Solve for normalized wavefunction shape (dimensionless)
    Stage 2: Minimize energy over scale parameters (Delta_x, Delta_sigma, A)

The unified solver combines shape_solver.py, spatial_coupling.py, and
energy_minimizer.py into a single easy-to-use interface.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings

from sfm_solver.core.shape_solver import DimensionlessShapeSolver, DimensionlessShapeResult
from sfm_solver.core.spatial_coupling import SpatialCouplingBuilder
from sfm_solver.core.energy_minimizer import UniversalEnergyMinimizer, EnergyMinimizationResult
from sfm_solver.core.constants import (
    ALPHA as ALPHA_DEFAULT,
    G_INTERNAL as G_INTERNAL_DEFAULT,
    G1 as G1_DEFAULT,
    G2 as G2_DEFAULT,
    V0 as V0_DEFAULT,
    V1 as V1_DEFAULT,
    BETA as BETA_DEFAULT
)


@dataclass
class UnifiedSolverResult:
    """
    Result from unified two-stage solver.
    
    Contains shape, scale parameters, and predicted mass.
    """
    # Predicted mass
    mass: float  # GeV
    
    # Scale parameters
    A: float
    Delta_x: float
    Delta_sigma: float
    
    # Shape structure {(n,l,m): chi_nlm(sigma)}
    shape_structure: Dict[Tuple[int, int, int], NDArray]
    
    # Energy breakdown
    energy_components: Dict[str, float]
    E_total: float
    
    # Convergence info
    shape_converged: bool
    shape_iterations: int
    energy_converged: bool
    energy_iterations: int
    
    # Particle info
    particle_type: str
    quantum_numbers: Dict


class UnifiedSFMSolver:
    """
    Two-stage solver: Shape → Scale.
    
    This is the new architecture implementing proper separation of
    quantum (shape) and classical (scale) physics.
    
    By default, loads fundamental constants from constants.json.
    
    Usage:
        # Use defaults from constants.json
        solver = UnifiedSFMSolver()
        
        # Or override specific parameters
        solver = UnifiedSFMSolver(beta=0.001, verbose=True)
        
        # Solve for a particle
        result = solver.solve_baryon(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3)
        )
        print(f"Predicted mass: {result.mass} GeV")
    """
    
    def __init__(
        self,
        beta: Optional[float] = None,
        g_internal: Optional[float] = None,
        g1: Optional[float] = None,
        g2: Optional[float] = None,
        V0: Optional[float] = None,
        V1: Optional[float] = None,
        alpha: Optional[float] = None,
        n_max: int = 5,
        l_max: int = 2,
        N_sigma: int = 64,
        auto_calibrate_beta: bool = False,
        verbose: bool = False
    ):
        """
        Initialize unified solver.
        
        By default, loads fundamental constants from constants.json.
        All parameters can be overridden by passing explicit values.
        
        Args:
            beta: Mass conversion factor (m = beta * A^2). 
                  If None and auto_calibrate_beta=False, loads from constants.json.
                  If None and auto_calibrate_beta=True, will be calibrated from electron.
            g_internal: Gravitational self-confinement (FUNDAMENTAL).
                        If None, loads from constants.json (default: G_INTERNAL)
            g1: Nonlinear coupling.
                If None, loads from constants.json (default: G1)
            g2: EM coupling.
                If None, loads from constants.json (default: G2)
            V0: Three-well primary depth (energy reference).
                If None, loads from constants.json (default: V0)
            V1: Three-well secondary depth.
                If None, loads from constants.json (default: V1)
            alpha: Spatial-subspace coupling.
                   If None, loads from constants.json (default: ALPHA)
            n_max: Maximum spatial quantum number
            l_max: Maximum angular momentum
            N_sigma: Grid points in subspace
            auto_calibrate_beta: If True and beta is None, automatically calibrate
                                beta from electron mass on initialization.
            verbose: Print diagnostic information
        """
        # Load defaults from constants.json if not provided
        # For beta: keep as None if not provided (either for auto-calibration or manual calibration)
        self.beta = beta  # Can be None (will be calibrated if auto_calibrate_beta=True)
        
        self.g_internal = g_internal if g_internal is not None else G_INTERNAL_DEFAULT
        self.g1 = g1 if g1 is not None else G1_DEFAULT
        self.g2 = g2 if g2 is not None else G2_DEFAULT
        self.V0 = V0 if V0 is not None else V0_DEFAULT
        self.V1 = V1 if V1 is not None else V1_DEFAULT
        self.alpha = alpha if alpha is not None else ALPHA_DEFAULT
        self.n_max = n_max
        self.l_max = l_max
        self.N_sigma = N_sigma
        self.auto_calibrate_beta = auto_calibrate_beta
        self.verbose = verbose
        
        # Initialize Stage 1: Shape solver (dimensionless)
        self.shape_solver = DimensionlessShapeSolver(
            g1_dimensionless=self.g1 / self.V0,
            g2_dimensionless=self.g2 / self.V0,
            V0=self.V0,
            V1=self.V1,
            N_sigma=N_sigma,
            verbose=verbose
        )
        
        # Initialize spatial coupling builder
        self.spatial_coupling = SpatialCouplingBuilder(
            alpha_dimensionless=self.alpha / self.V0,
            n_max=n_max,
            l_max=l_max,
            verbose=verbose
        )
        
        # Initialize Stage 2: Energy minimizer (with dimensions)
        self.energy_minimizer = UniversalEnergyMinimizer(
            g_internal=self.g_internal,
            g1=self.g1,
            g2=self.g2,
            V0=self.V0,
            V1=self.V1,
            alpha=self.alpha,
            beta=self.beta if not auto_calibrate_beta else None,
            use_scaled_energy=True,
            verbose=verbose
        )
        
        if self.verbose:
            print("=== UnifiedSFMSolver Initialized ===")
            print(f"  Parameters loaded from constants.json")
            if self.beta is not None:
                print(f"  beta = {self.beta:.6f} GeV")
            else:
                print(f"  beta = Not yet calibrated (will use beta-independent formulation)")
            print(f"  alpha = {self.alpha:.6f} GeV")
            print(f"  g_internal = {self.g_internal:.6f} (FUNDAMENTAL)")
            print(f"  g1 = {self.g1:.3f}")
            print(f"  g2 = {self.g2:.6f}")
            print(f"  V0 = {self.V0:.3f} GeV")
            print(f"  V1 = {self.V1:.3f} GeV")
            print(f"  Two-stage architecture ready")
            print(f"  Stage 1: Shape solver (dimensionless)")
            print(f"  Stage 2: Energy minimizer (physical scales)")
        
        # Auto-calibrate beta if requested
        if auto_calibrate_beta:
            if self.verbose:
                print(f"\n  Auto-calibrating beta from electron mass...")
            self.calibrate_beta_from_electron(verbose=verbose)
    
    def solve_baryon(
        self,
        quark_windings: Tuple[int, int, int],
        color_phases: Tuple[float, float, float] = (0, 2*np.pi/3, 4*np.pi/3),
        n_target: int = 1,
        max_scf_iter: int = 500,
        scf_tol: float = 1e-4,
        scf_mixing: float = 0.1,
        initial_scale_guess: Optional[Tuple[float, float, float]] = None
    ) -> UnifiedSolverResult:
        """
        Solve for baryon using two-stage approach.
        
        Args:
            quark_windings: (k1, k2, k3) winding numbers
            color_phases: (phi1, phi2, phi3) for color neutrality
            n_target: Target spatial quantum number (usually 1 for baryons)
            max_scf_iter: Maximum SCF iterations for shape
            scf_tol: Convergence tolerance for shape
            scf_mixing: Mixing parameter for SCF stability
            initial_scale_guess: (Delta_x_0, Delta_sigma_0, A_0) or None
            
        Returns:
            UnifiedSolverResult with mass, scales, and shape
        """
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Baryon")
            print("="*70)
            print(f"Quark windings: {quark_windings}")
            print(f"Color phases: {color_phases}")
        
        # =====================================================================
        # STAGE 1: Solve for normalized shape (dimensionless)
        # =====================================================================
        if self.verbose:
            print("\n" + "-"*70)
            print("STAGE 1: Solving for dimensionless shape")
            print("-"*70)
        
        shape_result = self.shape_solver.solve_baryon_shape(
            quark_windings=quark_windings,
            color_phases=color_phases,
            max_iter=max_scf_iter,
            tol=scf_tol,
            mixing=scf_mixing
        )
        
        if not shape_result.converged:
            warnings.warn("Shape solver did not converge")
        
        # Build 4D structure with spatial coupling
        structure_4d = self.spatial_coupling.build_4d_structure(
            subspace_shape=shape_result.composite_shape,
            n_target=n_target,
            l_target=0,  # Baryons are spatial ground state (s-wave)
            m_target=0
        )
        
        # =====================================================================
        # STAGE 2: Minimize energy over scale parameters
        # =====================================================================
        if self.verbose:
            print("\n" + "-"*70)
            print("STAGE 2: Minimizing energy over scale parameters")
            print("-"*70)
        
        energy_result = self.energy_minimizer.minimize_baryon_energy(
            shape_structure=structure_4d,
            initial_guess=initial_scale_guess
        )
        
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Complete")
            print("="*70)
            print(f"Predicted mass: {energy_result.mass:.6f} GeV")
            print(f"Amplitude A: {energy_result.A:.6f}")
            print(f"Spatial scale Delta_x: {energy_result.Delta_x:.6f} fm")
            print(f"Subspace width Delta_sigma: {energy_result.Delta_sigma:.6f}")
        
        # Return unified result
        return UnifiedSolverResult(
            mass=energy_result.mass,
            A=energy_result.A,
            Delta_x=energy_result.Delta_x,
            Delta_sigma=energy_result.Delta_sigma,
            shape_structure=structure_4d,
            energy_components={
                'E_total': energy_result.E_total,
                'E_sigma': energy_result.E_sigma,
                'E_kinetic_sigma': energy_result.E_kinetic_sigma,
                'E_potential_sigma': energy_result.E_potential_sigma,
                'E_nonlinear_sigma': energy_result.E_nonlinear_sigma,
                'E_spatial': energy_result.E_spatial,
                'E_coupling': energy_result.E_coupling,
                'E_curvature': energy_result.E_curvature,
                'E_em': energy_result.E_em
            },
            E_total=energy_result.E_total,
            shape_converged=shape_result.converged,
            shape_iterations=shape_result.iterations,
            energy_converged=energy_result.converged,
            energy_iterations=energy_result.iterations,
            particle_type='baryon',
            quantum_numbers={
                'quark_windings': quark_windings,
                'color_phases': color_phases,
                'n_target': n_target
            }
        )
    
    def solve_lepton(
        self,
        winding_k: int,
        generation_n: int,
        max_iter: int = 200,
        tol: float = 1e-6,
        initial_scale_guess: Optional[Tuple[float, float, float]] = None
    ) -> UnifiedSolverResult:
        """
        Solve for lepton using two-stage approach.
        
        Args:
            winding_k: Winding number (determines charge)
            generation_n: Generation number (1=e, 2=mu, 3=tau)
            max_iter: Maximum iterations for shape
            tol: Convergence tolerance for shape
            initial_scale_guess: (Delta_x_0, Delta_sigma_0, A_0) or None
            
        Returns:
            UnifiedSolverResult with mass, scales, and shape
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"UNIFIED SOLVER: Lepton (n={generation_n}, k={winding_k})")
            print("="*70)
        
        # =====================================================================
        # STAGE 1: Solve for normalized shape (dimensionless)
        # =====================================================================
        if self.verbose:
            print("\n" + "-"*70)
            print("STAGE 1: Solving for dimensionless shape")
            print("-"*70)
        
        shape_result = self.shape_solver.solve_lepton_shape(
            winding_k=winding_k,
            generation_n=generation_n,
            max_iter=max_iter,
            tol=tol
        )
        
        # Build 4D structure with spatial coupling
        structure_4d = self.spatial_coupling.build_4d_structure(
            subspace_shape=shape_result.composite_shape,
            n_target=generation_n,  # Leptons: generation = spatial quantum number
            l_target=0,
            m_target=0
        )
        
        # =====================================================================
        # STAGE 2: Minimize energy over scale parameters
        # =====================================================================
        if self.verbose:
            print("\n" + "-"*70)
            print("STAGE 2: Minimizing energy over scale parameters")
            print("-"*70)
        
        energy_result = self.energy_minimizer.minimize_lepton_energy(
            shape_structure=structure_4d,
            generation_n=generation_n,
            initial_guess=initial_scale_guess
        )
        
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Complete")
            print("="*70)
            print(f"Predicted mass: {energy_result.mass:.6f} GeV")
            print(f"Amplitude A: {energy_result.A:.6f}")
        
        # Return unified result
        return UnifiedSolverResult(
            mass=energy_result.mass,
            A=energy_result.A,
            Delta_x=energy_result.Delta_x,
            Delta_sigma=energy_result.Delta_sigma,
            shape_structure=structure_4d,
            energy_components={
                'E_total': energy_result.E_total,
                'E_sigma': energy_result.E_sigma,
                'E_kinetic_sigma': energy_result.E_kinetic_sigma,
                'E_potential_sigma': energy_result.E_potential_sigma,
                'E_nonlinear_sigma': energy_result.E_nonlinear_sigma,
                'E_spatial': energy_result.E_spatial,
                'E_coupling': energy_result.E_coupling,
                'E_curvature': energy_result.E_curvature,
                'E_em': energy_result.E_em
            },
            E_total=energy_result.E_total,
            shape_converged=shape_result.converged,
            shape_iterations=shape_result.iterations,
            energy_converged=energy_result.converged,
            energy_iterations=energy_result.iterations,
            particle_type='lepton',
            quantum_numbers={
                'winding_k': winding_k,
                'generation_n': generation_n
            }
        )
    
    def solve_meson(
        self,
        quark_winding: int,
        antiquark_winding: int,
        quark_phase: float = 0.0,
        antiquark_phase: float = np.pi,
        n_target: int = 1,
        max_scf_iter: int = 500,
        scf_tol: float = 1e-4,
        scf_mixing: float = 0.1,
        initial_scale_guess: Optional[Tuple[float, float, float]] = None
    ) -> UnifiedSolverResult:
        """
        Solve for meson using two-stage approach.
        
        Args:
            quark_winding: Quark winding number
            antiquark_winding: Antiquark winding number
            quark_phase: Quark color phase
            antiquark_phase: Antiquark color phase
            n_target: Target spatial quantum number
            max_scf_iter: Maximum SCF iterations for shape
            scf_tol: Convergence tolerance for shape
            scf_mixing: Mixing parameter for SCF
            initial_scale_guess: (Delta_x_0, Delta_sigma_0, A_0) or None
            
        Returns:
            UnifiedSolverResult with mass, scales, and shape
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"UNIFIED SOLVER: Meson (k_q={quark_winding}, k_qbar={antiquark_winding})")
            print("="*70)
        
        # =====================================================================
        # STAGE 1: Solve for normalized shape (dimensionless)
        # =====================================================================
        if self.verbose:
            print("\n" + "-"*70)
            print("STAGE 1: Solving for dimensionless shape")
            print("-"*70)
        
        shape_result = self.shape_solver.solve_meson_shape(
            quark_winding=quark_winding,
            antiquark_winding=antiquark_winding,
            quark_phase=quark_phase,
            antiquark_phase=antiquark_phase,
            max_iter=max_scf_iter,
            tol=scf_tol,
            mixing=scf_mixing
        )
        
        # Build 4D structure
        structure_4d = self.spatial_coupling.build_4d_structure(
            subspace_shape=shape_result.composite_shape,
            n_target=n_target,
            l_target=0,
            m_target=0
        )
        
        # =====================================================================
        # STAGE 2: Minimize energy over scale parameters
        # =====================================================================
        if self.verbose:
            print("\n" + "-"*70)
            print("STAGE 2: Minimizing energy over scale parameters")
            print("-"*70)
        
        energy_result = self.energy_minimizer.minimize_meson_energy(
            shape_structure=structure_4d,
            initial_guess=initial_scale_guess
        )
        
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Complete")
            print("="*70)
            print(f"Predicted mass: {energy_result.mass:.6f} GeV")
        
        # Return unified result
        return UnifiedSolverResult(
            mass=energy_result.mass,
            A=energy_result.A,
            Delta_x=energy_result.Delta_x,
            Delta_sigma=energy_result.Delta_sigma,
            shape_structure=structure_4d,
            energy_components={
                'E_total': energy_result.E_total,
                'E_sigma': energy_result.E_sigma,
                'E_kinetic_sigma': energy_result.E_kinetic_sigma,
                'E_potential_sigma': energy_result.E_potential_sigma,
                'E_nonlinear_sigma': energy_result.E_nonlinear_sigma,
                'E_spatial': energy_result.E_spatial,
                'E_coupling': energy_result.E_coupling,
                'E_curvature': energy_result.E_curvature,
                'E_em': energy_result.E_em
            },
            E_total=energy_result.E_total,
            shape_converged=shape_result.converged,
            shape_iterations=shape_result.iterations,
            energy_converged=energy_result.converged,
            energy_iterations=energy_result.iterations,
            particle_type='meson',
            quantum_numbers={
                'quark_winding': quark_winding,
                'antiquark_winding': antiquark_winding,
                'n_target': n_target
            }
        )
    
    def calibrate_beta_from_electron(
        self,
        electron_mass_exp: float = 0.000510999,  # GeV
        max_iter: int = 200,
        verbose: bool = True
    ) -> float:
        """
        Calibrate beta by solving for electron and matching experimental mass.
        
        This uses the beta-independent energy formulation to find the optimal
        amplitude A_electron, then calibrates beta = m_electron_exp / A_electron^2.
        
        Process:
            1. Solve electron shape (Stage 1, dimensionless)
            2. Minimize scaled energy to find optimal A_electron (beta-independent)
            3. Calculate beta = m_electron_exp / A_electron^2
            4. Update energy minimizer with calibrated beta
        
        Args:
            electron_mass_exp: Experimental electron mass (GeV)
            max_iter: Maximum iterations for shape solver
            verbose: Print calibration progress
            
        Returns:
            beta: Calibrated mass conversion factor (GeV)
        """
        if verbose:
            print("="*70)
            print("CALIBRATING BETA FROM ELECTRON MASS")
            print("="*70)
            print(f"Target: m_e = {electron_mass_exp*1000:.6f} MeV")
        
        # Stage 1: Solve electron shape (dimensionless)
        if verbose:
            print("\nStage 1: Solving electron shape...")
        
        shape_result = self.shape_solver.solve_lepton_shape(
            generation_n=1,
            winding_k=1,
            max_iter=max_iter,
            tol=1e-6
        )
        
        if verbose:
            print(f"  Shape converged: {shape_result.converged} ({shape_result.iterations} iters)")
        
        # Build 4D structure
        structure_4d = self.spatial_coupling.build_4d_structure(
            subspace_shape=shape_result.composite_shape,
            n_target=1,
            l_target=0,
            m_target=0
        )
        
        # Stage 2: Minimize energy (beta-independent)
        if verbose:
            print("\nStage 2: Minimizing beta-independent energy...")
        
        # Temporarily ensure energy minimizer uses scaled energy
        original_beta = self.energy_minimizer.beta
        original_use_scaled = self.energy_minimizer.use_scaled_energy
        self.energy_minimizer.beta = None  # Force beta-independent mode
        self.energy_minimizer.use_scaled_energy = True
        
        optimization_result = self.energy_minimizer.minimize_lepton_energy(
            shape_structure=structure_4d,
            generation_n=1
        )
        
        A_electron = optimization_result.A
        
        if verbose:
            print(f"  Optimal amplitude: A_electron = {A_electron:.6f}")
            print(f"  Optimal scales: Delta_x = {optimization_result.Delta_x:.3f} fm, "
                  f"Delta_sigma = {optimization_result.Delta_sigma:.3f}")
        
        # Calibrate beta
        beta_calibrated = electron_mass_exp / (A_electron**2)
        
        if verbose:
            print(f"\nCalibration:")
            print(f"  beta = m_e / A_e^2 = {electron_mass_exp:.6f} / {A_electron**2:.6f}")
            print(f"  beta = {beta_calibrated:.6f} GeV")
            print(f"\nVerification:")
            print(f"  m = beta x A^2 = {beta_calibrated * A_electron**2 * 1000:.6f} MeV")
            print(f"  Target m_e  = {electron_mass_exp * 1000:.6f} MeV")
            print(f"  Exact match by construction")
        
        # Update energy minimizer with calibrated beta
        self.energy_minimizer.beta = beta_calibrated
        self.energy_minimizer.use_scaled_energy = original_use_scaled
        self.beta = beta_calibrated
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BETA CALIBRATION COMPLETE")
            print(f"{'='*70}")
            print(f"All subsequent particle masses will use: m = beta x A^2")
            print(f"Mass ratios will emerge from amplitude ratios")
        
        return beta_calibrated

