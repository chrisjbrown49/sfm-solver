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
    V1 as V1_DEFAULT
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
    
    # Particle info (must come before fields with defaults)
    particle_type: str
    quantum_numbers: Dict
    
    # Outer loop convergence info (fields with defaults must come last)
    outer_iterations: int = 0
    outer_converged: bool = True
    scale_history: Optional[Dict[str, list]] = None


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
        solver = UnifiedSFMSolver(g_internal=1e6, verbose=True)
        
        # Solve for a particle
        result = solver.solve_baryon(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3)
        )
        # Test script reports results:
        print(f"Amplitude A: {result.A}")
        print(f"Mass: {result.mass} GeV")
    """
    
    def __init__(
        self,
        g_internal: Optional[float] = None,
        g1: Optional[float] = None,
        g2: Optional[float] = None,
        V0: Optional[float] = None,
        V1: Optional[float] = None,
        alpha: Optional[float] = None,
        n_max: int = 5,
        l_max: int = 2,
        N_sigma: int = 64,
        verbose: bool = False
    ):
        """
        Initialize unified solver.
        
        The solver finds optimal field configurations through pure energy minimization
        in a two-stage architecture. By default, loads parameters from constants.json.
        All parameters can be overridden by passing explicit values.
        
        The solver returns dimensionless amplitudes A. To convert to physical masses,
        use the helper function from calculate_beta module.
        
        Args:
            g_internal: Gravitational self-confinement strength (FUNDAMENTAL).
                        Controls how field amplitude creates spatial confinement.
                        If None, loads from constants.json (default: G_INTERNAL)
            g1: Nonlinear self-interaction coupling in subspace (dimensionless).
                If None, loads from constants.json (default: G1)
            g2: Electromagnetic circulation coupling (dimensionless).
                If None, loads from constants.json (default: G2)
            V0: Three-well primary depth (GeV, energy reference).
                If None, loads from constants.json (default: V0)
            V1: Three-well secondary depth (GeV).
                If None, loads from constants.json (default: V1)
            alpha: Spatial-subspace coupling strength (GeV).
                   If None, loads from constants.json (default: ALPHA)
            n_max: Maximum spatial quantum number for shape expansion
            l_max: Maximum angular momentum for shape expansion
            N_sigma: Grid points in subspace dimension
            verbose: Print diagnostic information
        """
        self.g_internal = g_internal if g_internal is not None else G_INTERNAL_DEFAULT
        self.g1 = g1 if g1 is not None else G1_DEFAULT
        self.g2 = g2 if g2 is not None else G2_DEFAULT
        self.V0 = V0 if V0 is not None else V0_DEFAULT
        self.V1 = V1 if V1 is not None else V1_DEFAULT
        self.alpha = alpha if alpha is not None else ALPHA_DEFAULT
        self.n_max = n_max
        self.l_max = l_max
        self.N_sigma = N_sigma
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
            verbose=verbose
        )
        
        if self.verbose:
            print("=== UnifiedSFMSolver Initialized ===")
            print(f"  Parameters loaded from constants.json")
            print(f"  g_internal = {self.g_internal:.6f} (Gravitational self-confinement)")
            print(f"  g1 = {self.g1:.3f} (Nonlinear subspace coupling)")
            print(f"  g2 = {self.g2:.6f} (EM circulation coupling)")
            print(f"  alpha = {self.alpha:.6f} GeV (Spatial-subspace coupling)")
            print(f"  V0 = {self.V0:.3f} GeV, V1 = {self.V1:.3f} GeV (Three-well depths)")
            print(f"  Two-stage architecture:")
            print(f"  Stage 1: Shape solver (dimensionless field configuration)")
            print(f"  Stage 2: Energy minimizer (scale-independent optimization)")
    
    def solve_baryon(
        self,
        quark_windings: Tuple[int, int, int],
        color_phases: Tuple[float, float, float] = (0, 2*np.pi/3, 4*np.pi/3),
        n_target: int = 1,
        max_scf_iter: int = 500,
        scf_tol: float = 1e-4,
        scf_mixing: float = 0.1,
        max_iter_outer: int = 30,
        tol_outer: float = 1e-4,
        initial_scale_guess: Optional[Tuple[float, float, float]] = None
    ) -> UnifiedSolverResult:
        """
        Solve for baryon using two-stage approach with outer iteration.
        
        Args:
            quark_windings: (k1, k2, k3) winding numbers
            color_phases: (phi1, phi2, phi3) for color neutrality
            n_target: Target spatial quantum number (usually 1 for baryons)
            max_scf_iter: Maximum SCF iterations for shape
            scf_tol: Convergence tolerance for shape
            scf_mixing: Mixing parameter for SCF stability
            max_iter_outer: Maximum outer loop iterations
            tol_outer: Convergence tolerance for outer loop (scale changes)
            initial_scale_guess: (Delta_x_0, Delta_sigma_0, A_0) or None
            
        Returns:
            UnifiedSolverResult with mass, scales, shape, and outer convergence info
        """
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Baryon")
            print("="*70)
            print(f"Quark windings: {quark_windings}")
            print(f"Color phases: {color_phases}")
        
        # Initialize scale tracking
        scale_history = {
            'A': [],
            'Delta_x': [],
            'Delta_sigma': [],
            'iteration': []
        }
        
        # Initialize current scales
        if initial_scale_guess is not None:
            Delta_x_current, Delta_sigma_current, A_current = initial_scale_guess
        else:
            # Baryons typically have larger amplitudes than leptons
            A_current = 50.0  # Reasonable starting point for baryons
            Delta_x_current = self.energy_minimizer._compute_optimal_delta_x(A_current)
            Delta_sigma_current = self.energy_minimizer._compute_optimal_delta_sigma(A_current)
        
        # Improved adaptive mixing with momentum for stability
        mixing = 0.05  # Start with heavy damping
        mixing_min = 0.01
        mixing_max = 0.3
        
        # Momentum terms
        momentum_A = 0.0
        momentum_Dx = 0.0
        momentum_Ds = 0.0
        momentum_weight = 0.15
        
        # History for oscillation detection
        history_window = 5
        dA_history = []
        dDx_history = []
        dDs_history = []
        
        converged_outer = False
        shape_result = None
        energy_result = None
        structure_4d = None
        
        for iter_outer in range(max_iter_outer):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"OUTER ITERATION {iter_outer + 1}/{max_iter_outer}")
                print(f"{'='*70}")
                print(f"Current scales: A={A_current:.6f}, Dx={Delta_x_current:.6f} fm, Ds={Delta_sigma_current:.6f}")
            
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
            
            # Build 4D structure with scale-aware coupling
            structure_4d = self.spatial_coupling.build_4d_structure(
                subspace_shape=shape_result.composite_shape,
                n_target=n_target,
                l_target=0,
                m_target=0,
                Delta_x=Delta_x_current  # Scale-aware coupling
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
                initial_guess=(Delta_x_current, Delta_sigma_current, A_current)
            )
            
            # Extract new scales
            A_new = energy_result.A
            Delta_x_new = energy_result.Delta_x
            Delta_sigma_new = energy_result.Delta_sigma
            
            # Store in history
            scale_history['A'].append(A_new)
            scale_history['Delta_x'].append(Delta_x_new)
            scale_history['Delta_sigma'].append(Delta_sigma_new)
            scale_history['iteration'].append(iter_outer)
            
            # =====================================================================
            # CHECK CONVERGENCE
            # =====================================================================
            if iter_outer > 0:
                delta_A = abs(A_new - A_current)
                delta_Dx = abs(Delta_x_new - Delta_x_current)
                delta_Ds = abs(Delta_sigma_new - Delta_sigma_current)
                
                rel_delta_A = delta_A / max(A_current, 0.01)
                rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
                rel_delta_Ds = delta_Ds / max(Delta_sigma_current, 0.01)
                
                if self.verbose:
                    print(f"\n  Outer convergence check:")
                    print(f"    dA/A={rel_delta_A:.2e}, dDx/Dx={rel_delta_Dx:.2e}, dDs/Ds={rel_delta_Ds:.2e}")
                    print(f"    mixing={mixing:.4f}, momentum_weight={momentum_weight:.3f}")
                
                if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer and rel_delta_Ds < tol_outer:
                    converged_outer = True
                    if self.verbose:
                        print(f"\n  OUTER LOOP CONVERGED after {iter_outer + 1} iterations")
                    break
                
                # Compute current changes
                dA_current = A_new - A_current
                dDx_current = Delta_x_new - Delta_x_current
                dDs_current = Delta_sigma_new - Delta_sigma_current
                
                # Add to history
                dA_history.append(dA_current)
                dDx_history.append(dDx_current)
                dDs_history.append(dDs_current)
                
                # Keep only last history_window iterations
                if len(dA_history) > history_window:
                    dA_history.pop(0)
                    dDx_history.pop(0)
                    dDs_history.pop(0)
                
                # Improved oscillation detection
                oscillating = False
                oscillation_count = 0
                
                if len(dA_history) >= 2:
                    for i in range(len(dA_history) - 1):
                        if (dA_history[i] * dA_history[i+1]) < 0:
                            oscillation_count += 1
                        if (dDx_history[i] * dDx_history[i+1]) < 0:
                            oscillation_count += 1
                        if (dDs_history[i] * dDs_history[i+1]) < 0:
                            oscillation_count += 1
                    
                    if oscillation_count >= 2:
                        oscillating = True
                
                # Adaptive mixing based on oscillation behavior
                if oscillating:
                    mixing = max(mixing_min, mixing * 0.5)
                    if self.verbose:
                        print(f"    OSCILLATION DETECTED ({oscillation_count} sign changes) - Reducing mixing to {mixing:.4f}")
                elif iter_outer > 10:
                    mixing = min(mixing_max, mixing * 1.05)
                    if self.verbose and mixing < mixing_max:
                        print(f"    Stable - Increasing mixing to {mixing:.4f}")
            
            # =====================================================================
            # UPDATE SCALES WITH MIXING AND MOMENTUM
            # =====================================================================
            if iter_outer == 0:
                A_current = A_new
                Delta_x_current = Delta_x_new
                Delta_sigma_current = Delta_sigma_new
            else:
                # Momentum-damped changes
                dA_damped = (1 - momentum_weight) * dA_current + momentum_weight * momentum_A
                dDx_damped = (1 - momentum_weight) * dDx_current + momentum_weight * momentum_Dx
                dDs_damped = (1 - momentum_weight) * dDs_current + momentum_weight * momentum_Ds
                
                # Apply mixing with momentum
                A_current = A_current + mixing * dA_damped
                Delta_x_current = Delta_x_current + mixing * dDx_damped
                Delta_sigma_current = Delta_sigma_current + mixing * dDs_damped
                
                # Update momentum
                momentum_A = dA_damped
                momentum_Dx = dDx_damped
                momentum_Ds = dDs_damped
                
                if self.verbose:
                    print(f"    Applied update: dA={mixing*dA_damped:.4e}, dDx={mixing*dDx_damped:.4e}, dDs={mixing*dDs_damped:.4e}")
        
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Complete")
            print("="*70)
            print(f"Final scales: A={A_current:.6f}, Dx={Delta_x_current:.6f} fm, Ds={Delta_sigma_current:.6f}")
            print(f"Outer iterations: {iter_outer + 1}, converged: {converged_outer}")
        
        # Return unified result with outer loop info
        return UnifiedSolverResult(
            mass=energy_result.mass,
            A=A_current,
            Delta_x=Delta_x_current,
            Delta_sigma=Delta_sigma_current,
            shape_structure=structure_4d,
            energy_components={
                'E_total': energy_result.E_total,
                'E_sigma': energy_result.E_sigma,
                'E_kinetic_sigma': energy_result.E_kinetic_sigma,
                'E_potential_sigma': energy_result.E_potential_sigma,
                'E_nonlinear_sigma': energy_result.E_nonlinear_sigma,
                'E_spatial': energy_result.E_spatial,
                'E_coupling': energy_result.E_coupling,
                'E_em': energy_result.E_em
            },
            E_total=energy_result.E_total,
            shape_converged=shape_result.converged,
            shape_iterations=shape_result.iterations,
            energy_converged=energy_result.converged,
            energy_iterations=energy_result.iterations,
            outer_iterations=iter_outer + 1,
            outer_converged=converged_outer,
            scale_history=scale_history,
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
        max_iter_outer: int = 30,
        tol_outer: float = 1e-4,
        initial_scale_guess: Optional[Tuple[float, float, float]] = None
    ) -> UnifiedSolverResult:
        """
        Solve for lepton using two-stage approach with outer iteration.
        
        The outer loop iterates between:
          1. Stage 1: Solve shape at current scales
          2. Stage 2: Optimize scales for that shape
          3. Check convergence and update scales with mixing
        
        Args:
            winding_k: Winding number (determines charge)
            generation_n: Generation number (1=e, 2=mu, 3=tau)
            max_iter: Maximum iterations for shape (inner loop)
            tol: Convergence tolerance for shape (inner loop)
            max_iter_outer: Maximum outer loop iterations
            tol_outer: Convergence tolerance for outer loop (scale changes)
            initial_scale_guess: (Delta_x_0, Delta_sigma_0, A_0) or None
            
        Returns:
            UnifiedSolverResult with mass, scales, shape, and outer convergence info
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"UNIFIED SOLVER: Lepton (n={generation_n}, k={winding_k})")
            print("="*70)
        
        # Initialize scale tracking
        scale_history = {
            'A': [],
            'Delta_x': [],
            'Delta_sigma': [],
            'iteration': []
        }
        
        # Initialize current scales (first iteration uses defaults)
        if initial_scale_guess is not None:
            Delta_x_current, Delta_sigma_current, A_current = initial_scale_guess
        else:
            # Use energy minimizer's generation-dependent defaults (legacy values)
            if generation_n == 1:
                A_current = 0.9
            elif generation_n == 2:
                A_current = 12.0
            elif generation_n == 3:
                A_current = 50.0
            else:
                A_current = max(1.0, 5.0 * generation_n)
            
            Delta_x_current = self.energy_minimizer._compute_optimal_delta_x(A_current)
            Delta_sigma_current = self.energy_minimizer._compute_optimal_delta_sigma(A_current)
        
        # Improved adaptive mixing with momentum for stability
        mixing = 0.05  # Start with heavy damping (was 0.3)
        mixing_min = 0.01  # Minimum mixing (very heavy damping)
        mixing_max = 0.3  # Maximum mixing (was 0.5, now more conservative)
        
        # Momentum terms to smooth updates
        momentum_A = 0.0
        momentum_Dx = 0.0
        momentum_Ds = 0.0
        momentum_weight = 0.15  # Typical value for momentum damping
        
        # History for better oscillation detection (track last 5 iterations)
        history_window = 5
        dA_history = []
        dDx_history = []
        dDs_history = []
        
        converged_outer = False
        shape_result = None
        energy_result = None
        structure_4d = None
        
        for iter_outer in range(max_iter_outer):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"OUTER ITERATION {iter_outer + 1}/{max_iter_outer}")
                print(f"{'='*70}")
                print(f"Current scales: A={A_current:.6f}, Dx={Delta_x_current:.6f} fm, Ds={Delta_sigma_current:.6f}")
            
            # =====================================================================
            # STAGE 1: Solve shape (dimensionless) at current scales
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
            
            # Build 4D structure with scale-aware coupling
            structure_4d = self.spatial_coupling.build_4d_structure(
                subspace_shape=shape_result.composite_shape,
                n_target=generation_n,
                l_target=0,
                m_target=0,
                Delta_x=Delta_x_current  # Pass current scale for scale-aware coupling
            )
            
            # =====================================================================
            # STAGE 2: Find optimal scales for this shape
            # =====================================================================
            if self.verbose:
                print("\n" + "-"*70)
                print("STAGE 2: Minimizing energy over scale parameters")
                print("-"*70)
            
            energy_result = self.energy_minimizer.minimize_lepton_energy(
                shape_structure=structure_4d,
                generation_n=generation_n,
                initial_guess=(Delta_x_current, Delta_sigma_current, A_current)
            )
            
            # Extract new scales
            A_new = energy_result.A
            Delta_x_new = energy_result.Delta_x
            Delta_sigma_new = energy_result.Delta_sigma
            
            # Store in history
            scale_history['A'].append(A_new)
            scale_history['Delta_x'].append(Delta_x_new)
            scale_history['Delta_sigma'].append(Delta_sigma_new)
            scale_history['iteration'].append(iter_outer)
            
            # =====================================================================
            # CHECK CONVERGENCE
            # =====================================================================
            if iter_outer > 0:
                delta_A = abs(A_new - A_current)
                delta_Dx = abs(Delta_x_new - Delta_x_current)
                delta_Ds = abs(Delta_sigma_new - Delta_sigma_current)
                
                rel_delta_A = delta_A / max(A_current, 0.01)
                rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
                rel_delta_Ds = delta_Ds / max(Delta_sigma_current, 0.01)
                
                if self.verbose:
                    print(f"\n  Outer convergence check:")
                    print(f"    dA/A={rel_delta_A:.2e}, dDx/Dx={rel_delta_Dx:.2e}, dDs/Ds={rel_delta_Ds:.2e}")
                    print(f"    mixing={mixing:.4f}, momentum_weight={momentum_weight:.3f}")
                
                if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer and rel_delta_Ds < tol_outer:
                    converged_outer = True
                    if self.verbose:
                        print(f"\n  OUTER LOOP CONVERGED after {iter_outer + 1} iterations")
                    break
                
                # Compute current changes
                dA_current = A_new - A_current
                dDx_current = Delta_x_new - Delta_x_current
                dDs_current = Delta_sigma_new - Delta_sigma_current
                
                # Add to history
                dA_history.append(dA_current)
                dDx_history.append(dDx_current)
                dDs_history.append(dDs_current)
                
                # Keep only last history_window iterations
                if len(dA_history) > history_window:
                    dA_history.pop(0)
                    dDx_history.pop(0)
                    dDs_history.pop(0)
                
                # Improved oscillation detection: check for sign changes in history
                oscillating = False
                oscillation_count = 0
                
                if len(dA_history) >= 2:
                    # Count sign changes in recent history
                    for i in range(len(dA_history) - 1):
                        if (dA_history[i] * dA_history[i+1]) < 0:
                            oscillation_count += 1
                        if (dDx_history[i] * dDx_history[i+1]) < 0:
                            oscillation_count += 1
                        if (dDs_history[i] * dDs_history[i+1]) < 0:
                            oscillation_count += 1
                    
                    # If we see multiple sign changes, it's oscillating
                    if oscillation_count >= 2:
                        oscillating = True
                
                # Adaptive mixing based on oscillation behavior
                if oscillating:
                    # Reduce mixing aggressively on oscillation
                    mixing = max(mixing_min, mixing * 0.5)
                    if self.verbose:
                        print(f"    OSCILLATION DETECTED ({oscillation_count} sign changes) - Reducing mixing to {mixing:.4f}")
                elif iter_outer > 10:
                    # Very slowly increase mixing after stable iterations
                    mixing = min(mixing_max, mixing * 1.05)
                    if self.verbose and mixing < mixing_max:
                        print(f"    Stable - Increasing mixing to {mixing:.4f}")
            
            # =====================================================================
            # UPDATE SCALES WITH MIXING AND MOMENTUM
            # =====================================================================
            if iter_outer == 0:
                # First iteration - use values directly
                A_current = A_new
                Delta_x_current = Delta_x_new
                Delta_sigma_current = Delta_sigma_new
            else:
                # Compute momentum-damped changes
                # Momentum smooths the direction of updates to prevent wild swings
                dA_damped = (1 - momentum_weight) * dA_current + momentum_weight * momentum_A
                dDx_damped = (1 - momentum_weight) * dDx_current + momentum_weight * momentum_Dx
                dDs_damped = (1 - momentum_weight) * dDs_current + momentum_weight * momentum_Ds
                
                # Apply adaptive mixing with momentum
                A_current = A_current + mixing * dA_damped
                Delta_x_current = Delta_x_current + mixing * dDx_damped
                Delta_sigma_current = Delta_sigma_current + mixing * dDs_damped
                
                # Update momentum for next iteration
                momentum_A = dA_damped
                momentum_Dx = dDx_damped
                momentum_Ds = dDs_damped
                
                if self.verbose:
                    print(f"    Applied update: dA={mixing*dA_damped:.4e}, dDx={mixing*dDx_damped:.4e}, dDs={mixing*dDs_damped:.4e}")
        
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Complete")
            print("="*70)
            print(f"Final scales: A={A_current:.6f}, Dx={Delta_x_current:.6f} fm, Ds={Delta_sigma_current:.6f}")
            print(f"Outer iterations: {iter_outer + 1}, converged: {converged_outer}")
        
        # Return unified result with outer loop info
        return UnifiedSolverResult(
            mass=energy_result.mass,
            A=A_current,
            Delta_x=Delta_x_current,
            Delta_sigma=Delta_sigma_current,
            shape_structure=structure_4d,
            energy_components={
                'E_total': energy_result.E_total,
                'E_sigma': energy_result.E_sigma,
                'E_kinetic_sigma': energy_result.E_kinetic_sigma,
                'E_potential_sigma': energy_result.E_potential_sigma,
                'E_nonlinear_sigma': energy_result.E_nonlinear_sigma,
                'E_spatial': energy_result.E_spatial,
                'E_coupling': energy_result.E_coupling,
                'E_em': energy_result.E_em
            },
            E_total=energy_result.E_total,
            shape_converged=shape_result.converged,
            shape_iterations=shape_result.iterations,
            energy_converged=energy_result.converged,
            energy_iterations=energy_result.iterations,
            outer_iterations=iter_outer + 1,
            outer_converged=converged_outer,
            scale_history=scale_history,
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
        max_iter_outer: int = 30,
        tol_outer: float = 1e-4,
        initial_scale_guess: Optional[Tuple[float, float, float]] = None
    ) -> UnifiedSolverResult:
        """
        Solve for meson using two-stage approach with outer iteration.
        
        Args:
            quark_winding: Quark winding number
            antiquark_winding: Antiquark winding number
            quark_phase: Quark color phase
            antiquark_phase: Antiquark color phase
            n_target: Target spatial quantum number
            max_scf_iter: Maximum SCF iterations for shape
            scf_tol: Convergence tolerance for shape
            scf_mixing: Mixing parameter for SCF
            max_iter_outer: Maximum outer loop iterations
            tol_outer: Convergence tolerance for outer loop (scale changes)
            initial_scale_guess: (Delta_x_0, Delta_sigma_0, A_0) or None
            
        Returns:
            UnifiedSolverResult with mass, scales, shape, and outer convergence info
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"UNIFIED SOLVER: Meson (k_q={quark_winding}, k_qbar={antiquark_winding})")
            print("="*70)
        
        # Initialize scale tracking
        scale_history = {
            'A': [],
            'Delta_x': [],
            'Delta_sigma': [],
            'iteration': []
        }
        
        # Initialize current scales
        if initial_scale_guess is not None:
            Delta_x_current, Delta_sigma_current, A_current = initial_scale_guess
        else:
            # Mesons typically have intermediate amplitudes
            A_current = 20.0  # Reasonable starting point for mesons
            Delta_x_current = self.energy_minimizer._compute_optimal_delta_x(A_current)
            Delta_sigma_current = self.energy_minimizer._compute_optimal_delta_sigma(A_current)
        
        # Improved adaptive mixing with momentum for stability
        mixing = 0.05  # Start with heavy damping
        mixing_min = 0.01
        mixing_max = 0.3
        
        # Momentum terms
        momentum_A = 0.0
        momentum_Dx = 0.0
        momentum_Ds = 0.0
        momentum_weight = 0.15
        
        # History for oscillation detection
        history_window = 5
        dA_history = []
        dDx_history = []
        dDs_history = []
        
        converged_outer = False
        shape_result = None
        energy_result = None
        structure_4d = None
        
        for iter_outer in range(max_iter_outer):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"OUTER ITERATION {iter_outer + 1}/{max_iter_outer}")
                print(f"{'='*70}")
                print(f"Current scales: A={A_current:.6f}, Dx={Delta_x_current:.6f} fm, Ds={Delta_sigma_current:.6f}")
            
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
            
            # Build 4D structure with scale-aware coupling
            structure_4d = self.spatial_coupling.build_4d_structure(
                subspace_shape=shape_result.composite_shape,
                n_target=n_target,
                l_target=0,
                m_target=0,
                Delta_x=Delta_x_current  # Scale-aware coupling
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
                initial_guess=(Delta_x_current, Delta_sigma_current, A_current)
            )
            
            # Extract new scales
            A_new = energy_result.A
            Delta_x_new = energy_result.Delta_x
            Delta_sigma_new = energy_result.Delta_sigma
            
            # Store in history
            scale_history['A'].append(A_new)
            scale_history['Delta_x'].append(Delta_x_new)
            scale_history['Delta_sigma'].append(Delta_sigma_new)
            scale_history['iteration'].append(iter_outer)
            
            # =====================================================================
            # CHECK CONVERGENCE
            # =====================================================================
            if iter_outer > 0:
                delta_A = abs(A_new - A_current)
                delta_Dx = abs(Delta_x_new - Delta_x_current)
                delta_Ds = abs(Delta_sigma_new - Delta_sigma_current)
                
                rel_delta_A = delta_A / max(A_current, 0.01)
                rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
                rel_delta_Ds = delta_Ds / max(Delta_sigma_current, 0.01)
                
                if self.verbose:
                    print(f"\n  Outer convergence check:")
                    print(f"    dA/A={rel_delta_A:.2e}, dDx/Dx={rel_delta_Dx:.2e}, dDs/Ds={rel_delta_Ds:.2e}")
                    print(f"    mixing={mixing:.4f}, momentum_weight={momentum_weight:.3f}")
                
                if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer and rel_delta_Ds < tol_outer:
                    converged_outer = True
                    if self.verbose:
                        print(f"\n  OUTER LOOP CONVERGED after {iter_outer + 1} iterations")
                    break
                
                # Compute current changes
                dA_current = A_new - A_current
                dDx_current = Delta_x_new - Delta_x_current
                dDs_current = Delta_sigma_new - Delta_sigma_current
                
                # Add to history
                dA_history.append(dA_current)
                dDx_history.append(dDx_current)
                dDs_history.append(dDs_current)
                
                # Keep only last history_window iterations
                if len(dA_history) > history_window:
                    dA_history.pop(0)
                    dDx_history.pop(0)
                    dDs_history.pop(0)
                
                # Improved oscillation detection
                oscillating = False
                oscillation_count = 0
                
                if len(dA_history) >= 2:
                    for i in range(len(dA_history) - 1):
                        if (dA_history[i] * dA_history[i+1]) < 0:
                            oscillation_count += 1
                        if (dDx_history[i] * dDx_history[i+1]) < 0:
                            oscillation_count += 1
                        if (dDs_history[i] * dDs_history[i+1]) < 0:
                            oscillation_count += 1
                    
                    if oscillation_count >= 2:
                        oscillating = True
                
                # Adaptive mixing based on oscillation behavior
                if oscillating:
                    mixing = max(mixing_min, mixing * 0.5)
                    if self.verbose:
                        print(f"    OSCILLATION DETECTED ({oscillation_count} sign changes) - Reducing mixing to {mixing:.4f}")
                elif iter_outer > 10:
                    mixing = min(mixing_max, mixing * 1.05)
                    if self.verbose and mixing < mixing_max:
                        print(f"    Stable - Increasing mixing to {mixing:.4f}")
            
            # =====================================================================
            # UPDATE SCALES WITH MIXING AND MOMENTUM
            # =====================================================================
            if iter_outer == 0:
                A_current = A_new
                Delta_x_current = Delta_x_new
                Delta_sigma_current = Delta_sigma_new
            else:
                # Momentum-damped changes
                dA_damped = (1 - momentum_weight) * dA_current + momentum_weight * momentum_A
                dDx_damped = (1 - momentum_weight) * dDx_current + momentum_weight * momentum_Dx
                dDs_damped = (1 - momentum_weight) * dDs_current + momentum_weight * momentum_Ds
                
                # Apply mixing with momentum
                A_current = A_current + mixing * dA_damped
                Delta_x_current = Delta_x_current + mixing * dDx_damped
                Delta_sigma_current = Delta_sigma_current + mixing * dDs_damped
                
                # Update momentum
                momentum_A = dA_damped
                momentum_Dx = dDx_damped
                momentum_Ds = dDs_damped
                
                if self.verbose:
                    print(f"    Applied update: dA={mixing*dA_damped:.4e}, dDx={mixing*dDx_damped:.4e}, dDs={mixing*dDs_damped:.4e}")
        
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED SOLVER: Complete")
            print("="*70)
            print(f"Final scales: A={A_current:.6f}, Dx={Delta_x_current:.6f} fm, Ds={Delta_sigma_current:.6f}")
            print(f"Outer iterations: {iter_outer + 1}, converged: {converged_outer}")
        
        # Return unified result with outer loop info
        return UnifiedSolverResult(
            mass=energy_result.mass,
            A=A_current,
            Delta_x=Delta_x_current,
            Delta_sigma=Delta_sigma_current,
            shape_structure=structure_4d,
            energy_components={
                'E_total': energy_result.E_total,
                'E_sigma': energy_result.E_sigma,
                'E_kinetic_sigma': energy_result.E_kinetic_sigma,
                'E_potential_sigma': energy_result.E_potential_sigma,
                'E_nonlinear_sigma': energy_result.E_nonlinear_sigma,
                'E_spatial': energy_result.E_spatial,
                'E_coupling': energy_result.E_coupling,
                'E_em': energy_result.E_em
            },
            E_total=energy_result.E_total,
            shape_converged=shape_result.converged,
            shape_iterations=shape_result.iterations,
            energy_converged=energy_result.converged,
            energy_iterations=energy_result.iterations,
            outer_iterations=iter_outer + 1,
            outer_converged=converged_outer,
            scale_history=scale_history,
            particle_type='meson',
            quantum_numbers={
                'quark_winding': quark_winding,
                'antiquark_winding': antiquark_winding,
                'n_target': n_target
            }
        )
