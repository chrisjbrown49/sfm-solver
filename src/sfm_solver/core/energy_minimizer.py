"""
Universal Energy Minimizer for SFM.

This module implements Stage 2 of the two-stage solver architecture.
Takes normalized shape from Stage 1 and finds optimal scale parameters
(Delta_x, Delta_sigma, A) by minimizing total energy.

NEW IMPLEMENTATION (Post-Diagnostic):
=====================================
All energy components are computed directly from the full 5D wavefunction
Ψ(r,σ) = φ_n(r; Δx) × χ(σ; Δσ, A), faithfully implementing the Hamiltonian
operators from Math Formulation Part A, Section 2.

This eliminates factorization errors in the previous analytical approximation.
The diagnostic (diagnose_coupling_energy.py) revealed the old E_coupling
formula was incorrect by a factor of 20-100×!

Energy Components:
- E_spatial: Spatial kinetic energy from -ℏ²/(2m)∇²
- E_sigma: Subspace kinetic + potential + nonlinear
- E_coupling: Mixed derivative -α(∂²/∂x∂σ + ...) [CORRECTED]
- E_em: Circulation penalty g₂|J|²
- E_curv: Gravitational self-confinement G_5D³A⁴/Δx

KEY PRINCIPLE: Stage 2 (physical scales)
    - Input: Normalized shape structure from Stage 1
    - Output: Optimal (Delta_x, Delta_sigma, A)
    - Energy functional: E_total(Delta_x, Delta_sigma, A) computed from 5D field
    - First-principles: Amplitude A emerges from energy minimization alone

The shape is FIXED (from Stage 1), only the SCALE varies.

CRITICAL: Energy minimization is performed in a scale-independent manner,
ensuring that the optimal amplitude A is determined purely by field dynamics
and energy balance, independent of any external mass scale.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize, minimize_scalar
from scipy.special import genlaguerre
import warnings

from .field_energy_5d import Full5DFieldEnergy

# Unit conversion constant: ℏc in GeV·fm
# Used to convert spatial lengths from fm to natural units (GeV^-1)
HBAR_C_GEV_FM = 0.1973269804  # GeV·fm


@dataclass
class EnergyMinimizationResult:
    """Result from Stage 2 energy minimization."""
    
    # Optimized scale parameters
    Delta_x: float  # Spatial extent (fm)
    Delta_sigma: float  # Subspace width
    A: float  # Amplitude
    
    # Predicted mass (requires mass scale calibration)
    mass: Optional[float]  # GeV, None if mass scale not yet set
    
    # Energy breakdown
    E_total: float
    E_sigma: float
    E_kinetic_sigma: float
    E_potential_sigma: float
    E_nonlinear_sigma: float
    E_spatial: float
    E_coupling: float
    E_em: float
    E_curvature: float  # Gravitational self-confinement
    
    # Convergence
    converged: bool
    iterations: int
    optimization_message: str
    
    # Input info
    particle_type: str
    n_target: Optional[int] = None
    k_winding: Optional[int] = None


class UniversalEnergyMinimizer:
    """
    Universal energy minimizer for all SFM particles.
    
    Given a converged wavefunction structure from Stage 1,
    find the optimal (Delta_x, Delta_sigma, A) that minimizes
    the total energy.
    
    The SAME code works for leptons, mesons, and baryons!
    The wavefunction structure encodes particle-specific physics.
    This minimizer applies UNIVERSAL energy balance.
    """
    
    def __init__(
        self,
        G_5D: float,
        g1: float,
        g2: float,
        V0: float,
        V1: float,
        alpha: float,
        lambda_so: float,
        verbose: bool = False
    ):
        """
        Initialize universal energy minimizer.
        
        This minimizer finds optimal (Delta_x, Delta_sigma, A) by minimizing
        the total energy functional. The energy minimization is performed in a
        scale-independent manner, ensuring amplitudes emerge from first principles.
        
        Returns dimensionless amplitudes only. To convert to physical masses,
        use the helper function in calculate_beta module.
        
        Args:
            G_5D: FUNDAMENTAL 5D gravitational constant (in GeV^-2).
                  Controls gravitational self-confinement and spatial energy.
                  Related to mass scale via beta = G_5D * c.
            g1: Nonlinear self-interaction coupling in subspace (dimensionless)
            g2: Electromagnetic circulation coupling (dimensionless)
            V0: Three-well primary depth (GeV)
            V1: Three-well secondary depth (GeV)
            alpha: Spatial-subspace coupling strength (GeV)
            lambda_so: Spin-orbit coupling strength
            verbose: Print diagnostic information
        """
        self.G_5D = G_5D  # FUNDAMENTAL - 5D gravitational constant
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        self.V1 = V1
        self.alpha = alpha
        self.lambda_so = lambda_so
        self.verbose = verbose
        
        # Radial grid for spatial wavefunction integrals
        self._r_grid = np.linspace(0.01, 20.0, 500)
        
        # Load grid parameters for 5D field energy computation
        from .constants import N_R_GRID, N_SIGMA_GRID, R_MAX_FM
        
        # Create 5D field energy computer
        self.field_energy = Full5DFieldEnergy(
            G_5D=self.G_5D,
            g1=self.g1,
            g2=self.g2,
            alpha=self.alpha,
            lambda_so=self.lambda_so,
            V0=self.V0,
            V1=self.V1,
            N_r=N_R_GRID,
            N_sigma=N_SIGMA_GRID,
            r_max=R_MAX_FM
        )
        
        # Create spatial coupling builder for rebuilding structure
        from .spatial_coupling import SpatialCouplingBuilder
        self.spatial_coupling = SpatialCouplingBuilder(
            alpha_dimensionless=self.alpha / self.V0,
            n_max=5,
            l_max=2,
            verbose=False  # Don't clutter output during energy evaluations
        )
        
        if self.verbose:
            print("=== UniversalEnergyMinimizer Initialized ===")
            print(f"  G_5D: {G_5D:.6e} GeV^-2 (5D Gravitational constant)")
            print(f"  g1: {g1:.3f} (Nonlinear subspace coupling)")
            print(f"  g2: {g2:.6f} (EM circulation coupling)")
            print(f"  alpha: {alpha:.3f} GeV (Spatial-subspace coupling)")
            print(f"  V0: {V0:.6f} GeV, V1: {V1:.6f} GeV (Three-well depths)")
            print(f"  5D Field Grid: N_r={N_R_GRID}, N_sigma={N_SIGMA_GRID}, r_max={R_MAX_FM} fm")
            print(f"  Energy minimization: Scale-independent (first-principles)")
    
    # =========================================================================
    # Helper methods for initial guesses (used by unified_solver outer loop)
    # =========================================================================
    
    def _compute_optimal_delta_x(self, A: float) -> float:
        """
        Compute initial guess for Delta_x from gravitational self-confinement.
        
        This is a simplified heuristic for the outer loop initial guess.
        The actual optimal value will be found by the energy minimizer.
        
        Args:
            A: Current amplitude estimate
            
        Returns:
            Initial guess for Delta_x in fm
        """
        if self.G_5D <= 0 or A < 1e-10:
            return 10.0  # Default fallback in fm
        
        # Heuristic: Delta_x ~ 1/(G_5D * A^2)^(1/3)
        A_sixth = A ** 6
        delta_x = HBAR_C_GEV_FM / (self.G_5D * A_sixth) ** (1.0/3.0)
        
        # Apply reasonable bounds
        MIN_DELTA_X = 0.1   # fm
        MAX_DELTA_X = 100.0  # fm
        delta_x = max(MIN_DELTA_X, min(MAX_DELTA_X, delta_x))
        
        return delta_x
    
    def minimize_baryon_energy(
        self,
        chi_normalized: NDArray,
        Delta_sigma_fixed: float,
        initial_guess: Optional[Tuple[float, float]] = None,
        method: str = 'Nelder-Mead'
    ) -> EnergyMinimizationResult:
        """
        Find optimal (Delta_x, A) for baryon with FIXED Delta_sigma from Stage 1.
        
        Args:
            chi_normalized: Normalized subspace shape from Stage 1
            Delta_sigma_fixed: Fixed natural width from Stage 1 shape solver
            initial_guess: (Delta_x_0, A_0) or None for automatic
            method: Optimization method
            
        Returns:
            EnergyMinimizationResult with optimal parameters and mass
        """
        if self.verbose:
            print("\n=== Minimizing Baryon Energy ===")
            print(f"  Delta_sigma FIXED from Stage 1: {Delta_sigma_fixed:.4f}")
        
        # Auto-generate initial guess if not provided
        if initial_guess is None:
            A_0 = 60.0  # Typical baryon amplitude
            Delta_x_0 = 1.0  # fm (typical hadronic scale)
            initial_guess = (Delta_x_0, A_0)
        
        if self.verbose:
            print(f"  Initial guess: Delta_x={initial_guess[0]:.3f} fm, A={initial_guess[1]:.3f}")
        
        # Minimize
        result = self._minimize_energy(
            chi_normalized=chi_normalized,
            Delta_sigma_fixed=Delta_sigma_fixed,
            initial_guess=initial_guess,
            method=method,
            particle_type='baryon'
        )
        
        return result
    
    def minimize_lepton_energy(
        self,
        chi_normalized: NDArray,
        generation_n: int,
        Delta_sigma_fixed: float,
        initial_guess: Optional[Tuple[float, float]] = None,
        method: str = 'Nelder-Mead'
    ) -> EnergyMinimizationResult:
        """
        Find optimal (Delta_x, A) for lepton with FIXED Delta_sigma from Stage 1.
        
        Args:
            chi_normalized: Normalized subspace shape from Stage 1
            generation_n: Generation number (1, 2, 3)
            Delta_sigma_fixed: Fixed natural width from Stage 1 shape solver
            initial_guess: (Delta_x_0, A_0) or None for automatic
            method: Optimization method
            
        Returns:
            EnergyMinimizationResult with optimal parameters and mass
        """
        if self.verbose:
            print(f"\n=== Minimizing Lepton Energy (n={generation_n}) ===")
            print(f"  Delta_sigma FIXED from Stage 1: {Delta_sigma_fixed:.4f}")
        
        # Generation-dependent initial guesses (from expected mass hierarchy)
        # Legacy uses: n=1 → A=0.9, n=2 → A=12.0, n=3 → A=50.0
        if generation_n == 1:  # Electron
            A_0 = 0.9
        elif generation_n == 2:  # Muon
            A_0 = 12.0
        elif generation_n == 3:  # Tau
            A_0 = 50.0
        else:
            A_0 = max(1.0, 5.0 * generation_n)
        
        # Auto-generate initial guess based on generation
        if initial_guess is None:
            Delta_x_0 = 10.0  # fm (typical atomic scale)
            initial_guess = (Delta_x_0, A_0)
        
        if self.verbose:
            print(f"  Initial guess: Delta_x={initial_guess[0]:.3f} fm, A={initial_guess[1]:.3f}")
        
        # Minimize
        result = self._minimize_energy(
            chi_normalized=chi_normalized,
            Delta_sigma_fixed=Delta_sigma_fixed,
            initial_guess=initial_guess,
            method=method,
            particle_type='lepton',
            n_target=generation_n
        )
        
        return result
    
    def minimize_meson_energy(
        self,
        chi_normalized: NDArray,
        Delta_sigma_fixed: float,
        initial_guess: Optional[Tuple[float, float]] = None,
        method: str = 'Nelder-Mead'
    ) -> EnergyMinimizationResult:
        """
        Find optimal (Delta_x, A) for meson with FIXED Delta_sigma from Stage 1.
        
        Args:
            chi_normalized: Normalized subspace shape from Stage 1
            Delta_sigma_fixed: Fixed natural width from Stage 1 shape solver
            initial_guess: (Delta_x_0, A_0) or None for automatic
            method: Optimization method
            
        Returns:
            EnergyMinimizationResult with optimal parameters and mass
        """
        if self.verbose:
            print("\n=== Minimizing Meson Energy ===")
            print(f"  Delta_sigma FIXED from Stage 1: {Delta_sigma_fixed:.4f}")
        
        # Auto-generate initial guess
        if initial_guess is None:
            A_0 = 40.0  # Typical meson amplitude (between lepton and baryon)
            Delta_x_0 = 1.0  # fm (typical hadronic scale)
            initial_guess = (Delta_x_0, A_0)
        
        # Minimize
        result = self._minimize_energy(
            chi_normalized=chi_normalized,
            Delta_sigma_fixed=Delta_sigma_fixed,
            initial_guess=initial_guess,
            method=method,
            particle_type='meson'
        )
        
        return result
    
    def _minimize_energy(
        self,
        chi_normalized: NDArray,
        Delta_sigma_fixed: float,
        initial_guess: Tuple[float, float],
        method: str,
        particle_type: str,
        n_target: Optional[int] = None,
        k_winding: Optional[int] = None,
        A_bounds: Optional[Tuple[float, float]] = None
    ) -> EnergyMinimizationResult:
        """
        Minimize energy using NESTED OPTIMIZATION approach with FIXED Delta_sigma.
        
        ARCHITECTURE (FULLY COUPLED - FIRST PRINCIPLES):
        =================================================
        Outer Loop: Optimize A (primary variable, 1D search)
          └─> Inner Loop: For each A, optimize Δx by minimizing full 5D energy (1D search)
               └─> Energy computed from full 5D wavefunction with FIXED Δσ from Stage 1
                    ├─> Spatial coupling structure rebuilt at current Δx
                    └─> All Hamiltonian operators applied
        
        This ensures complete self-consistency: the wavefunction structure
        depends on (A, Δx) through the spatial coupling, which is rebuilt
        for every energy evaluation. Delta_sigma is FIXED from Stage 1.
        
        Energy components (computed from 5D wavefunction):
        - E_spatial: Spatial kinetic energy from ∇² operator
        - E_sigma: Subspace kinetic + potential + nonlinear
        - E_coupling: Mixed derivative ∂²/∂r∂σ operator [generation-dependent!]
        - E_em: Circulation penalty
        - E_curvature: ∫|∇φ_n|² (generation-dependent via spatial gradients)
        
        Args:
            chi_normalized: Normalized subspace shape from Stage 1
            Delta_sigma_fixed: Fixed natural width from Stage 1 shape solver
            initial_guess: (Delta_x_0, A_0)
            method: Optimization method ('L-BFGS-B' recommended)
            particle_type: 'lepton', 'meson', or 'baryon'
            n_target: Target generation (for leptons)
            k_winding: Winding number
            A_bounds: Optional bounds on A
            
        Returns:
            EnergyMinimizationResult
        """
        # CRITICAL: Scale-independent energy minimization (first-principles)
        # The optimal amplitude A emerges purely from energy balance and field dynamics
        # No external mass scale affects the energy minimization process
        
        # Default n_target if not specified (for baryons/mesons)
        if n_target is None:
            n_target = 1  # Use ground state spatial structure
        
        # Extract initial guess
        Delta_x_initial, A_initial = initial_guess
        
        # Set bounds
        if A_bounds is not None:
            MIN_A, MAX_A = A_bounds
        else:
            MIN_A = 0.001  # Small but non-zero
            MAX_A = 100.0  # Maximum reasonable amplitude
        
        MIN_DELTA_X = 0.001   # fm - minimum localization
        MAX_DELTA_X = 100000.0  # fm - maximum spatial extent (widened to allow escape from local minima)
        
        # Ensure initial guess is within bounds
        A_initial = max(MIN_A, min(MAX_A, A_initial))
        
        if self.verbose:
            print(f"  NESTED Energy minimization:")
            print(f"    Delta_sigma FIXED: {Delta_sigma_fixed:.4f}")
            print(f"    Outer: Optimize A (1D)")
            print(f"    Inner: For each A, optimize Delta_x (1D)")
            print(f"  Initial A: {A_initial:.6f}")
            print(f"  Method: {method}")
        
        # Track iterations for reporting
        inner_optimization_count = [0]
        total_inner_iterations = [0]
        
        # === INNER OPTIMIZATION: For fixed A, find optimal Δx (Δσ is FIXED from Stage 1) ===
        def optimize_scales_for_amplitude(A: float) -> Tuple[float, float]:
            """
            For a given amplitude A, find the Δx that minimizes total energy.
            Delta_sigma is FIXED from Stage 1.
            
            Returns:
                (E_total, Delta_x_opt)
            """
            # Use analytical formula for initial guess
            Delta_x_init = self._compute_optimal_delta_x(A)
            
            # DON'T clip initial guess - let optimizer start from physics-based prediction
            # The bounds will still be enforced by minimize(), but we won't artificially
            # constrain the starting point, which was causing different local minima
            # to be found depending on bound values
            
            x0 = np.array([Delta_x_init])
            bounds = [(MIN_DELTA_X, MAX_DELTA_X)]
            
            # Inner objective: energy as function of Δx at fixed A and Δσ
            def energy_for_scales(x: NDArray) -> float:
                Delta_x = x[0]
                E_total, _ = self._compute_total_energy(
                    chi_normalized, n_target, Delta_x, Delta_sigma_fixed, A
                )
                return E_total
            
            # Optimize Δx only
            result = minimize(
                energy_for_scales,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 100}
            )
            
            # Track iterations
            inner_optimization_count[0] += 1
            total_inner_iterations[0] += result.nit
            
            Delta_x_opt = result.x[0]
            E_total = result.fun
            
            return E_total, Delta_x_opt
        
        # === OUTER OPTIMIZATION: Find optimal A ===
        def energy_for_amplitude(A: float) -> float:
            """
            Outer objective: for this A, optimize Δx and return minimum energy.
            Delta_sigma is FIXED from Stage 1.
            """
            E_total, _ = optimize_scales_for_amplitude(A)
            return E_total
        
        if self.verbose:
            print(f"  Running nested minimization...")
        
        # Optimize A using bounded scalar minimization
        from scipy.optimize import minimize_scalar
        
        outer_result = minimize_scalar(
            energy_for_amplitude,
            bounds=(MIN_A, MAX_A),
            method='bounded',
            options={'xatol': 1e-6, 'maxiter': 100}
        )
        
        # Extract optimal A
        A_opt = outer_result.x
        converged = outer_result.success
        outer_iterations = outer_result.nit
        
        # Get the final optimal scales for this A
        E_total_opt, Delta_x_opt = optimize_scales_for_amplitude(A_opt)
        
        # Total iterations (approximate - outer × inner)
        total_iterations = outer_iterations + total_inner_iterations[0]
        
        if self.verbose:
            print(f"  Nested optimization complete:")
            print(f"    Status: {'CONVERGED' if converged else 'FAILED'}")
            print(f"    Outer iterations: {outer_iterations}")
            print(f"    Inner optimizations: {inner_optimization_count[0]}")
            print(f"    Total inner iterations: {total_inner_iterations[0]}")
            print(f"    Optimal A: {A_opt:.6f}")
            print(f"    Optimal Delta_x: {Delta_x_opt:.6f} fm")
            print(f"    Delta_sigma (FIXED from Stage 1): {Delta_sigma_fixed:.6f}")
        
        # === COMPUTE FINAL ENERGY AND COMPONENTS (single computation) ===
        E_total, energy_components = self._compute_total_energy(
            chi_normalized, n_target, Delta_x_opt, Delta_sigma_fixed, A_opt
        )
        
        # Extract individual components from the returned dictionary
        E_spatial = energy_components['E_spatial']
        E_sigma = energy_components['E_sigma']
        E_kin_sigma = energy_components['E_kinetic_sigma']
        E_pot_sigma = energy_components['E_potential_sigma']
        E_nl_sigma = energy_components['E_nonlinear_sigma']
        E_coupling = energy_components['E_coupling']
        E_em = energy_components['E_em']
        E_curv = energy_components['E_curvature']
        
        if self.verbose:
            print(f"\n  Final solution:")
            print(f"    A = {A_opt:.6f}")
            print(f"    Delta_x = {Delta_x_opt:.6f} fm")
            print(f"    Delta_sigma = {Delta_sigma_opt:.6f}")
            print(f"    E_total = {E_total:.6e} GeV")
            print(f"  Energy breakdown:")
            print(f"    E_sigma:    {E_sigma:.6e} GeV")
            print(f"    E_spatial:  {E_spatial:.6e} GeV")
            print(f"    E_coupling: {E_coupling:.6e} GeV")
            print(f"    E_em:       {E_em:.6e} GeV")
            print(f"    E_curv:     {E_curv:.6e} GeV (gravitational self-confinement)")
        
        # === MASS IS NOT COMPUTED HERE ===
        # Mass calculation is done externally using The Beautiful Equation
        # m = G_5D * A^2 in natural units
        mass = None
        
        convergence_message = f"Converged (nested optimization, {method})" if converged else "Nested minimization failed"
        
        return EnergyMinimizationResult(
            Delta_x=Delta_x_opt,
            Delta_sigma=Delta_sigma_fixed,  # FIXED from Stage 1
            A=A_opt,
            mass=mass,
            E_total=E_total,
            E_sigma=E_sigma,
            E_kinetic_sigma=E_kin_sigma,
            E_potential_sigma=E_pot_sigma,
            E_nonlinear_sigma=E_nl_sigma,
            E_spatial=E_spatial,
            E_coupling=E_coupling,
            E_em=E_em,
            E_curvature=E_curv,
            converged=converged,
            iterations=total_iterations,
            optimization_message=convergence_message,
            particle_type=particle_type,
            n_target=n_target,
            k_winding=k_winding
        )
    
    def _compute_total_energy(
        self,
        chi_normalized: NDArray,
        n_target: int,
        Delta_x: float,
        Delta_sigma: float,
        A: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total energy with FRESH structure at current scales.
        
        FULLY COUPLED IMPLEMENTATION:
        ==============================
        This method is called 100-1000 times per outer iteration.
        Each call rebuilds the full structure to ensure self-consistency
        between scale parameters (A, Δx, Δσ) and the wavefunction structure.
        
        The spatial coupling structure is rebuilt at the current Δx,
        ensuring that all three scale parameters are properly reflected
        in the wavefunction shape.
        
        Energy Components:
        - E_spatial: Spatial kinetic energy from -ℏ²/(2m)∇²
        - E_sigma: Subspace kinetic + potential + nonlinear
        - E_coupling: Mixed derivative -α(∂²/∂x∂σ + ...)
        - E_em: Circulation penalty g₂|J|²
        - E_curv: Gravitational self-confinement G_5D³A⁴/Δx
        
        Args:
            chi_normalized: Normalized subspace shape from Stage 1 (scale-independent)
            n_target: Generation quantum number (1=e, 2=mu, 3=tau)
            Delta_x: Spatial extent (fm)
            Delta_sigma: Subspace width (dimensionless)
            A: Amplitude (dimensionless)
            
        Returns:
            Tuple of (E_total, energy_components_dict)
        """
        # STEP 1: Rebuild structure at current Delta_x
        # This ensures spatial coupling reflects the current trial parameters
        structure_4d = self.spatial_coupling.build_4d_structure(
            subspace_shape=chi_normalized,
            n_target=n_target,
            l_target=0,
            m_target=0,
            Delta_x=Delta_x  # Use CURRENT trial value
        )
        
        # STEP 2: Sum components and scale by (A/√Δσ)
        chi_shape = sum(structure_4d.values())
        scaling_factor = A / np.sqrt(Delta_sigma)
        chi_sigma = scaling_factor * chi_shape
        
        # STEP 3: Build spatial wavefunction at current Delta_x
        phi_r = self.field_energy.build_spatial_wavefunction(n_target, Delta_x)
        
        # STEP 4: Compute all energies from full 5D wavefunction
        E_total, components = self.field_energy.compute_all_energies(
            phi_r=phi_r,
            chi_sigma=chi_sigma,
            n_target=n_target,
            Delta_x=Delta_x,
            Delta_sigma=Delta_sigma,
            A=A
        )
        
        return E_total, components
