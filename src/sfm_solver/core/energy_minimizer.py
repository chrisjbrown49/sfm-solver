"""
Universal Energy Minimizer for SFM.

This module implements Stage 2 of the two-stage solver architecture.
Takes normalized shape from Stage 1 and finds optimal scale parameters
(Delta_x, Delta_sigma, A) by minimizing total energy.

KEY PRINCIPLE: Stage 2 (physical scales)
    - Input: Normalized shape structure from Stage 1
    - Output: Optimal (Delta_x, Delta_sigma, A)
    - Energy functional: E_total(Delta_x, Delta_sigma, A)
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
            verbose: Print diagnostic information
        """
        self.G_5D = G_5D  # FUNDAMENTAL - 5D gravitational constant
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        self.V1 = V1
        self.alpha = alpha
        self.verbose = verbose
        
        # Radial grid for spatial wavefunction integrals
        self._r_grid = np.linspace(0.01, 20.0, 500)
        
        if self.verbose:
            print("=== UniversalEnergyMinimizer Initialized ===")
            print(f"  G_5D: {G_5D:.6e} GeV^-2 (5D Gravitational constant)")
            print(f"  g1: {g1:.3f} (Nonlinear subspace coupling)")
            print(f"  g2: {g2:.6f} (EM circulation coupling)")
            print(f"  alpha: {alpha:.3f} GeV (Spatial-subspace coupling)")
            print(f"  V0: {V0:.6f} GeV, V1: {V1:.6f} GeV (Three-well depths)")
            print(f"  Energy minimization: Scale-independent (first-principles)")
    
    # =========================================================================
    # SPATIAL COUPLING FACTOR (Generation-Dependent Physics)
    # =========================================================================
    
    def _compute_spatial_factor(self, n: int, delta_x: float) -> float:
        """
        Compute the spatial coupling factor for harmonic oscillator wavefunction.
        
        CRITICAL PHYSICS - THIS IS THE MISSING GENERATION-DEPENDENT TERM:
        - n determines the number of radial nodes (n-1 nodes for s-wave)
        - delta_x determines the spatial scale: a = delta_x / sqrt(2n+1)
        
        The spatial wavefunction is:
            φ_n(r; Δx) = N × L_{n-1}^{1/2}(r²/a²) × exp(-r²/(2a²))
        
        where a = Δx/√(2n+1).
        
        The CORRECT spatial coupling factor (from Math Formulation Part A) is:
            ∫ |∇φ|² d³x = 4π ∫ (dφ/dr)² r² dr
        
        This is the SQUARED GRADIENT, which:
        - Is non-zero for all states (unlike ∫φ×dφ/dr which is always zero!)
        - Scales with n (more nodes → larger gradients)
        - Scales with 1/a² = (2n+1)/Δx² (tighter confinement → stronger gradients)
        
        This creates the generation hierarchy:
        - Electron (n=1): φ₁ = Gaussian (no nodes) → small gradient integral
        - Muon (n=2): φ₂ = (1-2r²/a²) × Gaussian (1 node) → larger gradient
        - Tau (n=3): φ₃ = (1-4r²/a²+2r⁴/a⁴) × Gaussian (2 nodes) → even larger
        
        Args:
            n: Spatial quantum number (1 for electron, 2 for muon, 3 for tau)
            delta_x: Spatial extent in fm (optimization variable)
            
        Returns:
            Spatial coupling factor (always positive, scales as (2n+1)/Δx²)
        """
        if delta_x < 1e-10:
            return 1e10  # Return large value for small delta_x
        
        # Scale parameter: a = Δx / √(2n+1)
        # This ensures the characteristic size is Δx, adjusted for the number of nodes
        a = delta_x / np.sqrt(2 * n + 1)
        
        if a < 1e-10:
            return 1e10
        
        r = self._r_grid
        x = (r / a) ** 2
        
        # Laguerre polynomial parameters for s-wave (l=0) harmonic oscillator
        # φ_n has n-1 radial nodes, corresponding to L_{n-1}^{1/2}(x)
        k = n - 1  # Polynomial degree
        alpha_lag = 0.5  # For 3D harmonic oscillator s-wave
        
        # Compute Laguerre polynomial and its derivative
        if k >= 0:
            L = genlaguerre(k, alpha_lag)(x)
            if k >= 1:
                dL_dx = -genlaguerre(k - 1, alpha_lag + 1)(x)
            else:
                dL_dx = np.zeros_like(x)
        else:
            L = np.ones_like(x)
            dL_dx = np.zeros_like(x)
        
        # Radial wavefunction (unnormalized)
        # φ = L(x) × exp(-x/2)  where x = (r/a)²
        exp_factor = np.exp(-x / 2)
        phi = L * exp_factor
        
        # Derivative: dφ/dr = dφ/dx × dx/dr = dφ/dx × (2r/a²)
        # dφ/dx = dL/dx × exp(-x/2) + L × (-1/2) × exp(-x/2)
        #       = exp(-x/2) × [dL/dx - L/2]
        dphi_dx = exp_factor * (dL_dx - L / 2)
        dphi_dr = dphi_dx * (2 * r / a**2)
        
        # Normalize: ∫ 4π φ² r² dr = 1
        norm_sq = 4 * np.pi * np.trapz(phi**2 * r**2, r)
        if norm_sq > 1e-20:
            phi = phi / np.sqrt(norm_sq)
            dphi_dr = dphi_dr / np.sqrt(norm_sq)
        
        # CORRECT spatial coupling factor: ∫ |∇φ|² d³x = 4π ∫ (dφ/dr)² r² dr
        # This is the SQUARED GRADIENT which is always positive and non-zero!
        # Scales as (2n+1)/Δx² - THIS IS WHERE GENERATION HIERARCHY COMES FROM!
        spatial_factor_fm = 4 * np.pi * np.trapz(dphi_dr**2 * r**2, r)  # Units: 1/fm²
        
        # CRITICAL: Make dimensionless by multiplying by Delta_x²
        # This gives spatial_factor = (Delta_x² × ∫|∇φ|² d³x), which is dimensionless
        # and represents the integrated gradient strength relative to the confinement scale.
        # The factor Delta_x² makes the coupling energy formula dimensionally correct:
        #   E_coupling [GeV] = alpha [GeV] × spatial_factor [dimensionless] × circ [dimensionless] × A [dimensionless]
        spatial_factor_dimensionless = spatial_factor_fm * (delta_x ** 2)  # Dimensionless
        
        return float(spatial_factor_dimensionless)
    
    # =========================================================================
    # ANALYTICAL FORMULAS for optimal parameters from energy balance
    # =========================================================================
    
    def _compute_optimal_delta_sigma(self, A: float) -> float:
        """
        Compute optimal subspace width from energy balance.
        
        FIRST-PRINCIPLES DERIVATION (from legacy solver):
        ================================================
        
        The subspace energy has two competing terms:
            E_kin,σ ~ 1/Δσ²     (kinetic - prefers large Δσ)
            E_nonlin ~ g₁A⁴/Δσ  (nonlinear - prefers large Δσ)
        
        Total: E = C₁/Δσ² + C₂×g₁×A⁴/Δσ
        
        Minimize: dE/dΔσ = -2C₁/Δσ³ - C₂×g₁×A⁴/Δσ² = 0
        
        This gives: Δσ_opt = (2/(g₁×A⁴))^(1/3) ∝ 1/(g₁×A⁴)^(1/3)
        
        CRITICAL: This prevents infinite delocalization by deriving Δσ from A
        rather than treating it as an independent optimization variable.
        
        Args:
            A: Current amplitude estimate
            
        Returns:
            Optimal delta_sigma from energy balance (bounded [0.1, 2.0])
        """
        if self.g1 <= 0 or A < 1e-10:
            return 0.5  # Default fallback
        
        # From energy minimization: Δσ_opt ∝ 1/(g₁×A⁴)^(1/3)
        A_fourth = A ** 4
        delta_sigma_opt = (2.0 / (self.g1 * A_fourth)) ** (1.0/3.0)
        
        # FIRST-PRINCIPLES: Apply physics-based bounds
        # - Minimum: envelope must be wide enough for spatial-subspace coupling
        # - Maximum: prevents infinite delocalization
        MIN_DELTA_SIGMA = 0.1   # Physics-based floor
        MAX_DELTA_SIGMA = 2.0   # Reasonable ceiling
        delta_sigma_opt = max(MIN_DELTA_SIGMA, min(MAX_DELTA_SIGMA, delta_sigma_opt))
        
        return delta_sigma_opt
    
    def _compute_optimal_delta_x(self, A: float) -> float:
        """
        Compute optimal spatial scale from gravitational self-confinement.
        
        FIRST-PRINCIPLES DERIVATION:
        ================================================
        
        The spatial confinement comes from gravitational self-energy balance.
        From "Origin of Mass and Gravity" (Section 4) and "Math Formulation (Part A)":
        
            Δx = (ℏ²/(G_5D × β³ × A⁶))^(1/3)
        
        For initial guess, approximate β ~ 1 GeV:
            Δx ≈ (ℏ²/(G_5D × A⁶))^(1/3)  [in GeV^-1]
            Δx_fm ≈ ℏc / (G_5D × A⁶)^(1/3)  [in fm]
        
        CRITICAL: The (1/3) power is essential! This comes from the cubic
        relationship between gravitational binding energy and spatial scale.
        
        This formula ensures:
        - Larger amplitude A → smaller Δx (more confined, Δx ∝ A^(-2))
        - Stronger G_5D → smaller Δx (more confined)
        - Prevents infinite spatial delocalization
        
        Args:
            A: Current amplitude estimate
            
        Returns:
            Optimal Delta_x from gravitational self-confinement (in fm)
        """
        if self.G_5D <= 0 or A < 1e-10:
            return 1.0  # Default fallback in fm
        
        # From gravitational self-confinement
        # Δx_fm = ℏc / (G_5D × A⁶)^(1/3) with β ~ 1 GeV approximation for initial guess
        A_sixth = A ** 6
        delta_x = HBAR_C_GEV_FM / (self.G_5D * A_sixth) ** (1.0/3.0)  # Result in fm
        
        # Apply reasonable physical bounds (spatial scale should be in fm range)
        MIN_DELTA_X = 0.01   # fm - minimum localization 
        MAX_DELTA_X = 100.0  # fm - maximum spread 
        delta_x_opt = max(MIN_DELTA_X, min(MAX_DELTA_X, delta_x))
        
        return delta_x_opt
    
    def minimize_baryon_energy(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        initial_guess: Optional[Tuple[float, float, float]] = None,
        method: str = 'Nelder-Mead'
    ) -> EnergyMinimizationResult:
        """
        Find optimal (Delta_x, Delta_sigma, A) for baryon.
        
        Args:
            shape_structure: Normalized 4D shape from Stage 1
                            {(n,l,m): chi_nlm(sigma)} with integral_Sigma|chi|^2 = 1
            initial_guess: (Delta_x_0, Delta_sigma_0, A_0) or None for automatic
            method: Optimization method
            
        Returns:
            EnergyMinimizationResult with optimal parameters and mass
        """
        if self.verbose:
            print("\n=== Minimizing Baryon Energy ===")
        
        # Auto-generate initial guess if not provided
        if initial_guess is None:
            A_0 = 60.0  # Typical baryon amplitude
            # Use gravitational self-confinement formula WITH (1/3) power
            Delta_x_0 = self._compute_optimal_delta_x(A_0)  # Already in fm, includes (1/3) power
            Delta_sigma_0 = 0.5
            initial_guess = (Delta_x_0, Delta_sigma_0, A_0)
        
        if self.verbose:
            print(f"  Initial guess: Delta_x={initial_guess[0]:.3f} fm, "
                  f"Delta_sigma={initial_guess[1]:.3f}, A={initial_guess[2]:.3f}")
        
        # Minimize
        result = self._minimize_energy(
            shape_structure=shape_structure,
            initial_guess=initial_guess,
            method=method,
            particle_type='baryon'
        )
        
        return result
    
    def minimize_lepton_energy(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        generation_n: int,
        initial_guess: Optional[Tuple[float, float, float]] = None,
        method: str = 'Nelder-Mead'
    ) -> EnergyMinimizationResult:
        """
        Find optimal (Delta_x, Delta_sigma, A) for lepton.
        
        Args:
            shape_structure: Normalized 4D shape from Stage 1
            generation_n: Generation number (1, 2, 3)
            initial_guess: (Delta_x_0, Delta_sigma_0, A_0) or None for automatic
            method: Optimization method
            
        Returns:
            EnergyMinimizationResult with optimal parameters and mass
        """
        if self.verbose:
            print(f"\n=== Minimizing Lepton Energy (n={generation_n}) ===")
        
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
            # Compute Delta_x consistent with A
            Delta_x_0 = self._compute_optimal_delta_x(A_0)
            # Compute Delta_sigma consistent with A
            Delta_sigma_0 = self._compute_optimal_delta_sigma(A_0)
            initial_guess = (Delta_x_0, Delta_sigma_0, A_0)
        
        if self.verbose:
            print(f"  Initial guess: Delta_x={initial_guess[0]:.3f} fm, "
                  f"Delta_sigma={initial_guess[1]:.3f}, A={initial_guess[2]:.3f}")
        
        # Minimize
        result = self._minimize_energy(
            shape_structure=shape_structure,
            initial_guess=initial_guess,
            method=method,
            particle_type='lepton',
            n_target=generation_n
        )
        
        return result
    
    def minimize_meson_energy(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        initial_guess: Optional[Tuple[float, float, float]] = None,
        method: str = 'Nelder-Mead'
    ) -> EnergyMinimizationResult:
        """
        Find optimal (Delta_x, Delta_sigma, A) for meson.
        
        Args:
            shape_structure: Normalized 4D shape from Stage 1
            initial_guess: (Delta_x_0, Delta_sigma_0, A_0) or None for automatic
            method: Optimization method
            
        Returns:
            EnergyMinimizationResult with optimal parameters and mass
        """
        if self.verbose:
            print("\n=== Minimizing Meson Energy ===")
        
        # Auto-generate initial guess
        if initial_guess is None:
            A_0 = 40.0  # Typical meson amplitude (between lepton and baryon)
            # Use gravitational self-confinement formula WITH (1/3) power
            Delta_x_0 = self._compute_optimal_delta_x(A_0)  # Already in fm, includes (1/3) power
            Delta_sigma_0 = 0.5
            initial_guess = (Delta_x_0, Delta_sigma_0, A_0)
        
        # Minimize
        result = self._minimize_energy(
            shape_structure=shape_structure,
            initial_guess=initial_guess,
            method=method,
            particle_type='meson'
        )
        
        return result
    
    def _minimize_energy(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        initial_guess: Tuple[float, float, float],
        method: str,
        particle_type: str,
        n_target: Optional[int] = None,
        k_winding: Optional[int] = None,
        A_bounds: Optional[Tuple[float, float]] = None
    ) -> EnergyMinimizationResult:
        """
        Minimize energy using full 3D optimization over (Delta_x, Delta_sigma, A).
        
        FIRST-PRINCIPLES APPROACH:
        =========================================
        All three parameters are independent optimization variables.
        The energy functional includes:
        - E_spatial: Quantum confinement (∝ 1/(A²·Δx²))
        - E_sigma: Subspace field energy
        - E_coupling: Spatial-subspace coupling (generation-dependent!)
        - E_em: Electromagnetic circulation
        - E_curvature: Gravitational self-confinement (∝ A⁴/Δx)
        
        The interplay between E_spatial (wants large Δx) and E_curvature (wants small Δx)
        naturally determines the optimal spatial scale. The optimizer finds the balance.
        
        Args:
            shape_structure: Normalized shape from Stage 1
            initial_guess: (Delta_x_0, Delta_sigma_0, A_0)
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
        Delta_x_initial, Delta_sigma_initial, A_initial = initial_guess
        
        # Set bounds for all three parameters
        if A_bounds is not None:
            MIN_A, MAX_A = A_bounds
        else:
            MIN_A = 0.001  # Small but non-zero
            MAX_A = 100.0  # Maximum reasonable amplitude
        
        MIN_DELTA_X = 0.001   # fm - minimum localization
        MAX_DELTA_X = 100.0   # fm - maximum spread
        MIN_DELTA_SIGMA = 0.1  # Minimum subspace width
        MAX_DELTA_SIGMA = 2.0  # Maximum subspace width
        
        # Ensure initial guess is within bounds
        Delta_x_initial = max(MIN_DELTA_X, min(MAX_DELTA_X, Delta_x_initial))
        Delta_sigma_initial = max(MIN_DELTA_SIGMA, min(MAX_DELTA_SIGMA, Delta_sigma_initial))
        A_initial = max(MIN_A, min(MAX_A, A_initial))
        
        x0 = np.array([Delta_x_initial, Delta_sigma_initial, A_initial])
        bounds = [(MIN_DELTA_X, MAX_DELTA_X), (MIN_DELTA_SIGMA, MAX_DELTA_SIGMA), (MIN_A, MAX_A)]
        
        if self.verbose:
            print(f"  3D Energy minimization over (Delta_x, Delta_sigma, A)")
            print(f"  Initial guess: Delta_x={Delta_x_initial:.6f} fm, Delta_sigma={Delta_sigma_initial:.6f}, A={A_initial:.6f}")
            print(f"  Method: {method}")
        
        # Define objective function for 3D optimization
        def energy_objective_3d(x: NDArray) -> float:
            """
            Compute total energy as a function of (Delta_x, Delta_sigma, A).
            All three are independent optimization variables.
            """
            Delta_x, Delta_sigma, A = x
            
            # Compute and return total energy (ignore components for optimization)
            E_total, _ = self._compute_total_energy(
                shape_structure, Delta_x, Delta_sigma, A, n_target
            )
            return E_total
        
        # Use scipy's minimize for full 3D optimization
        if self.verbose:
            print(f"  Running 3D minimization...")
        
        optimization_result = minimize(
            energy_objective_3d,
            x0,
            method=method,  # 'L-BFGS-B' for bounded optimization
            bounds=bounds,
            options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 200}
        )
        
        # Extract optimal parameters
        Delta_x_opt, Delta_sigma_opt, A_opt = optimization_result.x
        converged = optimization_result.success
        iterations = optimization_result.nit  # Number of iterations
        
        if self.verbose:
            print(f"  Optimization complete:")
            print(f"    Status: {'CONVERGED' if converged else 'FAILED'}")
            print(f"    Iterations: {iterations}")
            print(f"    Optimal Delta_x: {Delta_x_opt:.6f} fm")
            print(f"    Optimal Delta_sigma: {Delta_sigma_opt:.6f}")
            print(f"    Optimal A: {A_opt:.6f}")
        
        # === COMPUTE FINAL ENERGY AND COMPONENTS (single computation) ===
        E_total, energy_components = self._compute_total_energy(
            shape_structure, Delta_x_opt, Delta_sigma_opt, A_opt, n_target
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
            print(f"    Delta_x = {Delta_x_opt:.6f} fm")
            print(f"    Delta_sigma = {Delta_sigma_opt:.6f}")
            print(f"    A = {A_opt:.6f}")
            print(f"    E_total = {E_total:.6e} GeV")
            print(f"  Energy breakdown:")
            print(f"    E_sigma:    {E_sigma:.6e} GeV")
            print(f"    E_spatial:  {E_spatial:.6e} GeV")
            print(f"    E_coupling: {E_coupling:.6e} GeV")
            print(f"    E_em:       {E_em:.6e} GeV")
            print(f"    E_curv:     {E_curv:.6e} GeV (gravitational self-confinement)")
        
        # === MASS IS NOT COMPUTED HERE ===
        # Mass calculation is done externally using calculate_beta helper
        # m = beta * A^2, where beta is calibrated from electron
        mass = None
        
        convergence_message = f"Converged (3D minimization, {method})" if converged else "Minimization failed"
        
        return EnergyMinimizationResult(
            Delta_x=Delta_x_opt,
            Delta_sigma=Delta_sigma_opt,
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
            iterations=iterations,
            optimization_message=convergence_message,
            particle_type=particle_type,
            n_target=n_target,
            k_winding=k_winding
        )
    
    def _scale_wavefunctions(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        Delta_sigma: float,
        A: float
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Scale normalized shape to physical configuration.
        
        Input: psi_shape(sigma) with integral|psi|^2 dsigma = 1
        Output: chi(sigma) = (A/sqrt(Delta_sigma)) * psi_shape(sigma/Delta_sigma)
        
        Check: integral|chi|^2 dsigma = (A^2/Delta_sigma) * integral|psi(s)|^2 ds * Delta_sigma = A^2
        
        For simplicity, we assume Delta_sigma ~ 1 and use direct amplitude scaling.
        Proper implementation would interpolate for general Delta_sigma.
        """
        chi_scaled = {}
        
        scaling_factor = A / np.sqrt(Delta_sigma)
        
        for (n, l, m), psi_shape in shape_structure.items():
            chi_scaled[(n, l, m)] = scaling_factor * psi_shape
        
        return chi_scaled
    
    def _compute_kinetic_sigma(
        self,
        chi_scaled: Dict[Tuple[int, int, int], NDArray],
        Delta_sigma: float
    ) -> float:
        """
        Compute kinetic energy in subspace dimension.
        
        E_kin = integral (hbar^2 / 2m_sigma R^2) |d(chi)/dsigma|^2 dsigma
        
        For scaled wavefunction chi(sigma) = (A/sqrt(Delta_sigma)) * psi(sigma/Delta_sigma):
        E_kin ~ (A^2 / Delta_sigma^2) * (kinetic integral of normalized shape)
        
        Using natural units: hbar = c = 1
        """
        # Get composite wavefunction
        chi_total = sum(chi_scaled.values())
        N = len(chi_total)
        dsigma = 2*np.pi / N
        
        # Build derivative operator
        D1 = self._build_derivative_operator(N)
        dchi = D1 @ chi_total
        
        # Kinetic energy integral
        E_kin = np.sum(np.abs(dchi)**2) * dsigma
        
        # Scale factors (assuming natural units)
        # In full units: (hbar^2 / 2m_sigma R^2) with appropriate conversion
        # For now, use dimensionless version scaled by typical energy scale
        E_kin = E_kin / (2.0 * Delta_sigma**2)
        
        return E_kin
    
    def _compute_potential_sigma(self, chi_scaled: Dict) -> float:
        """
        Compute potential energy in subspace dimension.
        
        E_pot = integral V(sigma) |chi|^2 dsigma
        """
        chi_total = sum(chi_scaled.values())
        N = len(chi_total)
        dsigma = 2*np.pi / N
        sigma = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        # Three-well potential
        V_well = self.V0 * (1 - np.cos(3*sigma)) + self.V1 * (1 - np.cos(6*sigma))
        
        E_pot = np.sum(V_well * np.abs(chi_total)**2) * dsigma
        
        return E_pot
    
    def _compute_nonlinear_sigma(
        self,
        chi_scaled: Dict,
        Delta_sigma: float
    ) -> float:
        """
        Compute nonlinear self-interaction energy.
        
        E_nl = (g1/2) * integral |chi|^4 dsigma
        """
        chi_total = sum(chi_scaled.values())
        N = len(chi_total)
        dsigma = 2*np.pi / N
        
        E_nl = (self.g1 / 2.0) * np.sum(np.abs(chi_total)**4) * dsigma / Delta_sigma
        
        return E_nl
    
    def _compute_coupling_energy(
        self,
        chi_scaled: Dict,
        Delta_x: float,
        Delta_sigma: float,
        A: float,
        n_target: int
    ) -> float:
        """
        Compute spatial-subspace coupling energy.
        
        THIS IS THE KEY TERM THAT CREATES GENERATION HIERARCHY!
        
        E_coupling = -alpha × spatial_factor(n, Δx) × subspace_factor × A
        
        Where:
        - spatial_factor(n, Δx) = ∫ |∇φ_n|² d³x   [depends on generation n!]
        - subspace_factor = Im[∫ χ* ∂χ/∂σ dσ]   [from subspace wavefunction]
        
        The spatial_factor is DIFFERENT for each generation:
        - Electron (n=1): Small gradient integral (no nodes)
        - Muon (n=2): Larger gradient integral (1 radial node)
        - Tau (n=3): Even larger gradient integral (2 radial nodes)
        
        This n-dependence creates different energy landscapes for each generation,
        leading to different optimal amplitudes and the observed mass hierarchy!
        
        Args:
            chi_scaled: Scaled subspace wavefunction
            Delta_x: Spatial extent in fm
            Delta_sigma: Subspace width (dimensionless)
            A: Amplitude
            n_target: Generation quantum number (1=e, 2=mu, 3=tau)
            
        Returns:
            Coupling energy contribution (natural energy units)
        """
        # Compute spatial factor (generation-dependent!)
        spatial_factor = self._compute_spatial_factor(n_target, Delta_x)
        
        # Compute subspace factor from circulation
        # CRITICAL: The circulation integral depends on Delta_sigma scaling!
        # For chi_scaled(σ) = (A/√Δσ) × chi_shape(σ/Δσ), the circulation is:
        # Im[∫ χ* ∂χ/∂σ dσ] = (A²/Δσ) × Im[∫ χ_shape* ∂χ_shape/∂σ dσ]
        #
        # So subspace_factor scales as A²/Δσ, but we want it relative to the
        # normalized shape (which has circ ~ 1), so we multiply by Δσ/A² to get
        # the dimensionless winding number, then multiply by A to get the right
        # energy scaling.
        chi_total = sum(chi_scaled.values())
        N = len(chi_total)
        dsigma = 2*np.pi / N
        
        # Build derivative operator
        D1 = np.zeros((N, N), dtype=complex)
        for i in range(N):
            D1[i, (i+1) % N] = 1.0
            D1[i, (i-1) % N] = -1.0
        D1 = D1 / (2 * dsigma)
        
        # Compute ∂χ/∂σ
        dchi_total = D1 @ chi_total
        
        # Circulation integral (imaginary part carries the winding)
        circulation_scaled = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        
        # CRITICAL FIX: Correct for Delta_sigma dependence
        # circulation_scaled = (A²/Δσ) × circulation_shape
        # We want: subspace_factor = circulation_shape (the winding number ~ 1)
        # So: subspace_factor = circulation_scaled × (Δσ/A²) × A = circulation_scaled × (Δσ/A)
        # This gives the correct A-scaling for energy while preserving the winding.
        subspace_factor = np.imag(circulation_scaled) * (Delta_sigma / A)
        
        # Total coupling energy (NOTE: negative sign - this is attractive!)
        E_coupling = -self.alpha * spatial_factor * subspace_factor * A
        
        return E_coupling
    
    def _compute_spatial_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute spatial confinement energy.
        
        From Math Formulation Part A, line 304:
            E_spatial = ℏ²/(2β A² Δx²)
        
        Substituting β = G_5D × c, in natural units (ℏ = c = 1):
            E_spatial = 1/(2 G_5D A² Δx²)  [GeV]
        
        Quantum confinement energy scales inversely with G_5D, amplitude squared,
        and spatial extent squared. Larger G_5D means weaker confinement energy.
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude (dimensionless)
            
        Returns:
            Spatial energy contribution (GeV)
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        if A < 1e-10:
            return 1e10  # Prevent division by zero
        
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        # E_spatial = 1/(2 G_5D A² Δx²) with G_5D in [GeV^-2], giving result in [GeV]
        E_x = 1.0 / (2.0 * self.G_5D * A**2 * Delta_x_nat**2)
        
        return E_x
    
    def _compute_curvature_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute gravitational self-confinement energy.
        
        From Math Formulation Part A, line 305:
            E_curv = G_5D β² A⁴/Δx
        
        Substituting β = G_5D × c, in natural units (c = 1):
            E_curv = G_5D³ c² A⁴/Δx
            E_curv = G_5D³ A⁴/Δx  [GeV]
        
        The field's own energy-density creates spacetime curvature that
        confines it. Stronger fields (larger A) in smaller regions create
        stronger gravitational self-confinement. This is the mechanism that
        prevents infinite spatial delocalization.
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude (dimensionless)
            
        Returns:
            Gravitational self-energy contribution (GeV)
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        # E_curv = G_5D³ A⁴/Δx with G_5D in [GeV^-2], giving result in [GeV]
        E_curv = (self.G_5D ** 3) * A**4 / Delta_x_nat
        
        return E_curv
    
    def _compute_em_energy(self, chi_scaled: Dict) -> float:
        """
        Compute electromagnetic self-energy from subspace circulation.
        
        E_em = g2 * |J|²
        
        where J = integral chi* d(chi)/dsigma dsigma (circulation current)
        
        The winding of the field in the compactified subspace dimension
        creates an electromagnetic circulation current. The self-energy
        of this current contributes to the total energy.
        
        Args:
            chi_scaled: Scaled wavefunction dictionary {(n,l,m): chi_nlm}
            
        Returns:
            EM circulation energy contribution (natural energy units)
        """
        chi_total = sum(chi_scaled.values())
        N = len(chi_total)
        dsigma = 2*np.pi / N
        
        # Compute circulation current J
        D1 = self._build_derivative_operator(N)
        dchi = D1 @ chi_total
        J = np.sum(np.conj(chi_total) * dchi) * dsigma
        
        # EM circulation energy
        E_em = self.g2 * np.abs(J)**2
        
        return E_em
    
    def _compute_total_energy(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        Delta_x: float,
        Delta_sigma: float,
        A: float,
        n_target: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total energy and all components from field contributions.
        
        The energy functional determines optimal scale parameters purely from
        field dynamics and energy balance. The amplitude A that minimizes this
        energy is the natural field strength for this configuration.
        
        Energy components:
        - E_spatial = 1/(2·G_5D·A²·Δx²)  [spatial quantum confinement]
        - E_sigma = kinetic + potential + nonlinear  [subspace field energy]
        - E_coupling = -alpha × spatial_factor(n,Δx) × subspace_factor × A  [GENERATION-DEPENDENT!]
        - E_em = g2·|J|²  [electromagnetic circulation]
        - E_curvature = G_5D³ × A⁴/Δx  [gravitational self-confinement]
        
        The minimum of this functional gives the stable field configuration.
        This is pure first-principles physics - no external scales required.
        
        CRITICAL: The coupling energy depends on n (generation number), which creates
        different energy landscapes for electron (n=1), muon (n=2), and tau (n=3).
        This is the physical mechanism behind the generation hierarchy!
        
        Args:
            shape_structure: Normalized shape from Stage 1
            Delta_x: Spatial extent (fm)
            Delta_sigma: Subspace width (dimensionless)
            A: Amplitude (dimensionless)
            n_target: Generation quantum number (1=e, 2=mu, 3=tau) - CREATES HIERARCHY!
            
        Returns:
            Tuple of (E_total, energy_components_dict)
            where energy_components_dict contains all individual energy terms
        """
        # Scale wavefunctions
        chi_scaled = self._scale_wavefunctions(shape_structure, Delta_sigma, A)
        
        # Compute energy components from field configuration
        E_spatial = self._compute_spatial_energy(Delta_x, A)
        
        # Subspace energy components
        E_kin_sigma = self._compute_kinetic_sigma(chi_scaled, Delta_sigma)
        E_pot_sigma = self._compute_potential_sigma(chi_scaled)
        E_nl_sigma = self._compute_nonlinear_sigma(chi_scaled, Delta_sigma)
        E_sigma = E_kin_sigma + E_pot_sigma + E_nl_sigma
        
        # Coupling energy (GENERATION-DEPENDENT!)
        E_coupling = self._compute_coupling_energy(chi_scaled, Delta_x, Delta_sigma, A, n_target)
        
        # Electromagnetic circulation energy
        E_em = self._compute_em_energy(chi_scaled)
        
        # Gravitational self-confinement energy (CRITICAL: prevents infinite delocalization!)
        E_curvature = self._compute_curvature_energy(Delta_x, A)
        
        # Total energy
        E_total = E_spatial + E_sigma + E_coupling + E_em + E_curvature
        
        # Package components into dictionary
        energy_components = {
            'E_spatial': E_spatial,
            'E_sigma': E_sigma,
            'E_kinetic_sigma': E_kin_sigma,
            'E_potential_sigma': E_pot_sigma,
            'E_nonlinear_sigma': E_nl_sigma,
            'E_coupling': E_coupling,
            'E_em': E_em,
            'E_curvature': E_curvature
        }
        
        return E_total, energy_components
    
    def _build_derivative_operator(self, N: int) -> NDArray:
        """
        Build spectral derivative operator for periodic functions.
        
        Uses FFT-based spectral differentiation.
        """
        # Wavenumbers for periodic domain [0, 2*pi]
        k = np.fft.fftfreq(N, d=1.0/N)
        
        # Derivative operator in Fourier space
        k_deriv = 1j * k
        
        # Build matrix representation
        D = np.zeros((N, N), dtype=complex)
        for i in range(N):
            delta = np.zeros(N)
            delta[i] = 1.0
            
            # Transform, multiply by ik, inverse transform
            delta_fft = np.fft.fft(delta)
            deriv_fft = k_deriv * delta_fft
            deriv = np.fft.ifft(deriv_fft)
            
            D[:, i] = deriv
        
        return D

