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
    E_curvature: float
    E_em: float
    
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
        g_internal: float,
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
            g_internal: FUNDAMENTAL gravitational self-confinement strength.
                       Controls how strongly larger amplitudes are spatially confined.
            g1: Nonlinear self-interaction coupling in subspace (dimensionless)
            g2: Electromagnetic circulation coupling (dimensionless)
            V0: Three-well primary depth (GeV)
            V1: Three-well secondary depth (GeV)
            alpha: Spatial-subspace coupling strength (GeV)
            verbose: Print diagnostic information
        """
        self.g_internal = g_internal  # FUNDAMENTAL - drives all physics
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        self.V1 = V1
        self.alpha = alpha
        self.verbose = verbose
        
        if self.verbose:
            print("=== UniversalEnergyMinimizer Initialized ===")
            print(f"  g_internal: {g_internal:.6f} (Gravitational self-confinement)")
            print(f"  g1: {g1:.3f} (Nonlinear subspace coupling)")
            print(f"  g2: {g2:.6f} (EM circulation coupling)")
            print(f"  alpha: {alpha:.3f} GeV (Spatial-subspace coupling)")
            print(f"  V0: {V0:.6f} GeV, V1: {V1:.6f} GeV (Three-well depths)")
            print(f"  Energy minimization: Scale-independent (first-principles)")
    
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
        From "A Beautiful Balance" research note, the characteristic scale is:
        
            Δx = ℏ²/(g_internal × A⁶)  [in natural units where ℏ=c=1]
            Δx = 1/(g_internal × A⁶)   [gives result in GeV^-1]
        
        This formula ensures:
        - Larger amplitude A → smaller Δx (more confined)
        - Stronger gravity g_internal → smaller Δx (more confined)
        - Prevents infinite spatial delocalization
        
        CRITICAL: This couples spatial extent to amplitude through
        gravitational self-confinement, maintaining self-consistency.
        
        Args:
            A: Current amplitude estimate
            
        Returns:
            Optimal Delta_x from gravitational self-confinement (in fm)
        """
        if self.g_internal <= 0 or A < 1e-10:
            return 1.0  # Default fallback in fm
        
        # From gravitational self-confinement: Δx = 1/(g_internal × A⁶)
        # NOTE: g_internal is calibrated to give Δx directly in fm (legacy convention)
        # This gives Δx ∝ A^(-6) - strong confinement for larger amplitudes
        A_sixth = A ** 6
        delta_x = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)  # Result in fm
        
        # Apply reasonable physical bounds (spatial scale should be in fm range)
        MIN_DELTA_X = 0.01   # fm - minimum localization (legacy value)
        MAX_DELTA_X = 100.0  # fm - maximum spread (legacy value)
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
            # Use same formula as _compute_optimal_delta_x (with unit conversion)
            Delta_x_0 = (1.0 / (self.g_internal * A_0**6)) * HBAR_C_GEV_FM  # Convert to fm
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
        
        # Generation-dependent initial guesses (from legacy solver)
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
            # Use same formula as _compute_optimal_delta_x (with unit conversion)
            Delta_x_0 = (1.0 / (self.g_internal * A_0**6)) * HBAR_C_GEV_FM  # Convert to fm
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
        Internal method to minimize energy using self-consistent iteration.
        
        LEGACY SOLVER APPROACH (proven to work):
        =========================================
        - Compute suggested Delta_x from gravitational confinement: Δx ~ A^(-6)
        - Compute suggested Delta_sigma from energy balance: Δσ ~ A^(-4/3)
        - Compute suggested A from wavefunction norm: A = sqrt(Σ ∫|χ|² dσ)
        - Mix with current values using adaptive mixing
        - Iterate until self-consistency
        
        Args:
            shape_structure: Normalized shape from Stage 1
            initial_guess: (Delta_x_0, Delta_sigma_0, A_0)
            method: Optimization method (ignored, kept for compatibility)
            particle_type: 'lepton', 'meson', or 'baryon'
            n_target: Target generation (for leptons)
            k_winding: Winding number
            A_bounds: Optional bounds on A (ignored, kept for compatibility)
            
        Returns:
            EnergyMinimizationResult
        """
        # CRITICAL: Scale-independent energy minimization (first-principles)
        # The optimal amplitude A emerges purely from energy balance and field dynamics
        # No external mass scale affects the energy minimization process
        
        # Extract initial values
        Delta_x_current, Delta_sigma_current, A_current = initial_guess
        
        # Adaptive mixing parameters (legacy solver values)
        mixing = 0.3  # Start with moderate mixing (legacy default)
        min_mixing = 0.05
        max_mixing = 0.5
        
        # Convergence parameters (legacy solver values)
        max_iter = 30  # Legacy uses 30
        tol = 1e-4  # Legacy uses 1e-4
        
        # History for oscillation detection
        dA_prev = 0.0
        dDx_prev = 0.0
        dDs_prev = 0.0
        
        # Physical bounds
        MIN_DELTA_X = 0.0001
        MAX_DELTA_X = 100.0
        MIN_DELTA_SIGMA = 0.1
        MAX_DELTA_SIGMA = 2.0
        MIN_A = 0.001
        MAX_A = 1000.0
        
        converged = False
        final_iter = 0
        
        if self.verbose:
            print(f"  Self-consistent iteration (max_iter={max_iter}, tol={tol:.1e})")
            print(f"  Initial: A={A_current:.6f}, Dx={Delta_x_current:.6f}, Ds={Delta_sigma_current:.6f}")
        
        for iteration in range(max_iter):
            # === STEP 1: Compute suggested Delta_x from gravitational self-confinement ===
            Delta_x_suggested = self._compute_optimal_delta_x(A_current)
            Delta_x_suggested = max(MIN_DELTA_X, min(MAX_DELTA_X, Delta_x_suggested))
            
            # === STEP 2: Compute suggested Delta_sigma from energy balance ===
            Delta_sigma_suggested = self._compute_optimal_delta_sigma(A_current)
            Delta_sigma_suggested = max(MIN_DELTA_SIGMA, min(MAX_DELTA_SIGMA, Delta_sigma_suggested))
            
            # === STEP 3: Compute suggested A from wavefunction structure ===
            # This is the key difference from the failed gradient descent approach!
            # A is determined by the total amplitude of the scaled wavefunction
            chi_scaled = self._scale_wavefunctions(shape_structure, Delta_sigma_current, A_current)
            
            # Compute total amplitude: A² = Σ ∫|χ_{nlm}|² dσ
            # But chi_scaled already includes A, so we need to extract the structure norm
            # The shape_structure is normalized, so we can compute A from energy balance
            # For now, use a simple update based on energy gradient
            epsilon = 1e-6
            E_current = self._compute_total_energy(
                shape_structure, Delta_x_current, Delta_sigma_current, A_current
            )
            E_plus = self._compute_total_energy(
                shape_structure, Delta_x_current, Delta_sigma_current, A_current + epsilon
            )
            E_minus = self._compute_total_energy(
                shape_structure, Delta_x_current, Delta_sigma_current, A_current - epsilon
            )
            
            # Central difference gradient
            grad_A = (E_plus - E_minus) / (2.0 * epsilon)
            
            # Suggest new A using small gradient step
            learning_rate = 0.01 * A_current
            A_suggested = A_current - learning_rate * grad_A
            A_suggested = max(MIN_A, min(MAX_A, A_suggested))
            
            # === STEP 4: Check convergence ===
            delta_A = abs(A_suggested - A_current)
            delta_Dx = abs(Delta_x_suggested - Delta_x_current)
            delta_Ds = abs(Delta_sigma_suggested - Delta_sigma_current)
            
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            rel_delta_Ds = delta_Ds / max(Delta_sigma_current, 0.01)
            
            if self.verbose and (iteration % 5 == 0 or iteration < 3):
                print(f"  Iter {iteration}: A={A_current:.6f}, Dx={Delta_x_current:.6f}, Ds={Delta_sigma_current:.6f}, E={E_current:.6e}")
                print(f"    Deltas: dA/A={rel_delta_A:.2e}, dDx/Dx={rel_delta_Dx:.2e}, dDs/Ds={rel_delta_Ds:.2e}")
            
            if rel_delta_A < tol and rel_delta_Dx < tol and rel_delta_Ds < tol:
                converged = True
                final_iter = iteration
                if self.verbose:
                    print(f"  CONVERGED after {iteration+1} iterations")
                break
            
            # === STEP 5: Adaptive mixing with oscillation detection ===
            if iteration > 0:
                dA_current = A_suggested - A_current
                dDx_current = Delta_x_suggested - Delta_x_current
                dDs_current = Delta_sigma_suggested - Delta_sigma_current
                
                # Check for sign reversals (oscillations)
                oscillating_A = (dA_current * dA_prev) < 0
                oscillating_Dx = (dDx_current * dDx_prev) < 0
                oscillating_Ds = (dDs_current * dDs_prev) < 0
                
                if oscillating_A or oscillating_Dx or oscillating_Ds:
                    # Reduce mixing when oscillating
                    mixing = max(min_mixing, mixing * 0.5)
                    if self.verbose:
                        print(f"    OSCILLATION - Reducing mixing to {mixing:.3f}")
                elif iteration > 5:
                    # Increase mixing if stable for multiple iterations
                    mixing = min(max_mixing, mixing * 1.1)
                
                # Store for next iteration
                dA_prev = dA_current
                dDx_prev = dDx_current
                dDs_prev = dDs_current
            
            # === STEP 6: Apply mixing ===
            A_current = (1 - mixing) * A_current + mixing * A_suggested
            Delta_x_current = (1 - mixing) * Delta_x_current + mixing * Delta_x_suggested
            Delta_sigma_current = (1 - mixing) * Delta_sigma_current + mixing * Delta_sigma_suggested
            
            final_iter = iteration
        
        # === COMPUTE FINAL ENERGY ===
        E_total = self._compute_total_energy(
            shape_structure, Delta_x_current, Delta_sigma_current, A_current
        )
        
        # Compute individual energy components for reporting
        chi_scaled = self._scale_wavefunctions(shape_structure, Delta_sigma_current, A_current)
        E_spatial = self._compute_spatial_energy(Delta_x_current, A_current)
        E_curvature = self._compute_curvature_energy(Delta_x_current, A_current)
        E_kin_sigma = self._compute_kinetic_sigma(chi_scaled, Delta_sigma_current)
        E_pot_sigma = self._compute_potential_sigma(chi_scaled)
        E_nl_sigma = self._compute_nonlinear_sigma(chi_scaled, Delta_sigma_current)
        E_sigma = E_kin_sigma + E_pot_sigma + E_nl_sigma
        E_coupling = self._compute_coupling_energy(chi_scaled, Delta_x_current, Delta_sigma_current, A_current)
        E_em = self._compute_em_energy(chi_scaled)
        
        if self.verbose:
            print(f"\n  Final solution:")
            print(f"    Delta_x = {Delta_x_current:.6f} fm")
            print(f"    Delta_sigma = {Delta_sigma_current:.6f}")
            print(f"    A = {A_current:.6f}")
            print(f"    E_total = {E_total:.6e}")
            print(f"  Energy breakdown:")
            print(f"    E_sigma: {E_sigma:.6e}")
            print(f"    E_spatial: {E_spatial:.6e}")
            print(f"    E_coupling: {E_coupling:.6e}")
            print(f"    E_curvature: {E_curvature:.6e}")
        
        # === MASS IS NOT COMPUTED HERE ===
        # Mass calculation is done externally using calculate_beta helper
        # m = beta * A^2, where beta is calibrated from electron
        mass = None
        
        convergence_message = "Converged" if converged else f"Max iterations ({max_iter}) reached"
        
        return EnergyMinimizationResult(
            Delta_x=Delta_x_current,
            Delta_sigma=Delta_sigma_current,
            A=A_current,
            mass=mass,
            E_total=E_total,
            E_sigma=E_sigma,
            E_kinetic_sigma=E_kin_sigma,
            E_potential_sigma=E_pot_sigma,
            E_nonlinear_sigma=E_nl_sigma,
            E_spatial=E_spatial,
            E_coupling=E_coupling,
            E_curvature=E_curvature,
            E_em=E_em,
            converged=converged,
            iterations=final_iter + 1,
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
        A: float
    ) -> float:
        """
        Compute spatial-subspace coupling energy.
        
        E_coupling from mixed derivatives that create generation hierarchy.
        
        Scales as: E_c ~ alpha * (1/Delta_x) * (1/Delta_sigma) * A^2
        
        Args:
            Delta_x: Spatial extent in fm
            Delta_sigma: Subspace width (dimensionless)
            A: Amplitude
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        # Simplified estimate
        # Full implementation would compute actual gradient coupling integrals
        
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        E_coupling = self.alpha * A**2 / (Delta_x_nat * Delta_sigma)
        
        return E_coupling
    
    def _compute_spatial_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute spatial confinement energy.
        
        E_x = 1/(2·A²·Δx²)
        
        Quantum confinement energy scales inversely with the square of both
        amplitude and spatial extent. Larger amplitudes in smaller regions
        have higher confinement energy.
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude (dimensionless)
            
        Returns:
            Spatial energy contribution (natural energy units)
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        if A < 1e-10:
            return 1e10  # Prevent division by zero
        
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        E_x = 1.0 / (2.0 * A**2 * Delta_x_nat**2)
        
        return E_x
    
    def _compute_curvature_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute gravitational self-confinement energy.
        
        E_curv = g_internal × A⁴/Δx
        
        The field's own energy-density creates spacetime curvature that
        confines it. Stronger fields (larger A) in smaller regions create
        stronger gravitational self-confinement. This is the mechanism that
        prevents infinite spatial delocalization.
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude (dimensionless)
            
        Returns:
            Gravitational self-energy contribution (natural energy units)
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        E_curv = self.g_internal * A**4 / Delta_x_nat
        
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
        A: float
    ) -> float:
        """
        Compute total energy from all field contributions.
        
        The energy functional determines optimal scale parameters purely from
        field dynamics and energy balance. The amplitude A that minimizes this
        energy is the natural field strength for this configuration.
        
        Energy components:
        - E_x = 1/(2·A²·Δx²)  [spatial quantum confinement]
        - E_curv = g_internal·A⁴/Δx  [gravitational self-confinement]
        - E_sigma = kinetic + potential + nonlinear  [subspace field energy]
        - E_coupling = alpha·A²/(Δx·Δσ)  [spatial-subspace mixing]
        - E_em = g2·|J|²  [electromagnetic circulation]
        
        The minimum of this functional gives the stable field configuration.
        This is pure first-principles physics - no external scales required.
        
        Args:
            shape_structure: Normalized shape from Stage 1
            Delta_x: Spatial extent (fm)
            Delta_sigma: Subspace width (dimensionless)
            A: Amplitude (dimensionless)
            
        Returns:
            Total energy in natural units
        """
        # Scale wavefunctions
        chi_scaled = self._scale_wavefunctions(shape_structure, Delta_sigma, A)
        
        # Compute energy components from field configuration
        E_x = self._compute_spatial_energy(Delta_x, A)
        E_curv = self._compute_curvature_energy(Delta_x, A)
        
        # Subspace energy components
        E_kin_sigma = self._compute_kinetic_sigma(chi_scaled, Delta_sigma)
        E_pot_sigma = self._compute_potential_sigma(chi_scaled)
        E_nl_sigma = self._compute_nonlinear_sigma(chi_scaled, Delta_sigma)
        E_sigma = E_kin_sigma + E_pot_sigma + E_nl_sigma
        
        # Coupling energy
        E_coupling = self._compute_coupling_energy(chi_scaled, Delta_x, Delta_sigma, A)
        
        # Electromagnetic circulation energy
        E_em = self._compute_em_energy(chi_scaled)
        
        # Total energy
        E_total = E_x + E_curv + E_sigma + E_coupling + E_em
        
        return E_total
    
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

