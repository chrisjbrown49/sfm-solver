"""
Universal Energy Minimizer for SFM.

This module implements Stage 2 of the two-stage solver architecture.
Takes normalized shape from Stage 1 and finds optimal scale parameters
(Delta_x, Delta_sigma, A) by minimizing total energy.

KEY PRINCIPLE: Stage 2 (physical scales)
    - Input: Normalized shape structure from Stage 1
    - Output: Optimal (Delta_x, Delta_sigma, A) and predicted mass
    - Energy functional: E_total(Delta_x, Delta_sigma, A)
    - Mass prediction: m = beta * A^2

The shape is FIXED (from Stage 1), only the SCALE varies.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
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
    
    # Predicted mass
    mass: Optional[float]  # GeV, None if beta not calibrated
    
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
        beta: Optional[float] = None,
        use_scaled_energy: bool = True,
        verbose: bool = False
    ):
        """
        Initialize universal energy minimizer.
        
        Args:
            g_internal: FUNDAMENTAL gravitational strength (g_internal = G_5D x beta^3)
            g1: Nonlinear coupling (GeV)
            g2: EM coupling (GeV)
            V0: Three-well primary depth (GeV)
            V1: Three-well secondary depth (GeV)
            alpha: Spatial-subspace coupling (GeV)
            beta: Mass conversion factor (m = beta * A^2). If None, will be calibrated.
            use_scaled_energy: If True, minimize E_tilde = beta×E (beta-independent formulation)
            verbose: Print diagnostic information
        """
        self.g_internal = g_internal  # FUNDAMENTAL - drives all physics
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        self.V1 = V1
        self.alpha = alpha
        self.beta = beta  # Can be None initially for calibration
        self.use_scaled_energy = use_scaled_energy
        self.verbose = verbose
        
        if self.verbose:
            print("=== UniversalEnergyMinimizer Initialized ===")
            if beta is not None:
                print(f"  beta: {beta:.6f} GeV")
            else:
                print(f"  beta: Not yet calibrated (will use beta-independent formulation)")
            print(f"  g_internal: {g_internal:.6f} (FUNDAMENTAL)")
            print(f"  g1: {g1:.3f}")
            print(f"  g2: {g2:.6f}")
            print(f"  alpha: {alpha:.3f} GeV")
            print(f"  use_scaled_energy: {use_scaled_energy}")
    
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
        
        # From gravitational self-confinement: Δx = 1/(g_internal × A⁶) [GeV^-1]
        # This gives Δx ∝ A^(-6) - strong confinement for larger amplitudes
        A_sixth = A ** 6
        delta_x_nat = 1.0 / (self.g_internal * A_sixth)  # Result in GeV^-1
        
        # Convert from GeV^-1 to fm
        delta_x_fm = delta_x_nat * HBAR_C_GEV_FM
        
        # Apply reasonable physical bounds (spatial scale should be in fm range)
        MIN_DELTA_X = 0.001  # fm - minimum resolvable scale
        MAX_DELTA_X = 1000.0  # fm - nuclear to atomic scale
        delta_x_opt = max(MIN_DELTA_X, min(MAX_DELTA_X, delta_x_fm))
        
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
            Delta_x_0 = (1.0 / (self.g_internal * A_0**6))**(1/3)
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
        
        # Auto-generate initial guess based on generation
        if initial_guess is None:
            # Leptons have much smaller amplitudes than baryons
            A_scales = {1: 1.0, 2: 5.0, 3: 20.0}  # Rough estimates
            A_0 = A_scales.get(generation_n, 1.0)
            Delta_x_0 = (1.0 / (self.g_internal * A_0**6))**(1/3)
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
            Delta_x_0 = (1.0 / (self.g_internal * A_0**6))**(1/3)
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
        k_winding: Optional[int] = None
    ) -> EnergyMinimizationResult:
        """
        Internal method to minimize energy using self-consistent iteration with mixing.
        
        This mirrors the legacy solver's approach:
        - Compute suggested updates for (A, Delta_x, Delta_sigma) from analytical formulas
        - Mix with current values using adaptive mixing parameter
        - Iterate until self-consistency
        
        Args:
            shape_structure: Normalized shape from Stage 1
            initial_guess: (Delta_x_0, Delta_sigma_0, A_0)
            method: Optimization method (ignored, kept for compatibility)
            particle_type: 'lepton', 'meson', or 'baryon'
            n_target: Target generation (for leptons)
            k_winding: Winding number
            
        Returns:
            EnergyMinimizationResult
        """
        # Choose energy function based on beta availability
        use_scaled = self.use_scaled_energy or self.beta is None
        
        # Extract initial values
        Delta_x_current, Delta_sigma_current, A_current = initial_guess
        
        # Adaptive mixing parameters (following legacy solver)
        mixing = 0.2  # Initial mixing parameter
        min_mixing = 0.05
        max_mixing = 0.5
        
        # Convergence parameters
        max_iter = 200
        tol = 1e-6
        
        # History for oscillation detection
        dA_prev = 0.0
        dDx_prev = 0.0
        dDs_prev = 0.0
        
        # Physical bounds (increased for diagnostic check)
        MIN_DELTA_X = 0.01
        MAX_DELTA_X = 100000.0  # 100 km (3 orders of magnitude increase)
        MIN_DELTA_SIGMA = 0.1
        MAX_DELTA_SIGMA = 200.0  # 2 orders of magnitude increase
        MIN_A = 0.001
        MAX_A = 100000.0  # 2 orders of magnitude increase
        
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
            
            # === STEP 3: Compute energy and gradient for A ===
            # We update A using energy gradient
            epsilon = 1e-6
            
            # Energy at current A
            if use_scaled:
                E_current = self._compute_scaled_total_energy(
                    shape_structure, Delta_x_current, Delta_sigma_current, A_current
                )
                # Energy at A + epsilon
                E_plus = self._compute_scaled_total_energy(
                    shape_structure, Delta_x_current, Delta_sigma_current, A_current + epsilon
                )
            else:
                E_current, _ = self._compute_total_energy(
                    shape_structure, Delta_x_current, Delta_sigma_current, A_current
                )
                E_plus, _ = self._compute_total_energy(
                    shape_structure, Delta_x_current, Delta_sigma_current, A_current + epsilon
                )
            
            # Numerical gradient
            grad_A = (E_plus - E_current) / epsilon
            
            # Suggest new A using gradient descent
            learning_rate = 0.01 * A_current  # Adaptive learning rate
            A_suggested = A_current - learning_rate * grad_A
            A_suggested = max(MIN_A, min(MAX_A, A_suggested))
            
            # === STEP 4: Check convergence ===
            delta_A = abs(A_suggested - A_current)
            delta_Dx = abs(Delta_x_suggested - Delta_x_current)
            delta_Ds = abs(Delta_sigma_suggested - Delta_sigma_current)
            
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            rel_delta_Ds = delta_Ds / max(Delta_sigma_current, 0.01)
            
            if self.verbose and iteration % 20 == 0:
                print(f"  Iter {iteration}: A={A_current:.6f}, Dx={Delta_x_current:.6f}, Ds={Delta_sigma_current:.6f}, E={E_current:.6e}")
                print(f"    Convergence: dA/A={rel_delta_A:.2e}, dDx/Dx={rel_delta_Dx:.2e}, dDs/Ds={rel_delta_Ds:.2e}")
            
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
                    if self.verbose and iteration % 20 == 0:
                        print(f"    OSCILLATION DETECTED - Reducing mixing to {mixing:.3f}")
                elif iteration > 5:
                    # Increase mixing if stable for multiple iterations
                    mixing = min(max_mixing, mixing * 1.05)
                
                # Store for next iteration
                dA_prev = dA_current
                dDx_prev = dDx_current
                dDs_prev = dDs_current
            
            # === STEP 6: Apply mixing ===
            A_current = (1 - mixing) * A_current + mixing * A_suggested
            Delta_x_current = (1 - mixing) * Delta_x_current + mixing * Delta_x_suggested
            Delta_sigma_current = (1 - mixing) * Delta_sigma_current + mixing * Delta_sigma_suggested
            
            final_iter = iteration
        
        # Final energy and components
        if use_scaled:
            E_total = self._compute_scaled_total_energy(
                shape_structure, Delta_x_current, Delta_sigma_current, A_current
            )
            # Create energy components (approximate - not fully broken down in scaled mode)
            energy_components = {
                'E_scaled_total': E_total,
                'note': 'Using beta-independent formulation'
            }
        else:
            E_total, energy_components = self._compute_total_energy(
                shape_structure, Delta_x_current, Delta_sigma_current, A_current
            )
        
        # Compute mass (if beta is available)
        if self.beta is not None:
            mass = self.beta * A_current**2
        else:
            mass = None  # Will be calibrated later
        
        if self.verbose:
            print(f"\n  Final solution:")
            print(f"    Delta_x = {Delta_x_current:.6f} fm")
            print(f"    Delta_sigma = {Delta_sigma_current:.6f}")
            print(f"    A = {A_current:.6f}")
            if mass is not None:
                print(f"    Mass = {mass:.6f} GeV")
            else:
                print(f"    Mass = (will be calibrated from beta)")
            print(f"    E_total = {E_total:.6e}")
            print(f"  Energy breakdown:")
            for name, value in energy_components.items():
                if isinstance(value, (int, float)):
                    print(f"    {name}: {value:.6e}")
                else:
                    print(f"    {name}: {value}")
        
        convergence_message = "Converged" if converged else f"Max iterations ({max_iter}) reached"
        
        return EnergyMinimizationResult(
            Delta_x=Delta_x_current,
            Delta_sigma=Delta_sigma_current,
            A=A_current,
            mass=mass,
            E_total=E_total,
            E_sigma=energy_components.get('E_sigma', 0.0),
            E_kinetic_sigma=energy_components.get('E_kinetic_sigma', 0.0),
            E_potential_sigma=energy_components.get('E_potential_sigma', 0.0),
            E_nonlinear_sigma=energy_components.get('E_nonlinear_sigma', 0.0),
            E_spatial=energy_components.get('E_spatial', 0.0),
            E_coupling=energy_components.get('E_coupling', 0.0),
            E_curvature=energy_components.get('E_curvature', 0.0),
            E_em=energy_components.get('E_em', 0.0),
            converged=converged,
            iterations=final_iter + 1,
            optimization_message=convergence_message,
            particle_type=particle_type,
            n_target=n_target,
            k_winding=k_winding
        )
    
    def _compute_total_energy(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        Delta_x: float,
        Delta_sigma: float,
        A: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total energy for given scale parameters.
        
        E_total = E_sigma + E_spatial + E_coupling + E_curvature + E_em
        
        Args:
            shape_structure: Normalized shape from Stage 1
            Delta_x: Spatial extent
            Delta_sigma: Subspace width
            A: Amplitude
            
        Returns:
            (E_total, energy_components_dict)
        """
        # Scale the wavefunctions
        chi_scaled = self._scale_wavefunctions(shape_structure, Delta_sigma, A)
        
        # 1. Subspace energy
        E_kinetic_sigma = self._compute_kinetic_sigma(chi_scaled, Delta_sigma)
        E_potential_sigma = self._compute_potential_sigma(chi_scaled)
        E_nonlinear_sigma = self._compute_nonlinear_sigma(chi_scaled, Delta_sigma)
        E_sigma = E_kinetic_sigma + E_potential_sigma + E_nonlinear_sigma
        
        # 2. Spatial energy (quantum confinement)
        E_spatial = self._compute_spatial_energy(Delta_x, A)
        
        # 3. Coupling energy (spatial-subspace)
        E_coupling = self._compute_coupling_energy(chi_scaled, Delta_x, Delta_sigma, A)
        
        # 4. Curvature energy (gravitational self-confinement)
        E_curvature = self._compute_curvature_energy(Delta_x, A)
        
        # 5. EM self-energy
        E_em = self._compute_em_energy(chi_scaled)
        
        # Total
        E_total = E_sigma + E_spatial + E_coupling + E_curvature + E_em
        
        energy_components = {
            'E_sigma': E_sigma,
            'E_kinetic_sigma': E_kinetic_sigma,
            'E_potential_sigma': E_potential_sigma,
            'E_nonlinear_sigma': E_nonlinear_sigma,
            'E_spatial': E_spatial,
            'E_coupling': E_coupling,
            'E_curvature': E_curvature,
            'E_em': E_em
        }
        
        return E_total, energy_components
    
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
    
    def _compute_spatial_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute spatial quantum confinement energy.
        
        E_x = hbar^2 / (2m * Delta_x^2) where m = beta * A^2
        
        Using natural units where hbar = c = 1.
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        m = self.beta * A**2
        
        if m < 1e-20:
            return 1e10  # Prevent division by zero
        
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        E_x = 1.0 / (2.0 * m * Delta_x_nat**2)
        
        return E_x
    
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
    
    def _compute_curvature_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute gravitational self-confinement energy.
        
        E_curv = G_5D * m^2 / Delta_x where m = beta * A^2
        
        Using g_internal = G_5D * beta^3, so G_5D = g_internal / beta^3
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        m = self.beta * A**2
        
        G_5D = self.g_internal / (self.beta**3)
        
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        E_curv = G_5D * m**2 / Delta_x_nat
        
        return E_curv
    
    def _compute_em_energy(self, chi_scaled: Dict) -> float:
        """
        Compute electromagnetic self-energy from circulation.
        
        E_EM = beta * g2 * |J|^2
        
        where J = integral chi* d(chi)/dsigma dsigma (circulation)
        """
        chi_total = sum(chi_scaled.values())
        N = len(chi_total)
        dsigma = 2*np.pi / N
        
        D1 = self._build_derivative_operator(N)
        dchi = D1 @ chi_total
        
        J = np.sum(np.conj(chi_total) * dchi) * dsigma
        
        E_em = self.beta * self.g2 * np.abs(J)**2
        
        return E_em
    
    def _compute_scaled_spatial_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute scaled spatial energy: Ẽ_x = β × E_x = 1/(2·A²·Δx²)
        
        This is β-independent! The spatial confinement energy in amplitude space.
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        if A < 1e-10:
            return 1e10  # Prevent division by zero
        
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        E_tilde_x = 1.0 / (2.0 * A**2 * Delta_x_nat**2)
        
        return E_tilde_x
    
    def _compute_scaled_curvature_energy(self, Delta_x: float, A: float) -> float:
        """
        Compute scaled curvature energy: Ẽ_curv = β × E_curv = g_internal × A⁴/Δx
        
        Uses g_internal directly (fundamental parameter).
        This is the gravitational self-confinement in amplitude space.
        
        Args:
            Delta_x: Spatial extent in fm
            A: Amplitude
            
        Note: Delta_x is converted from fm to GeV^-1 for calculation
        """
        # Convert Delta_x from fm to GeV^-1 (natural units)
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        E_tilde_curv = self.g_internal * A**4 / Delta_x_nat
        
        return E_tilde_curv
    
    def _compute_scaled_total_energy(
        self,
        shape_structure: Dict[Tuple[int, int, int], NDArray],
        Delta_x: float,
        Delta_sigma: float,
        A: float
    ) -> float:
        """
        Compute Ẽ_total = β × E_total (β-independent formulation).
        
        This allows energy minimization without knowing β.
        After finding optimal A, calibrate: β = m_exp / A²
        
        The scaled energy uses:
        - Ẽ_x = 1/(2·A²·Δx²)  [no β!]
        - Ẽ_curv = g_internal·A⁴/Δx  [uses g_internal directly]
        - Ẽ_other = β × E_other  [β as placeholder]
        
        The minimum occurs at the same (Δx, Δσ, A) regardless of β value.
        """
        # Use β=1 as placeholder for scaling factors in other energy terms
        beta_placeholder = 1.0
        
        # Scale wavefunctions
        chi_scaled = self._scale_wavefunctions(shape_structure, Delta_sigma, A)
        
        # β-independent energy components
        E_tilde_x = self._compute_scaled_spatial_energy(Delta_x, A)
        E_tilde_curv = self._compute_scaled_curvature_energy(Delta_x, A)
        
        # Other components (multiply by beta_placeholder for consistency)
        E_kin_sigma = self._compute_kinetic_sigma(chi_scaled, Delta_sigma)
        E_pot_sigma = self._compute_potential_sigma(chi_scaled)
        E_nl_sigma = self._compute_nonlinear_sigma(chi_scaled, Delta_sigma)
        E_coupling = self._compute_coupling_energy(chi_scaled, Delta_x, Delta_sigma, A)
        
        # EM energy (compute without beta factor for scaled version)
        chi_total = sum(chi_scaled.values())
        N = len(chi_total)
        dsigma = 2*np.pi / N
        D1 = self._build_derivative_operator(N)
        dchi = D1 @ chi_total
        J = np.sum(np.conj(chi_total) * dchi) * dsigma
        E_em_base = self.g2 * np.abs(J)**2  # Without beta factor
        
        E_tilde_sigma = beta_placeholder * (E_kin_sigma + E_pot_sigma + E_nl_sigma)
        E_tilde_coupling = beta_placeholder * E_coupling
        E_tilde_em = beta_placeholder * E_em_base
        
        # Total scaled energy
        E_tilde_total = (E_tilde_x + E_tilde_curv + E_tilde_sigma + 
                         E_tilde_coupling + E_tilde_em)
        
        return E_tilde_total
    
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

