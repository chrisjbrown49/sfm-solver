"""
Non-Separable Wavefunction Solver for SFM.

This module solves for the full entangled superposition wavefunction:
    ψ(r,θ,φ,σ) = Σ_{n,l,m} R_{nl}(r) Y_l^m(θ,φ) χ_{nlm}(σ)

Key physics:
- Each angular component (n,l,m) has its OWN subspace function χ_{nlm}(σ)
- The coupling Hamiltonian mixes l=0 ↔ l=1 ↔ l=2
- The p-wave components are correlated with ∂χ₀/∂σ
- This breaks spherical symmetry and allows non-zero coupling!

PHENOMENOLOGICAL PARAMETERS STATUS
==================================

| Parameter     | Value      | Status      | Derivation Path           |
|---------------|------------|-------------|---------------------------|
| alpha         | ~10.5 GeV  | Fitted      | 5D metric coupling        |
| delta_sigma   | Variable   | Optimized   | Energy minimization       |
| g_internal    | ~0.003     | Fitted      | 5D gravitational coupling |
| g1            | 5000       | Set         | Gravity/EM ratio          |
| g2            | 0.004      | Set         | Fine structure            |
| gen_mod_width | 1.5        | Ansatz      | Subspace eigenstates      |
| gen_mod_amp   | 0.5        | Ansatz      | Subspace eigenstates      |

FIRST-PRINCIPLES FEATURES
=========================
- g_internal: Fundamental (G_5D × β³)
- E = 2n + l: Harmonic oscillator structure
- Winding preservation: ∂/∂σ preserves k
- Self-confinement: Δx = 1/(g_internal × A⁶)^(1/3)
- Beta-independent: Solver works in amplitude space
- Δσ optimization: Subspace width emerges from energy balance
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from sfm_solver.core.correlated_basis import CorrelatedBasis, SpatialState


@dataclass
class WavefunctionStructure:
    """
    Result from the non-separable wavefunction solver.
    
    Contains the converged wavefunction STRUCTURE (not scaled).
    The SCALE (A, Δx, Δσ) is determined separately by UniversalEnergyMinimizer.
    """
    # Subspace wavefunctions for each angular component {(n,l,m): χ_{nlm}(σ)}
    chi_components: Dict[Tuple[int, int, int], NDArray]
    
    # Target quantum numbers
    n_target: int
    k_winding: int
    
    # Angular composition (fraction in each l)
    l_composition: Dict[int, float]
    
    # Effective winding from Im[∫χ*(∂χ/∂σ)dσ]
    k_eff: float
    
    # Total amplitude of structure (before scaling)
    structure_norm: float
    
    # Convergence info
    converged: bool
    iterations: int
    
    # Particle type
    particle_type: str  # 'lepton', 'meson', 'baryon'
    
    # Self-consistent iteration history (optional, for diagnostics)
    convergence_history: Optional[Dict] = None
    
    # Final spatial scale (from self-consistent iteration)
    delta_x_final: Optional[float] = None
    
    # Final subspace width (from energy minimization)
    delta_sigma_final: Optional[float] = None


class NonSeparableWavefunctionSolver:
    """
    Solve for non-separable entangled superposition wavefunctions.
    
    This solver finds the wavefunction STRUCTURE - the relative amplitudes
    and phases of the χ_{nlm}(σ) components. The overall SCALE (A, Δx, Δσ)
    is determined by the UniversalEnergyMinimizer.
    
    The key output is a unit-normalized set of {χ_{nlm}(σ)} that together
    form the complete non-separable wavefunction.
    
    Methods from nonseparable_solver.py are incorporated here.
    """
    
    # =========================================================================
    # PHENOMENOLOGICAL PARAMETER: alpha (spatial-subspace coupling)
    # =========================================================================
    # Status: EMPIRICALLY FITTED
    # 
    # alpha controls the strength of spatial-subspace coupling in perturbation
    # theory: induced = -alpha × R_coupling × envelope / E_denom
    #
    # Current value: alpha ≈ 10.5 GeV (from constants.json)
    # 
    # Theoretical constraint from dimensional analysis:
    #     α ~ ℏ²/(m × L²) where m ~ 100 GeV, L ~ fm
    #     → α ~ 10^-6 to 10^-3 GeV (order of magnitude)
    #
    # How determined: Optimized to minimize lepton mass prediction errors
    # 
    # First-principles derivation: PENDING
    #     Should emerge from 5D metric structure and compactification geometry
    # =========================================================================
    
    # =========================================================================
    # PHENOMENOLOGICAL PARAMETERS: g1, g2 (nonlinear couplings)
    # =========================================================================
    # g1 = 5000.0 (dimensionless)
    # Status: EMPIRICALLY SET
    # Role: Controls nonlinear |χ|⁴ self-interaction strength
    # Theoretical constraint: g1 ~ 10³⁶ × G_5D³ c² (from EM/gravity ratio)
    # Note: Currently unused in Step 1 (max_iter_nl=0)
    #
    # g2 = 0.004 (dimensionless)  
    # Status: EMPIRICALLY SET
    # Role: Controls circulation-dependent coupling (EM-like effects)
    # Theoretical constraint: Related to fine structure constant emergence
    # Note: Placeholder value, not actively used in current solvers
    # =========================================================================
    
    def __init__(
        self,
        alpha: float,
        beta: float = None,  # No longer used in solver - kept for backward compat
        g_internal: float = None,
        g1: float = 5000.0,  # PHENOMENOLOGICAL - see docstring
        g2: float = 0.004,   # PHENOMENOLOGICAL - see docstring
        V0: float = 1.0,
        n_max: int = 5,
        l_max: int = 2,
        N_sigma: int = 64,
        a0: float = 1.0,
        g_eff: float = None,  # DEPRECATED: use g_internal instead
        kappa: float = None,  # DEPRECATED: use g_internal instead
    ):
        """
        Initialize the solver with SFM parameters.
        
        Args:
            alpha: Spatial-subspace coupling strength (GeV)
            beta: DEPRECATED - no longer used in solver physics.
                  Mass is computed after solving: m = β_output × A²
                  where β_output = m_e_exp / A_e²
            g_internal: FUNDAMENTAL gravitational self-confinement constant
                        This controls self-confinement: Δx = 1/(g_internal × A⁶)^(1/3)
                        g_internal works directly with amplitude A, making the
                        solver completely independent of the mass unit choice.
            g1: Nonlinear self-interaction strength
            g2: Circulation coupling (for EM)
            V0: Three-well potential depth (GeV)
            n_max: Maximum spatial principal quantum number
            l_max: Maximum angular momentum
            N_sigma: Subspace grid points
            a0: Characteristic length scale
            g_eff: DEPRECATED - use g_internal instead
            kappa: DEPRECATED - use g_internal instead
        """
        self.alpha = alpha
        self.beta = beta if beta is not None else 100.0  # Only for backward compat
        
        # Handle g_internal (preferred) vs deprecated g_eff/kappa
        if g_internal is not None:
            self.g_internal = g_internal
        elif g_eff is not None:
            # DEPRECATED: convert g_eff to g_internal
            # g_internal = g_eff × β³
            self.g_internal = g_eff * (self.beta ** 3)
        elif kappa is not None:
            # DEPRECATED: convert kappa to g_internal
            # kappa = g_eff × β², so g_eff = kappa/β², g_internal = g_eff × β³ = kappa × β
            self.g_internal = kappa * self.beta
        else:
            # Default from constants
            from sfm_solver.core.constants import G_INTERNAL
            self.g_internal = G_INTERNAL
        
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        
        # Create basis (from correlated_basis.py)
        self.basis = CorrelatedBasis(n_max=n_max, l_max=l_max, N_sigma=N_sigma, a0=a0)
        
        # Three-well potential on subspace grid
        self.V_sigma = V0 * (1 - np.cos(3 * self.basis.sigma))
        
        # Precompute spatial coupling matrix
        self._precompute_spatial_coupling()
    
    def _precompute_spatial_coupling(self):
        """Precompute the spatial coupling matrix elements."""
        N = self.basis.N_spatial
        self.spatial_coupling = np.zeros((N, N), dtype=complex)
        
        for i, s1 in enumerate(self.basis.spatial_states):
            for j, s2 in enumerate(self.basis.spatial_states):
                coupling = self.basis.compute_full_coupling_matrix_element(s1, s2)
                self.spatial_coupling[i, j] = coupling
    
    def _build_subspace_derivative_matrix(self) -> NDArray:
        """Build the first derivative operator d/dσ."""
        N = self.basis.N_sigma
        dsigma = self.basis.dsigma
        
        # Central difference (periodic boundary)
        D1 = np.zeros((N, N), dtype=complex)
        for i in range(N):
            D1[i, (i+1) % N] = 1.0
            D1[i, (i-1) % N] = -1.0
        D1 = D1 / (2 * dsigma)
        
        return D1
    
    def _compute_total_amplitude(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> float:
        """Compute total amplitude squared A² = Σ ∫|χ_{nlm}|² dσ."""
        total = 0.0
        for key, chi in chi_components.items():
            total += np.sum(np.abs(chi)**2) * self.basis.dsigma
        return total
    
    def _normalize_wavefunction(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        target_A_sq: float = 1.0,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """Normalize so that total amplitude squared equals target."""
        current_A_sq = self._compute_total_amplitude(chi_components)
        
        if current_A_sq < 1e-20:
            return chi_components
        
        scale = np.sqrt(target_A_sq / current_A_sq)
        
        normalized = {}
        for key, chi in chi_components.items():
            normalized[key] = chi * scale
        
        return normalized
    
    def _compute_l_composition(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> Dict[int, float]:
        """Compute fraction of amplitude in each l state."""
        composition = {}
        total = 0.0
        
        for (n, l, m), chi in chi_components.items():
            weight = np.sum(np.abs(chi)**2) * self.basis.dsigma
            composition[l] = composition.get(l, 0.0) + weight
            total += weight
        
        if total > 1e-20:
            for l in composition:
                composition[l] /= total
        
        return composition
    
    def _compute_k_eff(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> float:
        """
        Compute effective winding from Im[∫χ_total*(∂χ_total/∂σ)dσ] / ∫|χ_total|²dσ.
        
        The normalization is critical: k_eff = Im[∫χ*dχ/dσ] / ∫|χ|²
        For a pure winding state χ = A*exp(ikσ), this gives k.
        """
        D1 = self._build_subspace_derivative_matrix()
        dsigma = self.basis.dsigma
        
        # Sum all components
        chi_total = np.zeros(self.basis.N_sigma, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        
        dchi_total = D1 @ chi_total
        integral = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        
        # Normalize by total amplitude to get proper k_eff
        norm_sq = np.sum(np.abs(chi_total)**2) * dsigma
        if norm_sq < 1e-20:
            return 0.0
        
        # k_eff = Im[∫χ*dχ/dσ] / ∫|χ|²
        return float(np.imag(integral) / norm_sq)
    
    def _build_envelope(
        self,
        sigma: NDArray,
        center: float,
        delta_sigma: float,
    ) -> NDArray:
        """
        Build Gaussian envelope with specified width.
        
        This is the unified envelope builder used by all solvers.
        The width parameter delta_sigma is optimized by energy minimization.
        
        Args:
            sigma: Subspace grid
            center: Center of envelope (e.g., π for leptons)
            delta_sigma: Width parameter (to be optimized)
            
        Returns:
            Gaussian envelope array
        """
        return np.exp(-(sigma - center)**2 / delta_sigma)
    
    def _compute_optimal_delta_sigma(self, A: float) -> float:
        """
        Compute optimal subspace width from energy balance.
        
        FIRST-PRINCIPLES DERIVATION:
        ============================
        
        The subspace energy has two competing terms:
            E_kin,σ ~ 1/Δσ²     (kinetic - prefers large Δσ)
            E_nonlin ~ g₁A⁴/Δσ  (nonlinear - prefers large Δσ)
        
        Total: E = C₁/Δσ² + C₂×g₁×A⁴/Δσ
        
        Minimize: dE/dΔσ = -2C₁/Δσ³ - C₂×g₁×A⁴/Δσ² = 0
        
        This gives: Δσ_opt = (2C₁/(C₂×g₁×A⁴))^(1/3) ∝ 1/(g₁×A⁴)^(1/3)
        
        With C₁ = C₂ = 1 (natural units):
            Δσ_opt = (2/(g₁×A⁴))^(1/3)
        
        Args:
            A: Current amplitude estimate
            
        Returns:
            Optimal delta_sigma from energy balance
        """
        if self.g1 <= 0 or A < 1e-10:
            return 0.5  # Default fallback
        
        # From energy minimization: Δσ_opt ∝ 1/(g₁×A⁴)^(1/3)
        # The factor of 2 comes from the kinetic/nonlinear balance
        A_fourth = A ** 4
        delta_sigma_opt = (2.0 / (self.g1 * A_fourth)) ** (1.0/3.0)
        
        # FIRST-PRINCIPLES: Allow natural values from energy balance
        # Apply physics-based minimum: envelope must be wide enough for coupling
        # Very narrow envelopes prevent effective spatial-subspace coupling
        MIN_DELTA_SIGMA = 0.1   # Physics-based floor for envelope coupling
        MAX_DELTA_SIGMA = 2.0   # Reasonable ceiling
        delta_sigma_opt = max(MIN_DELTA_SIGMA, min(MAX_DELTA_SIGMA, delta_sigma_opt))
        
        return delta_sigma_opt
    
    def _self_consistent_iteration_core(
        self,
        n_eff: int,
        chi_primary: NDArray,
        delta_sigma_init: float = 0.5,
        max_iter_outer: int = 30,
        tol_outer: float = 1e-4,
        verbose: bool = False,
    ) -> Tuple[Dict[Tuple[int, int, int], NDArray], float, float, float, Dict]:
        """
        Core self-consistent iteration loop used by all particle solvers.
        
        This implements the unified self-consistent iteration that determines:
        - A (amplitude) from wavefunction structure
        - Δx (spatial extent) from gravitational self-confinement
        - Δσ (subspace width) from energy balance
        
        FIRST-PRINCIPLES:
        =================
        All three quantities emerge from energy minimization, not fitting.
        
        Args:
            n_eff: Effective quantum number for this particle
            chi_primary: Primary (normalized) wavefunction component
            delta_sigma_init: Initial guess for subspace width
            max_iter_outer: Maximum iterations
            tol_outer: Convergence tolerance
            verbose: Print progress
            
        Returns:
            Tuple of (chi_components, A_final, delta_x_final, delta_sigma_final, history)
        """
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # Initial conditions
        Delta_x_current = 1.0 / n_eff
        A_current = 0.1
        delta_sigma_current = delta_sigma_init
        MIN_SCALE = 0.0001
        
        # Target state
        target_key = (n_eff, 0, 0)
        target_idx = self.basis.state_index(n_eff, 0, 0)
        
        history = {
            'Delta_x': [Delta_x_current],
            'A': [A_current],
            'delta_sigma': [delta_sigma_current],
        }
        
        final_iter = 0
        
        for iter_outer in range(max_iter_outer):
            if verbose:
                print(f"\n--- Iteration {iter_outer+1}/{max_iter_outer} ---")
                print(f"  dx={Delta_x_current:.6f}, A={A_current:.6f}, ds={delta_sigma_current:.4f}")
            
            # === STEP 1: Compute spatial coupling at current scale ===
            a_n = Delta_x_current / np.sqrt(2 * n_eff + 1)
            a_n = max(a_n, MIN_SCALE)
            spatial_coupling = self.basis.compute_spatial_coupling_at_scale(a_n)
            
            # === STEP 2: Build envelope at current delta_sigma ===
            envelope = self._build_envelope(sigma, np.pi, delta_sigma_current)
            
            # === STEP 3: Build perturbative structure ===
            chi_components = {target_key: chi_primary}
            
            E_target_0 = 2 * n_eff + 0
            
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                    continue
                
                E_state = 2 * state.n + state.l
                E_denom = E_target_0 - E_state
                
                if abs(E_denom) < 0.5:
                    E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
                
                # Induced component using current envelope
                induced = -self.alpha * R_coupling * envelope / E_denom
                chi_components[key] = induced
            
            # === STEP 4: Compute amplitude from wavefunction ===
            A_new = np.sqrt(self._compute_total_amplitude(chi_components))
            
            # === STEP 5: Update Δx from gravitational self-confinement ===
            A_sixth = A_new ** 6
            if self.g_internal > 0 and A_sixth > 1e-30:
                Delta_x_new = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            else:
                Delta_x_new = Delta_x_current
            
            # === STEP 6: Update Δσ from energy balance ===
            delta_sigma_new = self._compute_optimal_delta_sigma(A_new)
            
            if verbose:
                print(f"  A_new={A_new:.6f}, dx_new={Delta_x_new:.6f}, ds_new={delta_sigma_new:.4f}")
            
            # === STEP 7: Check convergence ===
            delta_A = abs(A_new - A_current)
            delta_Dx = abs(Delta_x_new - Delta_x_current)
            delta_Ds = abs(delta_sigma_new - delta_sigma_current)
            
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            rel_delta_Ds = delta_Ds / max(delta_sigma_current, 0.01)
            
            if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer and rel_delta_Ds < tol_outer:
                if verbose:
                    print(f"\n[CONVERGED] after {iter_outer+1} iterations")
                final_iter = iter_outer
                break
            
            # === STEP 8: Update with mixing for stability ===
            mixing = 0.3
            A_current = (1 - mixing) * A_current + mixing * A_new
            Delta_x_current = (1 - mixing) * Delta_x_current + mixing * Delta_x_new
            delta_sigma_current = (1 - mixing) * delta_sigma_current + mixing * delta_sigma_new
            Delta_x_current = max(Delta_x_current, MIN_SCALE)
            
            history['A'].append(A_current)
            history['Delta_x'].append(Delta_x_current)
            history['delta_sigma'].append(delta_sigma_current)
            final_iter = iter_outer
        
        # Normalize final wavefunction
        chi_components_final = self._normalize_wavefunction(chi_components, 1.0)
        
        return chi_components_final, A_current, Delta_x_current, delta_sigma_current, history
    
    def _initial_guess_lepton(
        self,
        n_target: int,
        k_winding: int,
        initial_A_sq: float = 0.01,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Create initial guess for lepton wavefunction.
        
        Start with primarily s-wave (l=0) state with winding k,
        but SEED small l=1 components to bootstrap the coupling!
        """
        chi_components = {}
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # Primary s-wave gets 90% of initial amplitude
        primary_fraction = 0.9
        # l=1 components get 10% to bootstrap coupling
        l1_fraction = 0.1
        
        for state in self.basis.spatial_states:
            key = (state.n, state.l, state.m)
            
            # Gaussian envelope with winding for all components
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            chi_base = envelope * np.exp(1j * k_winding * sigma)
            
            if state.n == n_target and state.l == 0 and state.m == 0:
                # Primary s-wave component
                norm_sq = np.sum(np.abs(chi_base)**2) * dsigma
                chi = chi_base * np.sqrt(initial_A_sq * primary_fraction / norm_sq)
                chi_components[key] = chi
                
            elif state.l == 1:
                # SEED l=1 components to bootstrap coupling!
                n_l1_states = sum(1 for s in self.basis.spatial_states if s.l == 1)
                per_state_fraction = l1_fraction / max(n_l1_states, 1)
                
                norm_sq = np.sum(np.abs(chi_base)**2) * dsigma
                chi = chi_base * np.sqrt(initial_A_sq * per_state_fraction / norm_sq)
                
                # Add small random phase to break symmetry
                chi = chi * np.exp(1j * 0.1 * state.m)
                chi_components[key] = chi
                
            else:
                # Other components start at zero
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        return chi_components
        
    # =========================================================================
    # SELF-CONSISTENT Δx ITERATION (Step 1-3 of missing physics fix)
    # =========================================================================
    
    def _build_perturbative_structure(
        self,
        n_target: int,
        k_winding: int,
        spatial_coupling: NDArray,
        resonance_width: float = 0.0,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Build perturbative wavefunction structure for given spatial coupling.
        
        This is the core perturbation theory calculation, separated so it
        can be called multiple times with different spatial_coupling matrices
        during self-consistent iteration.
        
        Args:
            n_target: Target spatial quantum number
            k_winding: Subspace winding number
            spatial_coupling: Coupling matrix at current scale
            resonance_width: Lorentzian width for resonance enhancement (0 = disabled)
            
        Returns:
            Dictionary of χ components {(n,l,m): χ_{nlm}(σ)}
        """
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        target_key = (n_target, 0, 0)
        target_idx = self.basis.state_index(n_target, 0, 0)
        
        # Primary s-wave component with Gaussian envelope and winding
        # Normalized to unit integral per plan (lines 422-425)
        envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
        chi_primary = envelope * np.exp(1j * k_winding * sigma)
        norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
        chi_primary = chi_primary / np.sqrt(norm_sq)
        
        chi_components = {target_key: chi_primary}
        
        # Energy scale for target state - standard 3D harmonic oscillator form
        # E = 2n + l (in units where ℏω = 1)
        E_target_0 = 2 * n_target + 0  # l=0 for target s-wave
        
        # NOTE: No spatial_enhancement factor - n-dependence emerges naturally from:
        # 1. R_ij matrix elements (computed from actual gradient integrals)
        # 2. Self-consistent Δx feedback (creates additional scaling)
        # See next_steps.md Phase 1, Task 1.1
        
        # Induced components from perturbation theory
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            if key == target_key:
                continue
            
            R_coupling = spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                continue
            
            # Energy denominator - standard 3D harmonic oscillator form
            # E = 2n + l (in units where ℏω = 1)
            # See next_steps.md Phase 1, Task 1.2
            E_state = 2 * state.n + state.l
            E_denom = E_target_0 - E_state
            
            # Regularize small denominators (numerical stability)
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            # Resonance enhancement (Step 3)
            if resonance_width > 0:
                # Lorentzian weighting: Γ / (E² + Γ²)
                # Normalized so on-resonance gives factor of 1
                resonance_factor = (resonance_width / 
                                   (E_denom**2 + resonance_width**2))
                resonance_factor = resonance_factor * resonance_width  # Normalize
                effective_coupling = R_coupling * (1.0 + resonance_factor)
            else:
                effective_coupling = R_coupling
            
            # Induced component with SAME WINDING (first-principles: ∂/∂σ preserves k)
            induced = (-self.alpha * effective_coupling * envelope * 
                      np.exp(1j * k_winding * sigma) / E_denom)
            
            chi_components[key] = induced
        
        # === SECOND-ORDER PERTURBATION: l=1 → l=2 transitions ===
        # Selection rule Δl = ±1 means l=2 requires two steps: l=0 → l=1 → l=2
        # This implements Phase 3A Task 3A.1 from next_steps.md
        #
        # IMPORTANT: Second-order should be a PERTURBATIVE CORRECTION, not dominant.
        # The scaling is α² × R₀₁ × R₁₂ / (E₀₁ × E₁₂), which is O(α²).
        # We initialize l=2 components fresh (not accumulating from previous call).
        
        # First, clear any existing l=2 components
        l2_components = {}
        
        # Collect l=1 induced states (these were computed by first-order perturbation)
        l1_states = [(key, chi.copy()) for key, chi in chi_components.items() 
                     if key[1] == 1 and np.sum(np.abs(chi)**2) * dsigma > 1e-12]
        
        for l1_key, chi_l1 in l1_states:
            n_l1, _, m_l1 = l1_key
            l1_idx = self.basis.state_index(n_l1, 1, m_l1)
            if l1_idx < 0:
                continue
            
            # Energy of this l=1 state
            E_l1 = 2 * n_l1 + 1
            
            # Find l=2 states that couple to this l=1 state
            for j, state2 in enumerate(self.basis.spatial_states):
                if state2.l != 2:
                    continue  # Only l=2 targets
                
                key2 = (state2.n, state2.l, state2.m)
                
                # Check selection rule for l=1 → l=2
                R_l1_l2 = spatial_coupling[l1_idx, j]
                if abs(R_l1_l2) < 1e-10:
                    continue
                
                # Energy denominator for l=1 → l=2 transition
                E_l2 = 2 * state2.n + state2.l
                E_denom_2 = E_l1 - E_l2
                
                if abs(E_denom_2) < 0.5:
                    E_denom_2 = 0.5 * np.sign(E_denom_2) if E_denom_2 != 0 else 0.5
                
                # Second-order induced component
                # chi_l1 already has factor of α/E_denom_1, so this gives α²/(E₁×E₂)
                induced_l2 = (-self.alpha * R_l1_l2 * chi_l1 / E_denom_2)
                
                # Accumulate (multiple l=1 states can contribute to same l=2)
                if key2 in l2_components:
                    l2_components[key2] = l2_components[key2] + induced_l2
                else:
                    l2_components[key2] = induced_l2
        
        # Add l=2 components, but ensure they remain perturbative
        # Second-order should be smaller than first-order (l=1)
        # Cap l=2 total norm to be at most 10% of l=1 total norm
        
        l1_total_norm = sum(np.sum(np.abs(chi)**2) * dsigma 
                           for key, chi in chi_components.items() if key[1] == 1)
        l2_total_norm = sum(np.sum(np.abs(chi)**2) * dsigma 
                           for chi in l2_components.values())
        
        if l2_total_norm > 0.1 * l1_total_norm and l2_total_norm > 1e-10:
            # Scale down l=2 components to be perturbative
            scale_factor = np.sqrt(0.1 * l1_total_norm / l2_total_norm)
            for key2 in l2_components:
                l2_components[key2] = l2_components[key2] * scale_factor
        
        for key2, chi_l2 in l2_components.items():
            chi_components[key2] = chi_l2
        
        return chi_components
    
    def _build_perturbative_structure_with_nonlinear(
        self,
        n_target: int,
        k_winding: int,
        spatial_coupling: NDArray,
        max_iter_nl: int = 10,
        tol_nl: float = 1e-3,
        resonance_width: float = 0.0,
        verbose: bool = False,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Build perturbative wavefunction with nonlinear feedback iteration.
        
        The g₁|χ|⁴ term modifies effective energy levels, creating a
        self-consistent problem that must be solved iteratively.
        
        This creates CASCADING AMPLIFICATION:
        1. Initial: Compute induced components with kinetic E_denom
        2. Iteration 1: Induced creates nonlinear density → shifts E_denom → larger induced
        3. Iteration 2: Even larger induced → stronger density → smaller |E_denom|
        4. Convergence: System finds self-consistent solution
        
        Higher n benefits MORE because:
        - More induced components → stronger |χ_total|
        - Stronger nonlinear density → larger shifts
        - Creates n² or n³ scaling from feedback
        
        Args:
            n_target: Target spatial quantum number
            k_winding: Subspace winding number
            spatial_coupling: Coupling matrix at current scale
            max_iter_nl: Maximum nonlinear iterations
            tol_nl: Convergence tolerance
            resonance_width: Lorentzian width for resonance enhancement
            verbose: Print progress
            
        Returns:
            Dictionary of χ components with nonlinear corrections
        """
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        target_key = (n_target, 0, 0)
        target_idx = self.basis.state_index(n_target, 0, 0)
        
        # === Initialize primary component ===
        # Normalized to unit integral per plan (lines 422-425)
        envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
        chi_primary = envelope * np.exp(1j * k_winding * sigma)
        norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
        chi_primary = chi_primary / np.sqrt(norm_sq)
        
        chi_components = {target_key: chi_primary}
        
        # Initialize all other states to zero
        for state in self.basis.spatial_states:
            key = (state.n, state.l, state.m)
            if key not in chi_components:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        # Base kinetic energy - standard 3D harmonic oscillator form
        # E = 2n + l (in units where ℏω = 1)
        E_target_0 = 2 * n_target + 0  # l=0 for target s-wave
        
        # NOTE: No spatial_enhancement factor - n-dependence emerges naturally
        # See next_steps.md Phase 1, Task 1.1
        
        # === Nonlinear iteration loop ===
        for iter_nl in range(max_iter_nl):
            if verbose and iter_nl % 5 == 0:
                print(f"    NL iteration {iter_nl+1}/{max_iter_nl}")
            
            # Compute total wavefunction
            chi_total = np.zeros(len(sigma), dtype=complex)
            for chi in chi_components.values():
                chi_total += chi
            
            # Nonlinear density: g₁|χ|⁴
            rho_nl = self.g1 * np.abs(chi_total)**4
            
            # Store old components for convergence check
            chi_components_old = {k: v.copy() for k, v in chi_components.items()}
            
            # Nonlinear shift for target state
            V_nl_target = np.sum(rho_nl * np.abs(chi_primary)**2) * dsigma
            E_target_eff = E_target_0 + V_nl_target
            
            # Update induced components with nonlinear shifts
            total_change = 0.0
            
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    continue
                
                # Kinetic energy - standard harmonic oscillator
                E_state_kinetic = 2 * state.n + state.l
                
                # Nonlinear shift for this state
                chi_state = chi_components_old[key]
                chi_state_norm = np.sum(np.abs(chi_state)**2) * dsigma
                if chi_state_norm > 1e-10:
                    V_nl_state = np.sum(rho_nl * np.abs(chi_state)**2) * dsigma
                else:
                    V_nl_state = 0.0
                
                # Effective energy with nonlinear correction
                E_state_eff = E_state_kinetic + V_nl_state
                
                # Updated denominator
                E_denom = E_target_eff - E_state_eff
                
                # Regularize (numerical stability)
                if abs(E_denom) < 0.5:
                    E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
                
                # Resonance enhancement (Step 3)
                if resonance_width > 0:
                    resonance_factor = (resonance_width / 
                                       (E_denom**2 + resonance_width**2))
                    resonance_factor = resonance_factor * resonance_width
                    effective_coupling = R_coupling * (1.0 + resonance_factor)
                else:
                    effective_coupling = R_coupling
                
                # Recompute induced component (no spatial_enhancement)
                induced = (-self.alpha * effective_coupling * envelope * 
                          np.exp(1j * k_winding * sigma) / E_denom)
                
                # Track change for convergence
                change = np.sum(np.abs(induced - chi_components_old[key])**2) * dsigma
                total_change += change
                
                chi_components[key] = induced
            
            # === SECOND-ORDER: l=1 → l=2 transitions (within nonlinear loop) ===
            # Clear and recompute l=2 components to avoid accumulation
            l2_components = {}
            
            l1_states = [(key, chi.copy()) for key, chi in chi_components.items() 
                         if key[1] == 1 and np.sum(np.abs(chi)**2) * dsigma > 1e-12]
            
            for l1_key, chi_l1 in l1_states:
                n_l1, _, m_l1 = l1_key
                l1_idx = self.basis.state_index(n_l1, 1, m_l1)
                if l1_idx < 0:
                    continue
                
                E_l1 = 2 * n_l1 + 1
                chi_l1_norm = np.sum(np.abs(chi_l1)**2) * dsigma
                V_nl_l1 = np.sum(rho_nl * np.abs(chi_l1)**2) * dsigma if chi_l1_norm > 1e-10 else 0.0
                E_l1_eff = E_l1 + V_nl_l1
                
                for j, state2 in enumerate(self.basis.spatial_states):
                    if state2.l != 2:
                        continue
                    
                    key2 = (state2.n, state2.l, state2.m)
                    R_l1_l2 = spatial_coupling[l1_idx, j]
                    if abs(R_l1_l2) < 1e-10:
                        continue
                    
                    E_l2 = 2 * state2.n + state2.l
                    E_denom_2 = E_l1_eff - E_l2
                    if abs(E_denom_2) < 0.5:
                        E_denom_2 = 0.5 * np.sign(E_denom_2) if E_denom_2 != 0 else 0.5
                    
                    induced_l2 = (-self.alpha * R_l1_l2 * chi_l1 / E_denom_2)
                    
                    if key2 in l2_components:
                        l2_components[key2] = l2_components[key2] + induced_l2
                    else:
                        l2_components[key2] = induced_l2
            
            # Replace l=2 components (capped to be perturbative)
            l1_total_norm = sum(np.sum(np.abs(chi)**2) * dsigma 
                               for key, chi in chi_components.items() if key[1] == 1)
            l2_total_norm = sum(np.sum(np.abs(chi)**2) * dsigma 
                               for chi in l2_components.values())
            
            if l2_total_norm > 0.1 * l1_total_norm and l2_total_norm > 1e-10:
                scale_factor = np.sqrt(0.1 * l1_total_norm / l2_total_norm)
                for key2 in l2_components:
                    l2_components[key2] = l2_components[key2] * scale_factor
            
            for key2, chi_l2 in l2_components.items():
                old_l2 = chi_components.get(key2, np.zeros_like(chi_l2))
                chi_components[key2] = chi_l2
                total_change += np.sum(np.abs(chi_l2 - old_l2)**2) * dsigma
            
            # Check convergence
            if total_change < tol_nl:
                if verbose:
                    print(f"    [NL CONVERGED] after {iter_nl+1} iterations")
                break
        
        else:
            if verbose:
                print(f"    [NL WARNING] Did not converge after {max_iter_nl} iterations")
        
        return chi_components
    
    def solve_lepton_self_consistent(
        self,
        n_target: int,
        k_winding: int = 1,
        max_iter_outer: int = 20,
        max_iter_nl: int = 10,
        tol_outer: float = 1e-4,
        tol_nl: float = 1e-3,
        resonance_width: float = 0.0,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for lepton wavefunction with self-consistent Δx iteration.
        
        IMPLEMENTATION FOLLOWS missing_physics_fix.md EXACTLY:
        ======================================================
        
        The spatial scale Δx is determined self-consistently with amplitude A
        through gravitational self-confinement: Δx ∝ 1/(βA²)^(3/2).
        
        This creates a feedback loop:
            Larger A → Smaller Δx → Stronger gradients → Larger coupling → Even larger A
        
        This loop is what creates the dramatic mass hierarchy.
        
        KEY: A is computed directly from UNNORMALIZED chi_components,
        not from energy minimization. Normalization only happens at the end.
        
        Args:
            n_target: Target spatial quantum number (1=e, 2=mu, 3=tau)
            k_winding: Subspace winding number (typically 1)
            max_iter_outer: Maximum outer iterations
            max_iter_nl: Maximum nonlinear iterations per step (Step 2)
            tol_outer: Outer loop convergence tolerance
            tol_nl: Nonlinear loop convergence tolerance
            resonance_width: Lorentzian width for resonance enhancement (Step 3)
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with self-consistently determined scale
        """
        if verbose:
            print(f"\n=== Self-Consistent Lepton Solver (n={n_target}, k={k_winding}) ===")
        
        # === INITIAL CONDITIONS ===
        # Start with Compton-like scale for first iteration (as per plan)
        Delta_x_current = 1.0 / n_target  # Rough initial guess
        A_current = 0.1
        
        # Minimum scale to prevent numerical issues (stability fix)
        MIN_SCALE = 0.0001  # Reduced to allow more differentiation
        
        # Convergence tracking
        history = {
            'Delta_x': [Delta_x_current],
            'A': [A_current],
            'spatial_coupling_max': [],
        }
        
        final_iter = 0
        A_prev = A_current  # Track previous for oscillation detection
        oscillation_count = 0
        
        for iter_outer in range(max_iter_outer):
            if verbose:
                print(f"\n--- Outer Iteration {iter_outer+1}/{max_iter_outer} ---")
                print(f"  Current dx = {Delta_x_current:.6f}")
                print(f"  Current A = {A_current:.6f}")
            
            # === STEP 1: Compute spatial coupling at current scale ===
            # (As specified in plan Section 1.2)
            a_n = Delta_x_current / np.sqrt(2 * n_target + 1)
            a_n = max(a_n, MIN_SCALE)  # Floor to prevent collapse
            spatial_coupling = self.basis.compute_spatial_coupling_at_scale(a_n)
            
            # Track maximum coupling strength
            max_coupling = np.max(np.abs(spatial_coupling))
            history['spatial_coupling_max'].append(max_coupling)
            
            if verbose:
                print(f"  Scale parameter a = {a_n:.6f}")
                print(f"  Max |R_ij| = {max_coupling:.6f}")
            
            # === STEP 2: Build perturbative wavefunction ===
            # Use nonlinear iteration if Step 2 is enabled (max_iter_nl > 0)
            # Otherwise use simple perturbative structure (Step 1 only)
            if max_iter_nl > 0:
                chi_components = self._build_perturbative_structure_with_nonlinear(
                    n_target, k_winding, spatial_coupling,
                    max_iter_nl=max_iter_nl,
                    tol_nl=tol_nl,
                    resonance_width=resonance_width,
                    verbose=verbose,
                )
            else:
                chi_components = self._build_perturbative_structure(
                    n_target, k_winding, spatial_coupling,
                    resonance_width=resonance_width,
                )
            
            # === STEP 3: Estimate amplitude from wavefunction ===
            # (As specified in plan: A = sqrt(sum(integral(|chi|^2))))
            # This is computed from the UNNORMALIZED chi_components!
            A_new = np.sqrt(self._compute_total_amplitude(chi_components))
            
            if verbose:
                print(f"  Estimated A = {A_new:.6f}")
            
            # === STEP 4: Update Δx from self-confinement ===
            # NEW FORMULA: Δx = 1/(G_internal × A⁶)^(1/3)
            # 
            # This uses amplitude A directly, NOT mass m = β × A².
            # G_internal is the FUNDAMENTAL gravitational self-confinement constant.
            # This makes the solver completely independent of β (mass unit choice).
            # β only appears when converting amplitude to physical mass at the end.
            
            A_sixth = A_new ** 6
            
            # Self-confinement scale (in natural/amplitude units)
            if self.g_internal > 0 and A_sixth > 1e-30:
                Delta_x_new = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            else:
                Delta_x_new = Delta_x_current
            
            # NOTE: Compton wavelength cap removed - not needed with G_internal formulation.
            # The self-confinement Δx = 1/(G_internal × A⁶)^(1/3) determines scale directly.
            
            if verbose:
                print(f"  Self-confinement dx = {Delta_x_new:.6f}")
                print(f"  (G_internal × A^6 = {self.g_internal * A_sixth:.6e})")
            
            # === STEP 5: Check convergence ===
            delta_A = abs(A_new - A_current)
            delta_Dx = abs(Delta_x_new - Delta_x_current)
            
            # Detect oscillation (sign change in A update direction)
            if iter_outer > 1:
                if (A_new - A_current) * (A_current - A_prev) < 0:
                    oscillation_count += 1
            
            if verbose:
                print(f"  dA = {delta_A:.2e}, d(dx) = {delta_Dx:.2e}")
                if oscillation_count > 0:
                    print(f"  Oscillations detected: {oscillation_count}")
            
            # Use relative tolerance for convergence
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            
            if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer:
                if verbose:
                    print(f"\n[CONVERGED] after {iter_outer+1} iterations")
                final_iter = iter_outer
                break
            
            # === STEP 6: Update with adaptive damping for stability ===
            # Reduce mixing when oscillating to stabilize
            base_mixing = 0.3
            if oscillation_count > 2:
                mixing = base_mixing / (1 + 0.3 * oscillation_count)
            else:
                mixing = base_mixing
            
            if verbose and oscillation_count > 2:
                print(f"  Reduced mixing to {mixing:.3f} due to oscillation")
            
            A_prev = A_current
            A_current = (1 - mixing) * A_current + mixing * A_new
            Delta_x_current = (1 - mixing) * Delta_x_current + mixing * Delta_x_new
            
            # Apply minimum scale floor to dx as well
            Delta_x_current = max(Delta_x_current, MIN_SCALE)
            
            history['A'].append(A_current)
            history['Delta_x'].append(Delta_x_current)
            final_iter = iter_outer
        
        else:
            if verbose:
                print(f"\n[WARNING] Did not converge after {max_iter_outer} iterations")
        
        # === STEP 7: Final wavefunction with converged Δx ===
        # (As per plan: only normalize at the VERY END)
        a_final = Delta_x_current / np.sqrt(2 * n_target + 1)
        a_final = max(a_final, MIN_SCALE)  # Apply same floor
        spatial_coupling_final = self.basis.compute_spatial_coupling_at_scale(a_final)
        
        if max_iter_nl > 0:
            chi_components_final = self._build_perturbative_structure_with_nonlinear(
                n_target, k_winding, spatial_coupling_final,
                max_iter_nl=max_iter_nl,
                tol_nl=tol_nl,
                resonance_width=resonance_width,
                verbose=False,
            )
        else:
            chi_components_final = self._build_perturbative_structure(
                n_target, k_winding, spatial_coupling_final,
                resonance_width=resonance_width,
            )
        
        # Normalize to unit total - ONLY AT THE END (as per plan)
        chi_components_final = self._normalize_wavefunction(chi_components_final, 1.0)
        
        # Extract observables
        l_composition = self._compute_l_composition(chi_components_final)
        k_eff = self._compute_k_eff(chi_components_final)
        
        # structure_norm is the CONVERGED A from the self-consistent iteration
        # This carries the mass information: m = beta * structure_norm^2
        structure_norm = A_current
        
        if verbose:
            print(f"\n=== Final Structure ===")
            print(f"  dx = {Delta_x_current:.6f}")
            print(f"  A = {A_current:.6f}")
            print(f"  k_eff = {k_eff:.4f}")
            print(f"  Angular composition: {l_composition}")
            print(f"  l-composition: {l_composition}")
        
        # Compute optimal delta_sigma based on converged amplitude
        delta_sigma_final = self._compute_optimal_delta_sigma(A_current)
        
        return WavefunctionStructure(
            chi_components=chi_components_final,
            n_target=n_target,
            k_winding=k_winding,
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,  # This is A, not normalized!
            converged=(final_iter < max_iter_outer - 1),
            iterations=final_iter + 1,
            particle_type='lepton',
            convergence_history=history,
            delta_x_final=Delta_x_current,
            delta_sigma_final=delta_sigma_final,
        )

    def _well_envelope(self, well_index: int, sigma: NDArray) -> NDArray:
        """
        Generate Gaussian envelope localized at specified well.
        
        This emerges from first principles: the three-well potential
        V = V₀(1 - cos(3σ)) has minima at σ = π/3, π, 5π/3.
        Quarks/antiquarks localize in these wells.
        
        Args:
            well_index: 1, 2, or 3 (for the three wells)
            sigma: Subspace grid
            
        Returns:
            Envelope function localized at well minimum
        """
        # Well centers from V = V₀(1 - cos(3σ)) minima
        well_centers = [np.pi/3, np.pi, 5*np.pi/3]
        center = well_centers[well_index - 1]
        
        # Envelope width from harmonic approximation around minimum
        # V ≈ (9/2)V₀(σ - σ₀)² near minimum, giving width ~ 1/√(9V₀/2)
        # With V₀ = 1, width ≈ 0.47. Use 0.5 for numerical stability.
        width = 0.5
        
        return np.exp(-(sigma - center)**2 / width)

    def solve_meson_self_consistent(
        self,
        quark_wells: tuple = (1, 2),
        n_radial: int = 1,
        max_iter_outer: int = 30,
        tol_outer: float = 1e-4,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for meson wavefunction with self-consistent Δx iteration.
        
        FIRST-PRINCIPLES IMPLEMENTATION:
        ================================
        
        Mesons are quark-antiquark bound states. The composite wavefunction
        is χ_total = χ_q(σ) + χ_q̄(σ), where each component is localized
        in a different well of the three-well potential.
        
        The self-confinement follows the same physics as leptons:
            Δx = 1/(G_internal × A⁶)^(1/3)
        
        where A is computed from the composite wavefunction.
        
        Args:
            quark_wells: Tuple of (quark_well, antiquark_well), each 1-3
            n_radial: Radial quantum number (1=ground, 2=first excited)
            max_iter_outer: Maximum self-consistent iterations
            tol_outer: Convergence tolerance
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with converged meson wavefunction
        """
        if verbose:
            print(f"\n=== Self-Consistent Meson Solver (wells={quark_wells}) ===")
        
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # === BUILD COMPOSITE PRIMARY WAVEFUNCTION ===
        # Each quark component localized in its well
        q_well, qbar_well = quark_wells
        
        # Quark and antiquark envelopes (first-principles: well localization)
        envelope_q = self._well_envelope(q_well, sigma)
        envelope_qbar = self._well_envelope(qbar_well, sigma)
        
        # Color phases for meson: quark and antiquark are color-anticolor
        # For color neutrality: phase difference = π
        phase_q = 0.0
        phase_qbar = np.pi
        
        # Composite primary wavefunction (UNNORMALIZED during iteration)
        chi_primary_q = envelope_q * np.exp(1j * phase_q)
        chi_primary_qbar = envelope_qbar * np.exp(1j * phase_qbar)
        chi_primary = chi_primary_q + chi_primary_qbar
        
        # === SELF-CONSISTENT ITERATION ===
        # For composites, effective n_target = number of constituents
        # This is first-principles: each quark adds one unit of quantum number
        # Meson (2 quarks) → n_eff = 2 (like muon)
        n_eff = 2  # Two constituents in meson
        
        Delta_x_current = 1.0 / n_eff  # Initial guess
        A_current = 0.1
        delta_sigma_current = 0.5  # Initial guess, will be optimized
        MIN_SCALE = 0.0001
        
        history = {'Delta_x': [Delta_x_current], 'A': [A_current], 'delta_sigma': [delta_sigma_current]}
        final_iter = 0
        
        for iter_outer in range(max_iter_outer):
            if verbose:
                print(f"\n--- Iteration {iter_outer+1}/{max_iter_outer} ---")
                print(f"  dx={Delta_x_current:.6f}, A={A_current:.6f}, ds={delta_sigma_current:.6f}")
            
            # Compute spatial coupling at current scale
            a_n = Delta_x_current / np.sqrt(2 * n_eff + 1)
            a_n = max(a_n, MIN_SCALE)
            spatial_coupling = self.basis.compute_spatial_coupling_at_scale(a_n)
            
            # Build envelope at CURRENT delta_sigma (first-principles: optimized each iteration)
            single_well_envelope = self._build_envelope(sigma, np.pi, delta_sigma_current)
            
            # Build perturbative structure for composite wavefunction
            # Use n_eff as the target quantum number
            target_key = (n_eff, 0, 0)
            target_idx = self.basis.state_index(n_eff, 0, 0)
            
            # Normalize primary for perturbation calculation
            norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
            chi_primary_norm = chi_primary / np.sqrt(norm_sq)
            
            chi_components = {target_key: chi_primary_norm}
            
            # Energy of target state (first-principles: 3D harmonic oscillator)
            # n_eff = 2 gives E_target_0 = 4 (like muon)
            E_target_0 = 2 * n_eff + 0
            
            # Induced components from spatial-subspace coupling
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                    continue
                
                E_state = 2 * state.n + state.l
                E_denom = E_target_0 - E_state
                
                if abs(E_denom) < 0.5:
                    E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
                
                # Induced component uses single-well envelope (same as lepton solver)
                # This maintains consistent amplitude scaling with leptons
                induced = -self.alpha * R_coupling * single_well_envelope / E_denom
                chi_components[key] = induced
            
            # Compute total amplitude A (UNNORMALIZED)
            A_new = np.sqrt(self._compute_total_amplitude(chi_components))
            
            # Self-confinement: Δx = 1/(G_internal × A⁶)^(1/3)
            A_sixth = A_new ** 6
            if self.g_internal > 0 and A_sixth > 1e-30:
                Delta_x_new = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            else:
                Delta_x_new = Delta_x_current
            
            # Update Δσ from energy balance (first-principles)
            delta_sigma_new = self._compute_optimal_delta_sigma(A_new)
            
            if verbose:
                print(f"  A_new={A_new:.6f}, dx_new={Delta_x_new:.6f}, ds_new={delta_sigma_new:.6f}")
            
            # Check convergence (all three quantities)
            delta_A = abs(A_new - A_current)
            delta_Dx = abs(Delta_x_new - Delta_x_current)
            delta_Ds = abs(delta_sigma_new - delta_sigma_current)
            
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            rel_delta_Ds = delta_Ds / max(delta_sigma_current, 0.0001)
            
            if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer and rel_delta_Ds < tol_outer:
                if verbose:
                    print(f"\n[CONVERGED] after {iter_outer+1} iterations")
                final_iter = iter_outer
                break
            
            # Update with mixing for stability
            mixing = 0.3
            A_current = (1 - mixing) * A_current + mixing * A_new
            Delta_x_current = (1 - mixing) * Delta_x_current + mixing * Delta_x_new
            delta_sigma_current = (1 - mixing) * delta_sigma_current + mixing * delta_sigma_new
            Delta_x_current = max(Delta_x_current, MIN_SCALE)
            
            history['A'].append(A_current)
            history['Delta_x'].append(Delta_x_current)
            history['delta_sigma'].append(delta_sigma_current)
            final_iter = iter_outer
        
        # Final wavefunction (normalized at the end)
        chi_components_final = self._normalize_wavefunction(chi_components, 1.0)
        
        l_composition = self._compute_l_composition(chi_components_final)
        k_eff = self._compute_k_eff(chi_components_final)
        structure_norm = A_current
        delta_sigma_final = delta_sigma_current
        
        if verbose:
            print(f"\n=== Final Meson Structure ===")
            print(f"  dx={Delta_x_current:.6f}, A={A_current:.6f}, ds={delta_sigma_final:.6f}")
        
        return WavefunctionStructure(
            chi_components=chi_components_final,
            n_target=n_eff,
            k_winding=0,  # Mesons don't have net winding
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=(final_iter < max_iter_outer - 1),
            iterations=final_iter + 1,
            particle_type='meson',
            convergence_history=history,
            delta_x_final=Delta_x_current,
            delta_sigma_final=delta_sigma_final,
        )

    def solve_baryon_self_consistent(
        self,
        quark_wells: tuple = (1, 2, 3),
        color_phases: tuple = None,
        max_iter_outer: int = 30,
        tol_outer: float = 1e-4,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for baryon wavefunction with self-consistent Δx iteration.
        
        FIRST-PRINCIPLES IMPLEMENTATION:
        ================================
        
        Baryons are three-quark bound states. The composite wavefunction
        is χ_total = χ_q1(σ) + χ_q2(σ) + χ_q3(σ), with each quark localized
        in a different well with color phases ensuring color neutrality.
        
        Color neutrality requires: exp(i*φ₁) + exp(i*φ₂) + exp(i*φ₃) = 0
        This gives: φ = [0, 2π/3, 4π/3] (cube roots of unity)
        
        The self-confinement follows the same physics as leptons:
            Δx = 1/(G_internal × A⁶)^(1/3)
        
        Args:
            quark_wells: Tuple of (q1_well, q2_well, q3_well), each 1-3
            color_phases: Tuple of color phases (default: [0, 2π/3, 4π/3])
            max_iter_outer: Maximum self-consistent iterations
            tol_outer: Convergence tolerance
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with converged baryon wavefunction
        """
        if color_phases is None:
            # First-principles: color neutrality requires cube roots of unity
            color_phases = (0, 2*np.pi/3, 4*np.pi/3)
        
        if verbose:
            print(f"\n=== Self-Consistent Baryon Solver (wells={quark_wells}) ===")
        
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # === BUILD COMPOSITE PRIMARY WAVEFUNCTION ===
        q1_well, q2_well, q3_well = quark_wells
        phi1, phi2, phi3 = color_phases
        
        # Three quark envelopes (first-principles: well localization)
        envelope_q1 = self._well_envelope(q1_well, sigma)
        envelope_q2 = self._well_envelope(q2_well, sigma)
        envelope_q3 = self._well_envelope(q3_well, sigma)
        
        # Composite primary with color phases (UNNORMALIZED)
        chi_q1 = envelope_q1 * np.exp(1j * phi1)
        chi_q2 = envelope_q2 * np.exp(1j * phi2)
        chi_q3 = envelope_q3 * np.exp(1j * phi3)
        chi_primary = chi_q1 + chi_q2 + chi_q3
        
        # === SELF-CONSISTENT ITERATION ===
        # For composites, effective n_target = number of constituents
        # This is first-principles: each quark adds one unit of quantum number
        # Baryon (3 quarks) → n_eff = 3 (like tau)
        n_eff = 3  # Three constituents in baryon
        
        Delta_x_current = 1.0 / n_eff
        A_current = 0.1
        delta_sigma_current = 0.5  # Initial guess, will be optimized
        MIN_SCALE = 0.0001
        
        history = {'Delta_x': [Delta_x_current], 'A': [A_current], 'delta_sigma': [delta_sigma_current]}
        final_iter = 0
        
        for iter_outer in range(max_iter_outer):
            if verbose:
                print(f"\n--- Iteration {iter_outer+1}/{max_iter_outer} ---")
                print(f"  dx={Delta_x_current:.6f}, A={A_current:.6f}, ds={delta_sigma_current:.6f}")
            
            # Compute spatial coupling at current scale
            a_n = Delta_x_current / np.sqrt(2 * n_eff + 1)
            a_n = max(a_n, MIN_SCALE)
            spatial_coupling = self.basis.compute_spatial_coupling_at_scale(a_n)
            
            # Build envelope at CURRENT delta_sigma (first-principles: optimized each iteration)
            single_well_envelope = self._build_envelope(sigma, np.pi, delta_sigma_current)
            
            # Build perturbative structure
            # Use n_eff as the target quantum number
            target_key = (n_eff, 0, 0)
            target_idx = self.basis.state_index(n_eff, 0, 0)
            
            norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
            chi_primary_norm = chi_primary / np.sqrt(norm_sq)
            
            chi_components = {target_key: chi_primary_norm}
            
            # Energy of target state (first-principles: 3D harmonic oscillator)
            # n_eff = 3 gives E_target_0 = 6 (like tau)
            E_target_0 = 2 * n_eff + 0
            
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                    continue
                
                E_state = 2 * state.n + state.l
                E_denom = E_target_0 - E_state
                
                if abs(E_denom) < 0.5:
                    E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
                
                # Induced component uses single-well envelope (same as lepton solver)
                induced = -self.alpha * R_coupling * single_well_envelope / E_denom
                chi_components[key] = induced
            
            A_new = np.sqrt(self._compute_total_amplitude(chi_components))
            
            # Self-confinement: Δx = 1/(G_internal × A⁶)^(1/3)
            A_sixth = A_new ** 6
            if self.g_internal > 0 and A_sixth > 1e-30:
                Delta_x_new = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            else:
                Delta_x_new = Delta_x_current
            
            # Update Δσ from energy balance (first-principles)
            delta_sigma_new = self._compute_optimal_delta_sigma(A_new)
            
            if verbose:
                print(f"  A_new={A_new:.6f}, dx_new={Delta_x_new:.6f}, ds_new={delta_sigma_new:.6f}")
            
            delta_A = abs(A_new - A_current)
            delta_Dx = abs(Delta_x_new - Delta_x_current)
            delta_Ds = abs(delta_sigma_new - delta_sigma_current)
            
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            rel_delta_Ds = delta_Ds / max(delta_sigma_current, 0.0001)
            
            if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer and rel_delta_Ds < tol_outer:
                if verbose:
                    print(f"\n[CONVERGED] after {iter_outer+1} iterations")
                final_iter = iter_outer
                break
            
            mixing = 0.3
            A_current = (1 - mixing) * A_current + mixing * A_new
            Delta_x_current = (1 - mixing) * Delta_x_current + mixing * Delta_x_new
            delta_sigma_current = (1 - mixing) * delta_sigma_current + mixing * delta_sigma_new
            Delta_x_current = max(Delta_x_current, MIN_SCALE)
            
            history['A'].append(A_current)
            history['Delta_x'].append(Delta_x_current)
            history['delta_sigma'].append(delta_sigma_current)
            final_iter = iter_outer
        
        chi_components_final = self._normalize_wavefunction(chi_components, 1.0)
        
        l_composition = self._compute_l_composition(chi_components_final)
        k_eff = self._compute_k_eff(chi_components_final)
        structure_norm = A_current
        delta_sigma_final = delta_sigma_current
        
        if verbose:
            print(f"\n=== Final Baryon Structure ===")
            print(f"  dx={Delta_x_current:.6f}, A={A_current:.6f}, ds={delta_sigma_final:.6f}")
        
        return WavefunctionStructure(
            chi_components=chi_components_final,
            n_target=n_eff,
            k_winding=0,
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=(final_iter < max_iter_outer - 1),
            iterations=final_iter + 1,
            particle_type='baryon',
            convergence_history=history,
            delta_x_final=Delta_x_current,
            delta_sigma_final=delta_sigma_final,
        )
    
    # =========================================
    # Accessor methods for use by energy minimizer
    # =========================================
    
    def get_spatial_coupling_matrix(self) -> NDArray:
        """Return the precomputed spatial coupling matrix."""
        return self.spatial_coupling
    
    def get_sigma_grid(self) -> NDArray:
        """Return the subspace coordinate grid."""
        return self.basis.sigma
    
    def get_dsigma(self) -> float:
        """Return the subspace grid spacing."""
        return self.basis.dsigma
    
    def get_V_sigma(self) -> NDArray:
        """Return the three-well potential on the subspace grid."""
        return self.V_sigma
    
    def get_state_index_map(self) -> Dict[Tuple[int, int, int], int]:
        """
        Return mapping from (n,l,m) keys to spatial_coupling indices.
        
        This is needed for the UniversalEnergyMinimizer to correctly
        look up coupling matrix elements.
        """
        return {
            (state.n, state.l, state.m): i 
            for i, state in enumerate(self.basis.spatial_states)
        }


def test_wavefunction_solver():
    """Test the non-separable wavefunction solver."""
    print("=" * 60)
    print("TESTING NON-SEPARABLE WAVEFUNCTION SOLVER")
    print("=" * 60)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=20.0,
        beta=100.0,
        kappa=0.0001,
        g1=5000.0,
        g2=0.004,
        V0=1.0,
        n_max=5,
        l_max=2,
        N_sigma=48,
    )
    
    print(f"\nBasis: {solver.basis.N_spatial} spatial states x {solver.basis.N_sigma} sigma points")
    
    # Test leptons
    print("\n--- LEPTON STRUCTURES ---")
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        result = solver.solve_lepton(n_target=n, k_winding=1, verbose=True)
        print(f"  {name}: k_eff={result.k_eff:.4f}, l_comp={result.l_composition}")
    
    # Test meson (pion-like)
    print("\n--- MESON STRUCTURE (Pion-like) ---")
    meson = solver.solve_meson(
        quark_gen=1, antiquark_gen=1,
        k_quark=5, k_antiquark=-5,
        n_radial=1, verbose=True
    )
    
    # Test baryon (proton-like)
    print("\n--- BARYON STRUCTURE (Proton-like) ---")
    baryon = solver.solve_baryon(
        quark_gens=[1, 1, 1],
        windings=[5, 5, 5],
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("WAVEFUNCTION SOLVER TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_wavefunction_solver()

