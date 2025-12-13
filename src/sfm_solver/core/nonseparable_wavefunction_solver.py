"""
Non-Separable Wavefunction Solver for SFM.

This module solves for the full entangled superposition wavefunction:
    ψ(r,θ,φ,σ) = Σ_{n,l,m} R_{nl}(r) Y_l^m(θ,φ) χ_{nlm}(σ)

Key physics:
- Each angular component (n,l,m) has its OWN subspace function χ_{nlm}(σ)
- The coupling Hamiltonian mixes l=0 ↔ l=1 ↔ l=2
- The p-wave components are correlated with ∂χ₀/∂σ
- This breaks spherical symmetry and allows non-zero coupling!

This is Phase 1 of the refactor - extracted from nonseparable_solver.py.
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
    
    def __init__(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        g1: float,
        g2: float,
        V0: float = 1.0,
        n_max: int = 5,
        l_max: int = 2,
        N_sigma: int = 64,
        a0: float = 1.0,
    ):
        """
        Initialize the solver with SFM parameters.
        
        Args:
            alpha: Spatial-subspace coupling strength (GeV)
            beta: Mass coupling constant (GeV)
            kappa: Curvature coupling (GeV⁻²)
            g1: Nonlinear self-interaction strength
            g2: Circulation coupling (for EM)
            V0: Three-well potential depth (GeV)
            n_max: Maximum spatial principal quantum number
            l_max: Maximum angular momentum
            N_sigma: Subspace grid points
            a0: Characteristic length scale
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
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
    
    def solve_lepton(
        self,
        n_target: int,
        k_winding: int = 1,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for lepton wavefunction structure.
        
        This finds the STRUCTURE of the entangled wavefunction (the relative
        amplitudes and phases of χ_{nlm} components). The output is 
        unit-normalized.
        
        The overall SCALE (A, Δx, Δσ) is determined by UniversalEnergyMinimizer.
        
        Uses perturbative approach where:
        1. Primary s-wave carries most amplitude
        2. l=1 components induced by coupling
        3. Effective coupling determines l-mixing
        
        Args:
            n_target: Target spatial quantum number (1=e, 2=μ, 3=τ)
            k_winding: Subspace winding number (typically 1)
            max_iter: Maximum iterations for self-consistency
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with unit-normalized components
        """
        if verbose:
            print(f"Solving lepton wavefunction structure for n={n_target}, k={k_winding}")
        
        D1 = self._build_subspace_derivative_matrix()
        dsigma = self.basis.dsigma
        sigma = self.basis.sigma
        
        target_key = (n_target, 0, 0)
        target_idx = self.basis.state_index(n_target, 0, 0)
        
        # Energy scale for target state - standard 3D harmonic oscillator form
        # E = 2n + l (in units where ℏω = 1)
        E_target_0 = 2 * n_target + 0  # l=0 for target s-wave
        
        # === STEP 1: Compute effective coupling from perturbation theory ===
        effective_coupling = 0.0
        
        for i, state in enumerate(self.basis.spatial_states):
            if state.n == n_target and state.l == 0:
                continue  # Skip same state
            
            R_coupling = self.spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                continue
            
            # Energy denominator - standard harmonic oscillator
            E_state = 2 * state.n + state.l
            E_denom = E_target_0 - E_state
            
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            if abs(E_denom) < 1e-10:
                continue
            
            # Contribution to effective coupling (magnitude)
            effective_coupling += (abs(R_coupling) / abs(E_denom)) ** 2
        
        effective_coupling = np.sqrt(effective_coupling)
        
        if verbose:
            print(f"  Effective spatial coupling: {effective_coupling:.6f}")
        
        # === STEP 2: Build primary s-wave component ===
        # 
        # NO LONGER using hard-coded 90%/10% split!
        # Let the perturbative amplitudes emerge naturally from physics.
        #
        envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
        chi_primary = envelope * np.exp(1j * k_winding * sigma)
        
        # Normalize primary to unit amplitude (will be rescaled at end)
        norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
        chi_primary = chi_primary / np.sqrt(norm_sq)
        
        chi_components = {target_key: chi_primary}
        
        # === STEP 3: Compute induced l≠0 components ===
        # 
        # PHYSICS: The coupling Hamiltonian H = -α ∂²/∂x∂σ acts on the primary state.
        # 
        # CRITICAL: For non-zero Im[∫χ_i*(∂χ_j/∂σ)dσ], we need the proper
        # envelope structure. The winding k is PRESERVED by the ∂/∂σ operator.
        #
        # FIRST-PRINCIPLES: The ∂/∂σ operator is a DERIVATIVE, not a ladder operator.
        # It does NOT change the winding number:
        #   ∂/∂σ [exp(ikσ)] = ik × exp(ikσ)   (SAME winding!)
        #
        # Therefore, induced l=1 states should have the SAME winding as primary.
        #
        # The subspace integral is non-zero even with same winding:
        #   χ_primary = f(σ) × exp(ikσ)
        #   χ_induced = g(σ) × exp(ikσ)
        #   ∂χ_induced/∂σ = [g'(σ) + ik×g(σ)] × exp(ikσ)
        #   Im[∫χ*_primary × ∂χ_induced/∂σ dσ] = k × ⟨f|g⟩ ≠ 0
        #
        # The l-composition should EMERGE from physics:
        #   induced ~ α × R_coupling / E_denom
        
        # Get the primary winding
        # FIRST-PRINCIPLES: Induced states have SAME winding as primary!
        # The ∂/∂σ operator doesn't change winding: ∂/∂σ[exp(ikσ)] = ik×exp(ikσ)
        k_primary = k_winding
        k_induced = k_primary  # SAME winding (first-principles, not k-1!)
        
        # Track total induced amplitude for perturbative validity check
        total_induced_amp_sq = 0.0
        primary_amp_sq = np.sum(np.abs(chi_primary)**2) * dsigma
        
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            if key == target_key:
                continue
            
            R_coupling = self.spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                continue
            
            # Energy denominator - standard harmonic oscillator
            E_state = 2 * state.n + state.l
            E_denom = E_target_0 - E_state
            
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            # Induced component with SAME WINDING (first-principles)
            # No spatial_enhancement - n-dependence emerges from R_ij and self-consistent Δx
            envelope_primary = np.exp(-(sigma - np.pi)**2 / 0.5)
            induced = -self.alpha * R_coupling * envelope_primary * np.exp(1j * k_induced * sigma) / E_denom
            
            # Track amplitude for validity check
            induced_amp_sq = np.sum(np.abs(induced)**2) * dsigma
            total_induced_amp_sq += induced_amp_sq
            
            chi_components[key] = induced
        
        # PERTURBATIVE VALIDITY CHECK:
        # If induced >> primary, perturbation theory breaks down.
        # Cap induced to be at most 50% of primary to maintain validity.
        if total_induced_amp_sq > 0.25 * primary_amp_sq:  # 50% amplitude = 25% amplitude²
            cap_factor = np.sqrt(0.25 * primary_amp_sq / total_induced_amp_sq)
            for key in chi_components:
                if key != target_key:
                    chi_components[key] = chi_components[key] * cap_factor
        
        # Final normalization to unit total - this preserves RELATIVE amplitudes
        # The l-composition will now reflect the true perturbative mixing
        chi_components = self._normalize_wavefunction(chi_components, target_A_sq=1.0)
        
        # Extract observables
        l_composition = self._compute_l_composition(chi_components)
        k_eff = self._compute_k_eff(chi_components)
        structure_norm = self._compute_total_amplitude(chi_components)
        
        if verbose:
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
            print(f"  Effective k: {k_eff:.4f}")
        
        return WavefunctionStructure(
            chi_components=chi_components,
            n_target=n_target,
            k_winding=k_winding,
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=True,  # Perturbative is analytic
            iterations=1,
            particle_type='lepton',
        )
    
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
            # (As specified in plan Section 1.2, lines 318-329)
            # Δx = ℏ² / (G_eff × m³) where m = β × A²
            m_estimate = self.beta * A_new**2
            
            # Self-confinement scale (in natural units where ℏ = c = 1)
            # G_eff ~ kappa / beta² from dimensional analysis
            G_eff = self.kappa / (self.beta**2)
            
            # From plan: Delta_x_new = 1.0 / (G_eff * m_estimate**3)**(1/3)
            if G_eff > 0 and m_estimate > 1e-10:
                Delta_x_new = 1.0 / (G_eff * m_estimate**3)**(1/3)
            else:
                Delta_x_new = Delta_x_current
            
            # NOTE: Compton wavelength cap DISABLED for testing
            # The cap was forcing all particles to MIN_SCALE, destroying differentiation.
            # The self-confinement formula should determine Δx directly.
            # if m_estimate > 1e-10:
            #     lambda_compton = 1.0 / m_estimate
            #     Delta_x_new = min(Delta_x_new, lambda_compton)
            lambda_compton = 1.0 / m_estimate if m_estimate > 1e-10 else 1e10  # for printing only
            
            if verbose:
                print(f"  Estimated mass m = {m_estimate:.6f}")
                print(f"  Self-confinement dx = {Delta_x_new:.6f}")
                if m_estimate > 1e-10:
                    print(f"  Compton wavelength = {1.0/m_estimate:.6f}")
            
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
        )
    
    def solve_meson(
        self,
        quark_gen: int,
        antiquark_gen: int,
        k_quark: int,
        k_antiquark: int,
        n_radial: int = 1,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for meson (quark-antiquark) wavefunction structure.
        
        For mesons, the subspace wavefunction is composite:
            χ_{nlm}(σ) = χ_{nlm,q}(σ) + χ_{nlm,q̄}(σ)
        
        Each spatial component has BOTH quark contributions with interference.
        
        Args:
            quark_gen, antiquark_gen: Quark generations (1=u/d, 2=c/s, 3=b/t)
            k_quark, k_antiquark: Quark windings (typically +5, -5 for color)
            n_radial: Radial excitation (1=1S, 2=2S)
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with unit-normalized composite components
        """
        if verbose:
            print(f"Solving meson wavefunction: q_gen={quark_gen}, qbar_gen={antiquark_gen}")
        
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        D1 = self._build_subspace_derivative_matrix()
        
        # Build individual quark subspace functions
        def quark_subspace(gen: int, k: int) -> NDArray:
            """Generate quark subspace wavefunction with generation structure."""
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            # Generation-dependent radial structure
            if gen > 1:
                x = (sigma - np.pi) / 1.5
                radial_mod = 1.0 + 0.5 * np.cos((gen - 1) * np.pi * x / 2)
            else:
                radial_mod = np.ones_like(sigma)
            return envelope * radial_mod * np.exp(1j * k * sigma)
        
        chi_q = quark_subspace(quark_gen, k_quark)
        chi_qbar = quark_subspace(antiquark_gen, k_antiquark)
        
        # Composite primary (s-wave, n=n_radial)
        chi_composite = chi_q + chi_qbar
        
        # Normalize
        norm_sq = np.sum(np.abs(chi_composite)**2) * dsigma
        chi_composite = chi_composite / np.sqrt(norm_sq)
        
        # Use n_radial as the target spatial mode
        target_key = (n_radial, 0, 0)
        target_idx = self.basis.state_index(n_radial, 0, 0)
        
        # Primary fraction
        primary_fraction = 0.9
        chi_primary = chi_composite * np.sqrt(primary_fraction)
        
        chi_components = {target_key: chi_primary}
        
        # Induced components with SAME WINDING (derivative preserves k)
        # The ∂/∂σ operator doesn't change topological winding number:
        # Energy - standard 3D harmonic oscillator form
        E_target_0 = 2 * n_radial + 0  # l=0 for target s-wave
        
        # Compute average winding of composite
        k_avg = (k_quark + k_antiquark) / 2
        k_induced = k_avg   # Same winding from perturbation theory
        
        induced_total = 0.0
        induced_components = {}
        
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            if key == target_key:
                continue
            
            R_coupling = self.spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                continue
            
            # Energy denominator - standard harmonic oscillator
            E_state = 2 * state.n + state.l
            E_denom = E_target_0 - E_state
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            # Induced component - no spatial_enhancement
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            induced = -self.alpha * abs(R_coupling) * envelope * np.exp(1j * k_induced * sigma) / E_denom
            induced_components[key] = induced
            induced_total += np.sum(np.abs(induced)**2) * dsigma
        
        remaining_fraction = 1.0 - primary_fraction
        if induced_total > 1e-20:
            scale = np.sqrt(remaining_fraction / induced_total)
            for key, induced in induced_components.items():
                chi_components[key] = induced * scale
        else:
            for key in induced_components:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        # Normalize to unit total
        chi_components = self._normalize_wavefunction(chi_components, target_A_sq=1.0)
        
        # Extract observables
        l_composition = self._compute_l_composition(chi_components)
        k_eff = self._compute_k_eff(chi_components)
        structure_norm = self._compute_total_amplitude(chi_components)
        
        # Effective winding for mesons
        net_winding = k_quark + k_antiquark
        
        if verbose:
            print(f"  Net winding (k_q + k_qbar): {net_winding}")
            print(f"  Effective k from wavefunction: {k_eff:.4f}")
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
        
        return WavefunctionStructure(
            chi_components=chi_components,
            n_target=n_radial,
            k_winding=net_winding,
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=True,
            iterations=1,
            particle_type='meson',
        )
    
    def solve_baryon(
        self,
        quark_gens: List[int],
        windings: List[int],
        color_phases: List[float] = None,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for baryon (three-quark) wavefunction structure.
        
        For baryons, the subspace wavefunction is a 3-quark composite:
            χ_{nlm}(σ) = Σ_i χ_{nlm,qi}(σ) × exp(i × color_phase_i)
        
        Color neutrality requires: Σ exp(i × phase_i) = 0
        
        Args:
            quark_gens: List of 3 quark generations
            windings: List of 3 quark windings
            color_phases: Phase offsets for color (default: [0, 2π/3, 4π/3])
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with unit-normalized composite components
        """
        if color_phases is None:
            color_phases = [0, 2*np.pi/3, 4*np.pi/3]
        
        if verbose:
            print(f"Solving baryon wavefunction: gens={quark_gens}, k={windings}")
        
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        D1 = self._build_subspace_derivative_matrix()
        
        # Build individual quark subspace functions
        def quark_subspace(gen: int, k: int) -> NDArray:
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            if gen > 1:
                x = (sigma - np.pi) / 1.5
                radial_mod = 1.0 + 0.5 * np.cos((gen - 1) * np.pi * x / 2)
            else:
                radial_mod = np.ones_like(sigma)
            return envelope * radial_mod * np.exp(1j * k * sigma)
        
        # Three-quark composite with color phases
        chi_composite = np.zeros(self.basis.N_sigma, dtype=complex)
        for gen, k, phase in zip(quark_gens, windings, color_phases):
            chi_q = quark_subspace(gen, k)
            chi_composite += chi_q * np.exp(1j * phase)
        
        # Normalize
        norm_sq = np.sum(np.abs(chi_composite)**2) * dsigma
        if norm_sq > 1e-20:
            chi_composite = chi_composite / np.sqrt(norm_sq)
        
        # Ground state baryons: n=1
        n_target = 1
        target_key = (n_target, 0, 0)
        target_idx = self.basis.state_index(n_target, 0, 0)
        
        primary_fraction = 0.9
        chi_primary = chi_composite * np.sqrt(primary_fraction)
        
        chi_components = {target_key: chi_primary}
        
        # Induced components with SAME WINDING for non-zero coupling
        # Energy - standard 3D harmonic oscillator form
        E_target_0 = 2 * n_target + 0  # l=0 for target s-wave
        
        # Compute average winding of 3-quark composite
        k_avg = sum(windings) / 3
        k_induced = k_avg  # Same winding from perturbation theory
        
        induced_total = 0.0
        induced_components = {}
        
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            if key == target_key:
                continue
            
            R_coupling = self.spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                continue
            
            # Energy denominator - standard harmonic oscillator
            E_state = 2 * state.n + state.l
            E_denom = E_target_0 - E_state
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            # Induced component - no spatial_enhancement
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            induced = -self.alpha * abs(R_coupling) * envelope * np.exp(1j * k_induced * sigma) / E_denom
            induced_components[key] = induced
            induced_total += np.sum(np.abs(induced)**2) * dsigma
        
        remaining_fraction = 1.0 - primary_fraction
        if induced_total > 1e-20:
            scale = np.sqrt(remaining_fraction / induced_total)
            for key, induced in induced_components.items():
                chi_components[key] = induced * scale
        else:
            for key in induced_components:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        # Normalize
        chi_components = self._normalize_wavefunction(chi_components, target_A_sq=1.0)
        
        # Extract observables
        l_composition = self._compute_l_composition(chi_components)
        k_eff = self._compute_k_eff(chi_components)
        structure_norm = self._compute_total_amplitude(chi_components)
        
        # Baryon effective k_coupling (sum of magnitudes)
        k_coupling = sum(abs(k) for k in windings)
        
        if verbose:
            print(f"  k_coupling (sum|k|): {k_coupling}")
            print(f"  Effective k from wavefunction: {k_eff:.4f}")
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
        
        return WavefunctionStructure(
            chi_components=chi_components,
            n_target=n_target,
            k_winding=k_coupling,  # For baryons, use sum of |k|
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=True,
            iterations=1,
            particle_type='baryon',
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

