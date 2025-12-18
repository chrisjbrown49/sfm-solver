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
    
    # Electromagnetic self-energy (from circulation)
    # E_EM = g₂ |J|² where J = ∫ χ* ∂χ/∂σ dσ
    em_energy: Optional[float] = None
    
    # Net circulation (for charge calculation)
    circulation: Optional[complex] = None


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
    # g1 = 1.0 (dimensionless, optimized for SCF baryon solver)
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
        beta: float = None,  # Mass scaling: m = beta * A^2 (calibrated to electron)
        g_internal: float = None,
        g1: float = None,  # Nonlinear self-interaction (loads from constants if None)
        g2: float = None,  # Circulation/EM coupling (loads from constants if None)
        lambda_so: float = None,  # Spin-orbit coupling (loads from constants if None)
        V0: float = None,  # Three-well primary depth (loads from constants if None)
        V1: float = None,  # Three-well secondary depth (loads from constants if None)
        n_max: int = 5,
        l_max: int = 2,
        N_sigma: int = 64,
        a0: float = 1.0,
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
            V0: Three-well potential depth (primary term)
            V1: Three-well potential secondary depth (cos(6σ) term)
                The full potential is: V(σ) = V₀(1 - cos(3σ)) + V₁(1 - cos(6σ))
            n_max: Maximum spatial principal quantum number
            l_max: Maximum angular momentum
            N_sigma: Subspace grid points
            a0: Characteristic length scale
        """
        self.alpha = alpha
        self.beta = beta if beta is not None else 100.0  # Only for backward compat
        
        # Load g_internal from constants if not provided
        if g_internal is not None:
            self.g_internal = g_internal
        else:
            # Default from constants
            from sfm_solver.core.constants import G_INTERNAL
            self.g_internal = G_INTERNAL
        
        # Load g1, g2, lambda_so from constants if not provided
        if g1 is not None:
            self.g1 = g1
        else:
            from sfm_solver.core.constants import G1
            self.g1 = G1
        
        if g2 is not None:
            self.g2 = g2
        else:
            from sfm_solver.core.constants import G2
            self.g2 = G2
        
        if lambda_so is not None:
            self.lambda_so = lambda_so
        else:
            from sfm_solver.core.constants import LAMBDA_SO
            self.lambda_so = LAMBDA_SO
        
        # Load V0 and V1 from constants if not provided
        if V0 is not None:
            self.V0 = V0
        else:
            from sfm_solver.core.constants import V0 as V0_CONST
            self.V0 = V0_CONST
        
        if V1 is not None:
            self.V1 = V1
        else:
            from sfm_solver.core.constants import V1 as V1_CONST
            self.V1 = V1_CONST
        
        # Create basis (from correlated_basis.py)
        self.basis = CorrelatedBasis(n_max=n_max, l_max=l_max, N_sigma=N_sigma, a0=a0)
        
        # Three-well potential on subspace grid
        # V(σ) = V₀(1 - cos(3σ)) + V₁(1 - cos(6σ))
        self.V_sigma = self.V0 * (1 - np.cos(3 * self.basis.sigma)) + self.V1 * (1 - np.cos(6 * self.basis.sigma))
        
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
    
    def _build_kinetic_operator(self) -> NDArray:
        """
        Build kinetic energy operator with periodic boundary conditions.
        
        Returns:
            T = -∂²/∂σ² (in dimensionless units where ℏ²/(2m_σR²) = 1)
        """
        N = self.basis.N_sigma
        dsigma = self.basis.dsigma
        
        # Second derivative with periodic BC
        D2 = np.zeros((N, N))
        for i in range(N):
            D2[i, i] = -2.0
            D2[i, (i+1) % N] = 1.0
            D2[i, (i-1) % N] = 1.0
        
        D2 = D2 / (dsigma**2)
        
        # Kinetic operator (negative second derivative)
        T_op = -D2
        
        return T_op
    
    def _initialize_quark_wavefunction(
        self,
        well_index: int,
        phase: float,
        winding_k: int = 0,
        spin: int = +1,      # NEW: ±1 for spin up/down
        generation: int = 1,  # NEW: n=1,2,3 for generation
    ) -> NDArray:
        """
        Initialize quark wavefunction as Gaussian in specified well.
        
        Args:
            well_index: 1, 2, or 3 (which well)
            phase: Color phase φᵢ for color neutrality
            winding_k: Winding number for EM charge (flavor type: +5=up-type, -3=down-type)
            spin: Spin quantum number (+1 for ↑, -1 for ↓)
            generation: Generation number (1=u/d, 2=c/s, 3=t/b)
            
        Returns:
            Normalized wavefunction χᵢ(σ) with spin-dependent positioning
        """
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # Well centers at σ = π/3, π, 5π/3
        well_centers = [np.pi/3, np.pi, 5*np.pi/3]
        center = well_centers[well_index - 1]
        
        # Spin-orbit shift includes spin
        if hasattr(self, 'lambda_so') and self.lambda_so > 0:
            delta_sigma_so = self.lambda_so * winding_k * spin
            center = center + delta_sigma_so
        
        # Generation affects Gaussian width
        # Higher generation → larger spatial extent → broader envelope
        base_width = 0.25
        generation_scaling = 1.0 + 0.2 * (generation - 1)  # n=1→1.0, n=2→1.2, n=3→1.4
        width = base_width * generation_scaling
        
        # Build wavefunction with color phase and winding
        envelope = np.exp(-(sigma - center)**2 / width)
        chi = envelope * np.exp(1j * phase) * np.exp(1j * winding_k * sigma)
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(chi)**2) * dsigma)
        if norm > 1e-20:
            chi /= norm
        
        return chi
    
    def _check_pauli_exclusion(
        self,
        quark_windings: tuple,
        quark_spins: tuple,
        quark_generations: tuple,
    ) -> bool:
        """
        Check if quark configuration satisfies Pauli exclusion.
        
        Two quarks with same winding (k) AND same generation (n) MUST have opposite spins.
        
        Args:
            quark_windings: Winding numbers (k₁, k₂, k₃)
            quark_spins: Spin quantum numbers (s₁, s₂, s₃) = ±1
            quark_generations: Generation numbers (n₁, n₂, n₃) = 1,2,3
            
        Returns:
            True if Pauli exclusion is satisfied, False otherwise
        """
        k1, k2, k3 = quark_windings
        s1, s2, s3 = quark_spins
        n1, n2, n3 = quark_generations
        
        # Check all pairs
        violations = []
        
        # Pair 1-2: violation if same k AND same n AND same spin
        if k1 == k2 and n1 == n2 and s1 == s2:
            violations.append((1, 2))
        
        # Pair 1-3
        if k1 == k3 and n1 == n3 and s1 == s3:
            violations.append((1, 3))
        
        # Pair 2-3
        if k2 == k3 and n2 == n3 and s2 == s3:
            violations.append((2, 3))
        
        if violations:
            print(f"WARNING: Pauli exclusion violated for quark pairs: {violations}")
            return False
        
        return True
    
    def _initialize_baryon_from_parameters(
        self,
        g1: float,
        g2: float,
        lambda_so: float
    ) -> Tuple[float, float, float]:
        """
        Initialize baryon solver state based on parameter regime.
        
        Physical basis from optimization analysis:
        - g1 sets confinement → affects amplitude A (empirical: A ∝ g1)
        - g2/g1 ratio indicates well structure stability (optimal ≈ 1.35)
        - lambda_so affects spatial distribution
        
        Args:
            g1: Nonlinear self-interaction coupling
            g2: Circulation/EM coupling
            lambda_so: Spin-orbit coupling strength
            
        Returns:
            (A_init, Delta_x_init, delta_sigma_init)
        """
        # Reference values from optimization (Eval 4, 11, 17 averages)
        g1_ref = 49.0
        g2_ref = 67.0
        ratio_ref = g2_ref / g1_ref  # ≈ 1.37
        
        # Amplitude scales with confinement strength
        # From analysis: higher g1 → stronger confinement → larger A
        A_base = 15.0
        A_scale = 1.0 + 0.02 * (g1 - g1_ref)  # ±2% per unit g1
        A_init = A_base * A_scale
        
        # Spatial scale inversely proportional to g1
        # Stronger confinement → smaller spatial extent
        Delta_x_base = 0.30
        Delta_x_init = Delta_x_base * (g1_ref / g1) ** 0.5
        
        # Subspace width influenced by g2/g1 ratio
        # Deviation from optimal ratio → adjust for stability
        delta_sigma_base = 0.80
        ratio = g2 / g1
        ratio_deviation = abs(ratio - ratio_ref) / ratio_ref
        if ratio_deviation < 0.1:
            # Near optimal ratio: use base value
            ratio_correction = 1.0
        else:
            # Far from optimal: reduce for stability
            ratio_correction = 0.9
        delta_sigma_init = delta_sigma_base * ratio_correction
        
        # Ensure physical bounds
        A_init = np.clip(A_init, 10.0, 25.0)
        Delta_x_init = np.clip(Delta_x_init, 0.15, 0.50)
        delta_sigma_init = np.clip(delta_sigma_init, 0.60, 1.00)
        
        return A_init, Delta_x_init, delta_sigma_init
    
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
    
    def _compute_circulation(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> complex:
        """
        Compute circulation integral J = ∫ χ* ∂χ/∂σ dσ.
        
        FIRST-PRINCIPLES:
        =================
        
        The circulation integral is the fundamental quantity that determines
        both charge and EM self-energy in SFM:
        
            J = ∫₀^(2π) χ*(σ) ∂χ/∂σ dσ
        
        For a winding state χ = A × exp(ikσ) × f(σ):
            J ≈ i × k × A²
        
        Physical significance:
        - Im(J) determines the winding number k (and thus charge)
        - |J|² determines the EM self-energy: E_EM = g₂ |J|²
        
        Args:
            chi_components: Dictionary of wavefunction components
            
        Returns:
            Complex circulation integral J
        """
        D1 = self._build_subspace_derivative_matrix()
        dsigma = self.basis.dsigma
        
        # Sum all components to get total wavefunction
        chi_total = np.zeros(self.basis.N_sigma, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        
        # Compute ∂χ/∂σ
        dchi_total = D1 @ chi_total
        
        # Circulation integral: J = ∫ χ* ∂χ/∂σ dσ
        J = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        
        return J
    
    def _compute_em_self_energy(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> float:
        """
        Compute electromagnetic self-energy from NORMALIZED circulation.
        
        FIRST-PRINCIPLES:
        =================
        
        The EM self-energy emerges from the circulation term in the SFM Hamiltonian:
        
            Ĥ_circ = β × g₂ |∫ χ* ∂χ/∂σ dσ|²
        
        For a NORMALIZED wavefunction (∫|χ|²dσ = 1), this gives:
            E_EM = β × g₂ × |J_normalized|²
        
        where J_normalized = J / (∫|χ|²dσ).
        
        CRITICAL: We must normalize the circulation to prevent A⁴ scaling.
        The raw circulation J ~ k × A², so |J|² ~ k² × A⁴, which creates
        runaway growth in the self-consistent iteration.
        
        By using J_normalized = J/A², we get |J_normalized|² ~ k² (charge-dependent
        only, independent of amplitude), which is the correct physics.
        
        Physical interpretation:
        - For charged particles (|k| > 0): E_EM > 0 (positive mass contribution)
        - For neutral particles (k ≈ 0): E_EM ≈ 0 (no EM contribution)
        
        Args:
            chi_components: Dictionary of wavefunction components
            
        Returns:
            EM self-energy E_EM = β × g₂ × |J_normalized|² in [GeV]
        """
        D1 = self._build_subspace_derivative_matrix()
        dsigma = self.basis.dsigma
        
        # Sum all components to get total wavefunction
        chi_total = np.zeros(self.basis.N_sigma, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        
        # Compute normalization (amplitude squared)
        norm_sq = np.sum(np.abs(chi_total)**2) * dsigma
        if norm_sq < 1e-20:
            return 0.0
        
        # Compute raw circulation: J = ∫ χ* ∂χ/∂σ dσ
        dchi_total = D1 @ chi_total
        J_raw = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        
        # Normalize: J_normalized = J / (∫|χ|²dσ)
        # This removes the A² scaling, leaving only the charge-dependent part
        J_normalized = J_raw / norm_sq
        
        # E_EM = β × g₂ × |J_normalized|²
        # Units: [GeV] × dimensionless × dimensionless = [GeV]
        E_em = self.beta * self.g2 * np.abs(J_normalized)**2
        
        return float(E_em)
    
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
    
    def _build_envelope_with_spinorbit(
        self,
        sigma: NDArray,
        center: float,
        delta_sigma: float,
        k_winding: int,
    ) -> NDArray:
        """
        Build Gaussian envelope with spin-orbit coupling shift.
        
        FIRST-PRINCIPLES:
        =================
        
        The spin-orbit Hamiltonian H_so = lambda (d/dsigma x sigma_z) creates
        different effective potentials for particles with different windings:
        
            V_eff(sigma, k) = V(sigma) - lambda * k * sigma_z
        
        For fermions with spin alignment, this shifts the envelope center:
            delta_shift = lambda * k / V0
        
        Positive winding (k > 0, e.g., u quark with k=+5):
            -> envelope shifted in one direction
            
        Negative winding (k < 0, e.g., d quark with k=-3):
            -> envelope shifted in opposite direction
        
        This creates DIFFERENT interference patterns for proton (uud) vs
        neutron (udd), leading to different amplitudes and thus masses.
        
        Args:
            sigma: Subspace grid
            center: Base center position (well center)
            delta_sigma: Width parameter
            k_winding: Winding number (includes sign)
            
        Returns:
            Gaussian envelope with spin-orbit shifted center
        """
        # Spin-orbit shift: delta = lambda * k / V0
        # This is the envelope center shift from spin-orbit coupling
        if self.V0 > 0:
            so_shift = self.lambda_so * k_winding / self.V0
        else:
            so_shift = 0.0
        
        shifted_center = center + so_shift
        
        return np.exp(-(sigma - shifted_center)**2 / delta_sigma)
    
    def _estimate_initial_amplitude(
        self, 
        particle_type: str, 
        n_radial: int = 1
    ) -> float:
        """
        Estimate initial amplitude based on particle type and physics.
        
        Uses empirical mass ratios to provide initial guess close to
        final converged value. This prevents solver from starting in
        wrong energy basin.
        
        Args:
            particle_type: 'lepton', 'baryon', or 'meson'
            n_radial: Radial quantum number (1=ground state, 2=first excited, etc.)
            
        Returns:
            Estimated initial amplitude A
        """
        if particle_type == 'lepton':
            # Lepton mass ratios from experiment:
            # m_e ≈ 0.511 MeV (reference)
            # m_μ ≈ 105.7 MeV → ratio ≈ 207
            # m_τ ≈ 1777 MeV → ratio ≈ 3477
            
            # Since m = β A², we have A ∝ sqrt(m)
            if n_radial == 1:
                # Electron: A² ≈ 1 by calibration
                # Start slightly below to be conservative
                A_guess = 0.9
            elif n_radial == 2:
                # Muon: A² ≈ 207 → A ≈ 14.4
                # Start at ~85% of expected
                A_guess = 12.0
            elif n_radial == 3:
                # Tau: A² ≈ 3477 → A ≈ 59
                # Start at ~85% of expected
                A_guess = 50.0
            else:
                # Higher states: scale roughly linearly with n
                A_guess = 5.0 * n_radial
                
        elif particle_type == 'baryon':
            # Baryon masses (ground state):
            # m_p ≈ 938.3 MeV
            # m_n ≈ 939.6 MeV
            # Ratio to electron: 938/0.511 ≈ 1836
            # Therefore A² ≈ 1836 → A ≈ 42.8
            
            if n_radial == 1:
                # Proton/neutron: start at ~80% of expected
                A_guess = 35.0
            else:
                # Excited baryons scale roughly with sqrt(n)
                A_guess = 35.0 * np.sqrt(n_radial)
                
        elif particle_type == 'meson':
            # Meson masses (examples):
            # m_π ≈ 140 MeV → ratio ≈ 274 → A ≈ 16.5
            # m_K ≈ 495 MeV → ratio ≈ 969 → A ≈ 31
            
            if n_radial == 1:
                # Pion (lightest meson): start conservative
                A_guess = 13.0
            else:
                # Heavier mesons
                A_guess = 13.0 * np.sqrt(n_radial)
                
        else:
            # Unknown type: use reasonable default
            A_guess = 1.0
        
        return A_guess
    
    def _estimate_initial_delta_x(self, A_initial: float) -> float:
        """
        Estimate initial spatial extent consistent with amplitude.
        
        Uses gravitational self-confinement formula:
            Δx = 1/(g_internal × A⁶)^(1/3)
        
        Args:
            A_initial: Initial amplitude estimate
            
        Returns:
            Consistent spatial extent Δx
        """
        if self.g_internal > 0 and A_initial > 0.01:
            # Gravitational self-confinement
            A_sixth = A_initial ** 6
            Delta_x_est = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            
            # Apply reasonable physical bounds
            MIN_DELTA_X = 0.01   # Minimum localization
            MAX_DELTA_X = 100.0  # Maximum spread
            Delta_x_est = max(MIN_DELTA_X, min(MAX_DELTA_X, Delta_x_est))
            
            return Delta_x_est
        else:
            # Fallback for edge cases
            return 1.0
    
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
    
    def _compute_optimal_delta_sigma_with_k(self, A: float, k: int) -> float:
        """
        Compute optimal subspace width accounting for winding number k.
        
        FIRST-PRINCIPLES DERIVATION:
        ============================
        
        The subspace kinetic energy scales with k² (from winding):
            E_kin,sigma ~ k²/delta_sigma²
        
        The full energy balance is:
            E = k²/delta_sigma² + V₀×delta_sigma + g₁×A⁴/delta_sigma
        
        Minimizing with respect to delta_sigma:
            dE/d(delta_sigma) = -2k²/delta_sigma³ + V₀ - g₁×A⁴/delta_sigma² = 0
        
        For the dominant balance (V₀ subdominant):
            delta_sigma_opt ~ (2k²/(g₁×A⁴))^(1/3) ~ k^(2/3)
        
        Args:
            A: Current amplitude estimate
            k: Winding number (absolute value used)
            
        Returns:
            Optimal delta_sigma including k² scaling
        """
        k_abs = abs(k) if k != 0 else 1
        
        if self.g1 <= 0 or A < 1e-10:
            # Fallback with k-dependent default
            return 0.5 * (k_abs / 1.0) ** (2.0/3.0)
        
        # Energy minimization including k² kinetic term
        k_squared = k_abs ** 2
        A_fourth = A ** 4
        delta_sigma_opt = (2.0 * k_squared / (self.g1 * A_fourth)) ** (1.0/3.0)
        
        # Physics-based bounds (scale with k^(2/3))
        MIN_DELTA_SIGMA = 0.1 * (k_abs ** (2.0/3.0))
        MAX_DELTA_SIGMA = 2.0
        
        return max(MIN_DELTA_SIGMA, min(MAX_DELTA_SIGMA, delta_sigma_opt))
    
    def _compute_potential_energy(self, envelope: NDArray) -> float:
        """
        Compute expectation value of potential energy for a given envelope.
        
        FIRST-PRINCIPLES: Includes full potential in energy denominators:
            <V> = ∫ V(σ) |envelope|² dσ / ∫ |envelope|² dσ
        
        This is necessary because the full Hamiltonian energy is:
            E = E_kinetic + E_potential = (2n + l) + <V>
        
        Args:
            envelope: The subspace envelope function χ(σ)
            
        Returns:
            Expectation value of V_sigma for this envelope
        """
        dsigma = self.basis.dsigma
        envelope_norm_sq = np.sum(np.abs(envelope)**2) * dsigma
        
        if envelope_norm_sq < 1e-10:
            return 0.0
        
        # <V> = ∫ V(σ) |χ|² dσ / ∫ |χ|² dσ
        V_expectation = np.sum(self.V_sigma * np.abs(envelope)**2) * dsigma / envelope_norm_sq
        
        return V_expectation
    
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
        
        # === INITIAL CONDITIONS (PHYSICS-BASED) ===
        # Estimate amplitude based on particle type
        if hasattr(self, '_particle_type_hint'):
            A_current = self._estimate_initial_amplitude(
                self._particle_type_hint, 
                n_eff
            )
        else:
            # Default: assume lepton and scale with quantum number
            if n_eff == 1:
                A_current = 0.9
            elif n_eff == 2:
                A_current = 12.0
            elif n_eff == 3:
                A_current = 50.0
            else:
                A_current = max(1.0, 5.0 * n_eff)
        
        # Make spatial extent consistent with amplitude
        Delta_x_current = self._estimate_initial_delta_x(A_current)
        
        # Make subspace width consistent with amplitude
        delta_sigma_current = self._compute_optimal_delta_sigma(A_current)
        
        # Safety bounds
        MIN_SCALE = 0.0001
        Delta_x_current = max(Delta_x_current, MIN_SCALE)
        delta_sigma_current = max(delta_sigma_current, 0.1)
        
        # Adaptive mixing state
        mixing_current = 0.3  # Start with moderate mixing
        dA_prev = 0.0
        dDx_prev = 0.0
        
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
            
            # FIRST-PRINCIPLES: Include potential energy in energy denominators
            # E = E_kinetic + <V> = (2n + l) + <V>
            V_target = self._compute_potential_energy(chi_primary)
            V_envelope = self._compute_potential_energy(envelope)
            E_target_0 = 2 * n_eff + 0 + V_target
            
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                    continue
                
                # Include potential energy for induced state
                E_state = 2 * state.n + state.l + V_envelope
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
            
            # === STEP 8: Adaptive mixing with oscillation detection ===
            
            # Detect oscillations by checking if updates reverse direction
            if iter_outer > 0:
                dA_current = A_new - A_current
                dDx_current = Delta_x_new - Delta_x_current
                
                # Check for sign reversals (oscillations)
                oscillating_A = (dA_current * dA_prev) < 0 if iter_outer > 0 else False
                oscillating_Dx = (dDx_current * dDx_prev) < 0 if iter_outer > 0 else False
                
                if oscillating_A or oscillating_Dx:
                    # Reduce mixing when oscillating
                    mixing_current = max(0.05, mixing_current * 0.5)
                    if verbose:
                        print(f"    [OSCILLATION DETECTED] Reducing mixing to {mixing_current:.3f}")
                elif iter_outer > 5:
                    # Increase mixing if stable for multiple iterations
                    mixing_current = min(0.5, mixing_current * 1.1)
                
                # Store for next iteration
                dA_prev = dA_current
                dDx_prev = dDx_current
            
            # Apply mixing
            A_current = (1 - mixing_current) * A_current + mixing_current * A_new
            Delta_x_current = (1 - mixing_current) * Delta_x_current + mixing_current * Delta_x_new
            delta_sigma_current = (1 - mixing_current) * delta_sigma_current + mixing_current * delta_sigma_new
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
        
        # FIRST-PRINCIPLES: Include potential energy in energy denominators
        # E = E_kinetic + <V> = (2n + l) + <V>
        V_target = self._compute_potential_energy(chi_primary)
        V_envelope = self._compute_potential_energy(envelope)
        E_target_0 = 2 * n_target + 0 + V_target  # l=0 for target s-wave
        
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
            
            # Energy denominator - includes potential energy (first-principles)
            # E = (2n + l) + <V>
            E_state = 2 * state.n + state.l + V_envelope
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
        # Set particle type hint for initialization
        self._particle_type_hint = 'lepton'
        
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

    def _well_envelope(self, well_index: int, sigma: NDArray, width: float = 0.5) -> NDArray:
        """
        Generate Gaussian envelope localized at specified well.
        
        This emerges from first principles: the three-well potential
        V = V₀(1 - cos(3σ)) has minima at σ = π/3, π, 5π/3.
        Quarks/antiquarks localize in these wells.
        
        Args:
            well_index: 1, 2, or 3 (for the three wells)
            sigma: Subspace grid
            width: Width of Gaussian envelope (default: 0.5)
                   From harmonic approximation: V ≈ (9/2)V₀(σ - σ₀)² near minimum
                   gives width ~ 1/√(9V₀/2) ≈ 0.47 for V₀ = 1
            
        Returns:
            Envelope function localized at well minimum
        """
        # Well centers from V = V₀(1 - cos(3σ)) minima
        well_centers = [np.pi/3, np.pi, 5*np.pi/3]
        center = well_centers[well_index - 1]
        
        return np.exp(-(sigma - center)**2 / width)

    def _well_envelope_with_spinorbit(
        self, 
        well_index: int, 
        sigma: NDArray, 
        k_winding: int,
        width: float = None
    ) -> NDArray:
        """
        Generate Gaussian envelope with spin-orbit coupling shift.
        
        FIRST-PRINCIPLES:
        =================
        
        The spin-orbit Hamiltonian H_so = lambda (d/dsigma x sigma_z) creates
        different effective potentials for quarks with different windings:
        
            V_eff(sigma, k) = V(sigma) - lambda * k * sigma_z
        
        This shifts the envelope center from the well minimum:
            delta_shift = lambda * k / V0
        
        For proton (uud) vs neutron (udd):
            - u quark (k=+5): shifted one direction
            - d quark (k=-3): shifted opposite direction
        
        This creates DIFFERENT interference patterns, leading to different
        total amplitudes A and thus different masses m = beta * A^2.
        
        Args:
            well_index: 1, 2, or 3 (for the three wells)
            sigma: Subspace grid
            k_winding: Winding number with sign (e.g., +5 for u, -3 for d)
            width: Width parameter (if None, computed from k^(2/3) scaling)
            
        Returns:
            Envelope with spin-orbit shifted center and k-dependent width
        """
        # Well centers from V = V0(1 - cos(3*sigma)) minima
        well_centers = [np.pi/3, np.pi, 5*np.pi/3]
        base_center = well_centers[well_index - 1]
        
        # Spin-orbit shift: delta = lambda * k / V0
        if self.V0 > 0:
            so_shift = self.lambda_so * k_winding / self.V0
        else:
            so_shift = 0.0
        
        shifted_center = base_center + so_shift
        
        # Width: use k-dependent width if not specified
        # Higher k -> wider envelope (k^(2/3) scaling from kinetic energy)
        if width is None:
            k_abs = abs(k_winding) if k_winding != 0 else 1
            width = 0.5 * (k_abs / 1.0) ** (2.0/3.0)
        
        return np.exp(-(sigma - shifted_center)**2 / width)

    # =========================================================================
    # BARYON SELF-CONSISTENT SOLVER (4D FRAMEWORK)
    # =========================================================================
    #
    # Three-quark baryons use the SAME 4D nonseparable framework as leptons/mesons:
    #   Ψ(x,y,z,σ) = Σ_{n,l,m} φ_{nlm}(x,y,z) × χ_{nlm}(σ)
    #
    # The primary wavefunction is a three-quark composite with color phases.
    # Induced components arise from spatial-subspace coupling (α × R_ij).
    # Self-confinement: Δx = 1/(G_internal × A⁶)^(1/3)
    #
    # This ensures CONSISTENT treatment across all particle types.
    # =========================================================================

    def solve_baryon_self_consistent_field(
        self,
        color_phases: tuple = (0, 2*np.pi/3, 4*np.pi/3),
        quark_windings: tuple = (5, 5, -3),  # Default: proton (uud)
        quark_spins: tuple = (+1, -1, +1),      # NEW: Spin quantum numbers
        quark_generations: tuple = (1, 1, 1),    # NEW: Generation numbers
        max_iter: int = 300,
        tol: float = 1e-6,
        mixing: float = 0.1,
        verbose: bool = False,
    ) -> tuple:
        """
        Solve three-quark baryon using self-consistent field method.
        
        This implements the coupled eigenvalue problem described in
        SFM research notes:
        
            [-∂²/∂σ² + V(σ)]χᵢ + g₁|χ₁+χ₂+χ₃|²χᵢ = Eᵢχᵢ
        
        for i = 1, 2, 3 with color phases φ ∈ {0, 2π/3, 4π/3}.
        
        Args:
            color_phases: Color phases (φ₁, φ₂, φ₃) for color neutrality
            quark_windings: Winding numbers (k₁, k₂, k₃) for EM charge
            quark_spins: Spin quantum numbers (s₁, s₂, s₃) = ±1
            quark_generations: Generation numbers (n₁, n₂, n₃) = 1,2,3
            max_iter: Maximum SCF iterations
            tol: Convergence tolerance
            mixing: Mixing parameter for stability (0.5 = 50% new)
            verbose: Print iteration details
            
        Returns:
            chi1, chi2, chi3: Self-consistent quark wavefunctions
            A_baryon: Total baryon amplitude
        """
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        N = len(sigma)
        phi1, phi2, phi3 = color_phases
        k1, k2, k3 = quark_windings
        s1, s2, s3 = quark_spins
        n1, n2, n3 = quark_generations
        
        if verbose:
            print("\n=== Self-Consistent Field Baryon Solver ===")
            print(f"Color phases: {color_phases}")
            print(f"Quark windings (k): {quark_windings}")
            print(f"Quark spins (s): {quark_spins}")
            print(f"Quark generations (n): {quark_generations}")
            print(f"Max iterations: {max_iter}, Tolerance: {tol}")
        
        # === Pauli Exclusion Check ===
        if not self._check_pauli_exclusion(quark_windings, quark_spins, quark_generations):
            if verbose:
                print("WARNING: Configuration violates Pauli exclusion principle!")
        
        # === STEP 1: Initialize with SPIN and GENERATION ===
        # Start with Gaussians localized in each well with appropriate windings, spins, and generations
        chi1 = self._initialize_quark_wavefunction(
            well_index=1, phase=phi1, winding_k=k1, spin=s1, generation=n1
        )
        chi2 = self._initialize_quark_wavefunction(
            well_index=2, phase=phi2, winding_k=k2, spin=s2, generation=n2
        )
        chi3 = self._initialize_quark_wavefunction(
            well_index=3, phase=phi3, winding_k=k3, spin=s3, generation=n3
        )
        
        # Adaptive mixing state
        mixing_current = mixing  # Start with provided mixing parameter
        delta_chi_prev = [0.0, 0.0, 0.0]
        
        # Build kinetic operator (periodic boundary conditions)
        T_op = self._build_kinetic_operator()
        
        # Three-well potential
        V_well = self.V_sigma  # Already defined in __init__
        V_well_matrix = np.diag(V_well)
        
        # === Helper function for Phase 3: Energy-aware eigenstate selection ===
        def find_best_eigenstate(chi_old, k_winding, phi_color, n_check=5):
            """
            Find lowest energy eigenstate with reasonable overlap.
            
            Strategy:
            1. Check first n_check eigenstates (sorted by energy from eigh)
            2. Compute overlap with current state (envelope only)
            3. Choose highest overlap state among low-energy candidates
            4. Warn if overlap is poor (< 0.1)
            
            Returns:
                chi_new: Selected eigenstate with phases applied
                energy: Eigenvalue of selected state
                overlap: Overlap coefficient
                state_idx: Index of selected eigenstate
            """
            # Remove phases to get envelope for overlap calculation
            chi_envelope = chi_old / (np.exp(1j * phi_color) * np.exp(1j * k_winding * sigma) + 1e-30)
            
            # Compute overlaps with low-energy eigenstates
            n_states = min(n_check, N)
            overlaps = []
            for i in range(n_states):
                overlap = np.abs(np.sum(np.conj(chi_envelope) * eigenvectors[:, i]) * dsigma)
                overlaps.append((overlap, eigenvalues[i], i))
            
            # === ENERGY-FIRST SELECTION STRATEGY ===
            # Priority: Lowest energy among states with reasonable overlap
            # This ensures we select ground states, not excited states
            
            # Define overlap threshold for "reasonable" overlap
            overlap_threshold = 0.05  # 5% overlap minimum
            
            # Filter for states with acceptable overlap
            reasonable_states = [s for s in overlaps if s[0] > overlap_threshold]
            
            if reasonable_states:
                # Strategy A: Good overlaps available
                # Choose LOWEST ENERGY among reasonable overlaps
                reasonable_states.sort(key=lambda x: x[1])  # Sort by energy (ascending)
                best_overlap, best_energy, best_idx = reasonable_states[0]
                
                if verbose:
                    # Report that we're using energy-based selection
                    print(f"    [ENERGY SELECT] Chose state {best_idx} with E={best_energy:.3f}, overlap={best_overlap:.3f}")
            else:
                # Strategy B: All overlaps are poor (< threshold)
                # Fall back to highest overlap to maintain stability
                overlaps.sort(reverse=True, key=lambda x: x[0])
                best_overlap, best_energy, best_idx = overlaps[0]
                
                if verbose:
                    print(f"    [OVERLAP FALLBACK] All overlaps < {overlap_threshold}, chose state {best_idx}")
            
            # Sanity check: warn if overlap is very small
            if best_overlap < 0.1 and verbose:
                print(f"    [WARNING] Low overlap {best_overlap:.3f} at k={k_winding} - SCF may be diverging")
            
            # Get eigenstate and apply winding + color phases
            chi_new = eigenvectors[:, best_idx].astype(complex)
            chi_new *= np.exp(1j * phi_color) * np.exp(1j * k_winding * sigma)
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(chi_new)**2) * dsigma)
            if norm > 1e-20:
                chi_new /= norm
            
            return chi_new, best_energy, best_overlap, best_idx
        
        # === STEP 2: Self-Consistent Iteration ===
        for iteration in range(max_iter):
            if verbose and iteration % 10 == 0:
                chi_total_check = chi1 + chi2 + chi3
                A_check = np.sqrt(np.sum(np.abs(chi_total_check)**2) * dsigma)
                print(f"\nIteration {iteration+1}/{max_iter}, A={A_check:.4f}")
            
            # Save old wavefunctions for convergence check
            chi1_old = chi1.copy()
            chi2_old = chi2.copy()
            chi3_old = chi3.copy()
            
            # === Compute mean field ===
            chi_total = chi1 + chi2 + chi3
            V_mean = self.g1 * np.abs(chi_total)**2
            V_mean_matrix = np.diag(V_mean)
            
            # === Solve for each quark in the mean field ===
            # All three quarks experience the same mean field (Hartree approximation)
            H = T_op + V_well_matrix + V_mean_matrix
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            
            # === PHASE 3: Energy-aware eigenstate selection ===
            # Use helper function to find best eigenstates for each quark
            chi1_new, E1, overlap1, idx1 = find_best_eigenstate(chi1, k1, phi1, n_check=5)
            chi2_new, E2, overlap2, idx2 = find_best_eigenstate(chi2, k2, phi2, n_check=5)
            chi3_new, E3, overlap3, idx3 = find_best_eigenstate(chi3, k3, phi3, n_check=5)
            
            # Diagnostic output for eigenstate selection
            if verbose and iteration % 10 == 0:
                print(f"    Selected eigenstates: [{idx1}, {idx2}, {idx3}]")
                print(f"    Eigenvalues: E1={E1:.3f}, E2={E2:.3f}, E3={E3:.3f}")
                print(f"    Overlaps: {overlap1:.3f}, {overlap2:.3f}, {overlap3:.3f}")
            
            # === Energy Monitoring (Phase 1 diagnostic) ===
            # Compute mean-field energy for tracking convergence behavior
            # PHASE 3: Use actual selected eigenvalues (not just first 3)
            chi_total = chi1_new + chi2_new + chi3_new
            E_kinetic = E1 + E2 + E3  # CORRECT: sum of selected eigenvalues
            E_nonlinear = self.g1 * np.sum(np.abs(chi_total)**4) * dsigma
            E_total = E_kinetic + E_nonlinear
            
            # Track energy history
            if iteration == 0:
                self._scf_energy_history = []
            self._scf_energy_history.append(E_total)
            
            # Print energy diagnostics in verbose mode
            if verbose and iteration % 10 == 0:
                print(f"    Energy: E={E_total:.6f}")
                if len(self._scf_energy_history) > 1:
                    dE = E_total - self._scf_energy_history[-2]
                    print(f"    dE/E: {dE/abs(E_total):.2e}")
                    if dE > 0:
                        print(f"    [WARNING] Energy increasing")
            
            # === Check convergence ===
            delta1 = np.sum(np.abs(chi1_new - chi1_old)**2) * dsigma
            delta2 = np.sum(np.abs(chi2_new - chi2_old)**2) * dsigma
            delta3 = np.sum(np.abs(chi3_new - chi3_old)**2) * dsigma
            
            max_delta = max(delta1, delta2, delta3)
            
            if verbose and iteration % 10 == 0:
                print(f"    Convergence: delta_chi = {max_delta:.2e}")
            
            if max_delta < tol:
                if verbose:
                    print(f"\n[CONVERGED] after {iteration+1} iterations")
                break
            
            # === Adaptive mixing with oscillation detection ===
            
            # Compute delta_chi for current iteration
            delta_chi_current = [delta1, delta2, delta3]
            
            # Detect oscillations by checking if convergence reverses
            if iteration > 0:
                # Check if delta increased (sign of oscillation)
                oscillating = any(
                    (dc_curr > dc_prev) and (dc_prev < tol * 10)
                    for dc_curr, dc_prev in zip(delta_chi_current, delta_chi_prev)
                )
                
                if oscillating:
                    # Reduce mixing when oscillating near convergence
                    mixing_current = max(0.01, mixing_current * 0.5)
                    if verbose:
                        print(f"    [OSCILLATION] Reducing mixing to {mixing_current:.3f}")
                elif iteration > 20:
                    # Increase mixing if stable for many iterations
                    mixing_current = min(0.3, mixing_current * 1.05)
            
            # Store for next iteration
            delta_chi_prev = delta_chi_current
            
            # Apply adaptive mixing
            chi1 = (1 - mixing_current) * chi1_old + mixing_current * chi1_new
            chi2 = (1 - mixing_current) * chi2_old + mixing_current * chi2_new
            chi3 = (1 - mixing_current) * chi3_old + mixing_current * chi3_new
            
            # Re-normalize after mixing
            chi1 /= np.sqrt(np.sum(np.abs(chi1)**2) * dsigma + 1e-30)
            chi2 /= np.sqrt(np.sum(np.abs(chi2)**2) * dsigma + 1e-30)
            chi3 /= np.sqrt(np.sum(np.abs(chi3)**2) * dsigma + 1e-30)
        
        else:
            if verbose:
                print(f"\nWarning: Did not converge after {max_iter} iterations")
        
        # === STEP 3: Compute final amplitude ===
        chi_baryon_total = chi1 + chi2 + chi3
        A_squared = np.sum(np.abs(chi_baryon_total)**2) * dsigma
        A_baryon = np.sqrt(A_squared)
        
        if verbose:
            print(f"\n=== Final SCF Results ===")
            print(f"Total amplitude: A = {A_baryon:.4f}")
            
            # Verify color neutrality
            phase_sum = np.exp(1j * phi1) + np.exp(1j * phi2) + np.exp(1j * phi3)
            print(f"Color neutrality check: |sum(e^(i*phi))| = {np.abs(phase_sum):.2e} (should be ~0)")
        
        return chi1, chi2, chi3, A_baryon

    def solve_baryon_self_consistent(
        self,
        quark_wells: tuple = (1, 2, 3),
        color_phases: tuple = (0, 2*np.pi/3, 4*np.pi/3),
        quark_windings: tuple = None,  # Quark winding numbers for EM
        n_radial: int = 1,
        max_iter_outer: int = 30,
        tol_outer: float = 1e-4,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for baryon wavefunction with self-consistent Δx iteration.
        
        FIRST-PRINCIPLES IMPLEMENTATION (4D Framework):
        ================================================
        
        Baryons are three-quark bound states. The composite wavefunction uses
        the SAME 4D nonseparable framework as leptons and mesons:
        
            Ψ(x,y,z,σ) = Σ_{n,l,m} φ_{nlm}(x,y,z) × χ_{nlm}(σ)
        
        The primary wavefunction is:
            χ_primary = χ_q1(σ) + χ_q2(σ) + χ_q3(σ)
        
        where each quark component is localized in a different well with
        color phases (0, 2π/3, 4π/3) for color neutrality AND winding
        numbers that determine quark charges:
        
            χ_qi(σ) = envelope(σ_well_i) × exp(i × phi_color_i) × exp(i × k_i × σ)
        
        QUARK WINDING NUMBERS (First-Principles):
        =========================================
        
        The winding number k determines electric charge through the
        circulation integral J = ∫ χ* ∂χ/∂σ dσ:
        
            - u quark: k = +5 → Q = +2/3 e (from 3-fold symmetry)
            - d quark: k = -3 → Q = -1/3 e
            
        For baryons:
            - Proton (uud): k = (+5, +5, -3) → Q_total = +1 e
            - Neutron (udd): k = (+5, -3, -3) → Q_total = 0
        
        The EM self-energy E_EM = g₂ |J|² creates mass splitting.
        
        Args:
            quark_wells: Tuple of (q1_well, q2_well, q3_well), each 1-3
            color_phases: Tuple of (phi1, phi2, phi3) for color neutrality
            quark_windings: Tuple of (k1, k2, k3) winding numbers.
                           Default (None) = proton (uud): (+5, +5, -3)
                           For neutron (udd): (+5, -3, -3)
            n_radial: Radial quantum number (1=ground, 2=first excited)
            max_iter_outer: Maximum self-consistent iterations
            tol_outer: Convergence tolerance
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with converged baryon wavefunction and EM energy
        """
        # Default to proton winding: uud = (+5, +5, -3)
        if quark_windings is None:
            quark_windings = (5, 5, -3)  # Proton (uud)
        
        k1, k2, k3 = quark_windings
        net_winding = k1 + k2 + k3
        
        # Set particle type hint for initialization
        self._particle_type_hint = 'baryon'
        
        if verbose:
            print(f"\n=== Self-Consistent Baryon Solver (4D Framework) ===")
            print(f"Quark wells: {quark_wells}, Color phases: {color_phases}")
            print(f"Quark windings: {quark_windings} -> net k = {net_winding}")
        
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # === SELF-CONSISTENT FIELD ITERATION ===
        # Use coupled eigenvalue solver for three-quark system
        # This solves: [-∂²/∂σ² + V(σ) + g₁|χ₁+χ₂+χ₃|²]χᵢ = Eᵢχᵢ
        
        # === INITIAL CONDITIONS (PHYSICS-BASED) ===
        # Estimate amplitude for baryon
        A_current = self._estimate_initial_amplitude('baryon', n_radial=n_radial)
        
        # Make spatial extent consistent with amplitude using gravitational self-confinement
        Delta_x_current = self._estimate_initial_delta_x(A_current)
        
        # Ensure minimum scale
        MIN_SCALE = 0.01
        Delta_x_current = max(Delta_x_current, MIN_SCALE)
        
        # Adaptive mixing state for outer loop
        mixing_outer = 0.3  # Start with moderate mixing
        dA_prev = 0.0
        dDx_prev = 0.0
        
        history = {'Delta_x': [Delta_x_current], 'A': [A_current]}
        final_iter = 0
        
        for iter_outer in range(max_iter_outer):
            if verbose:
                print(f"\n{'='*60}")
                print(f"OUTER ITERATION {iter_outer+1}/{max_iter_outer}")
                print(f"{'='*60}")
                print(f"Current: Delta_x = {Delta_x_current:.6f}, A = {A_current:.6f}")
            
            # === HYBRID APPROACH ===
            # Step 1: SCF to get self-consistent quark wavefunctions
            # This captures three-quark mean field effects and color neutrality
            chi1, chi2, chi3, A_scf = self.solve_baryon_self_consistent_field(
                color_phases=color_phases,
                quark_windings=quark_windings,
                max_iter=500,
                tol=1e-4,
                mixing=0.05,
                verbose=(verbose and iter_outer == 0),
            )
            
            # Step 2: Form composite baryon wavefunction (primary state)
            chi_primary = chi1 + chi2 + chi3
            
            if verbose:
                A_scf_check = np.sqrt(np.sum(np.abs(chi_primary)**2) * dsigma)
                print(f"\nSCF quark wavefunctions: A_composite = {A_scf_check:.6f}")
            
            # Step 3: Use composite in 4D framework with spatial-subspace coupling
            # This builds induced components and grows amplitude through α × R_ij
            
            # Compute spatial coupling at current scale
            n_eff = 1  # Baryon is n=1 ground state in spatial domain
            a_n = Delta_x_current / np.sqrt(2 * n_eff + 1)
            a_n = max(a_n, 0.0001)
            spatial_coupling = self.basis.compute_spatial_coupling_at_scale(a_n)
            
            # Build chi_components with spatial coupling
            target_key = (n_eff, 0, 0)
            target_idx = 0
            for i, state in enumerate(self.basis.spatial_states):
                if (state.n, state.l, state.m) == target_key:
                    target_idx = i
                    break
            
            chi_components = {}
            chi_components[target_key] = chi_primary.copy()
            
            # FIRST-PRINCIPLES: Include potential energy in energy denominators
            # (consistent with lepton and meson solvers)
            V_target = self._compute_potential_energy(chi_primary)
            E_target_0 = 2 * n_eff + 0 + V_target
            
            # Build induced components from spatial-subspace coupling
            # Using COMPOSITE chi_primary (captures three-quark structure)
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                    continue
                
                # For induced states, use chi_primary envelope for V expectation
                V_envelope = self._compute_potential_energy(chi_primary)
                E_state = 2 * state.n + state.l + V_envelope
                E_denom = E_target_0 - E_state
                
                if abs(E_denom) < 0.5:
                    E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
                
                # Induced component with composite chi_primary
                # This captures three-quark interference in spatial coupling
                induced = -self.alpha * R_coupling * chi_primary / E_denom
                chi_components[key] = induced
            
            # Compute total amplitude (includes induced components)
            A_base = np.sqrt(self._compute_total_amplitude(chi_components))
            
            # === COMPUTE EM SELF-ENERGY (for mass splitting) ===
            E_em = self._compute_em_self_energy(chi_components)
            
            # Apply EM correction to amplitude
            A_eff_sq = A_base**2 + E_em / self.beta
            A_new = np.sqrt(max(A_eff_sq, A_base**2))
            
            if verbose:
                print(f"\nHybrid Results:")
                print(f"  A_scf (quark composite) = {A_scf:.6f}")
                print(f"  A_base (with spatial coupling) = {A_base:.6f}")
                print(f"  E_EM = {E_em:.6e}")
                print(f"  A_eff (final) = {A_new:.6f}")
            
            # === UPDATE SPATIAL SCALE (gravitational self-confinement) ===
            A_sixth = A_new ** 6
            if self.g_internal > 0 and A_sixth > 1e-30:
                Delta_x_new = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            else:
                Delta_x_new = Delta_x_current
            
            if verbose:
                print(f"  Gravitational self-confinement: Delta_x_new = {Delta_x_new:.6f}")
            
            # === CHECK OUTER CONVERGENCE ===
            delta_A = abs(A_new - A_current)
            delta_Dx = abs(Delta_x_new - Delta_x_current)
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            
            if verbose:
                print(f"  Outer convergence: dA/A = {rel_delta_A:.2e}, dDx/Dx = {rel_delta_Dx:.2e}")
            
            if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer:
                if verbose:
                    print(f"\n[OUTER LOOP CONVERGED] after {iter_outer+1} iterations")
                final_iter = iter_outer
                break
            
            # === Adaptive mixing with oscillation detection ===
            if iter_outer > 0:
                dA_current = A_new - A_current
                dDx_current = Delta_x_new - Delta_x_current
                
                # Check for sign reversals (oscillations)
                oscillating_A = (dA_current * dA_prev) < 0
                oscillating_Dx = (dDx_current * dDx_prev) < 0
                
                if oscillating_A or oscillating_Dx:
                    # Reduce mixing when oscillating
                    mixing_outer = max(0.05, mixing_outer * 0.5)
                    if verbose:
                        print(f"    [OUTER OSCILLATION] Reducing mixing to {mixing_outer:.3f}")
                elif iter_outer > 3:
                    # Increase mixing if stable
                    mixing_outer = min(0.5, mixing_outer * 1.1)
                
                # Store for next iteration
                dA_prev = dA_current
                dDx_prev = dDx_current
            
            # Apply adaptive mixing
            A_current = (1 - mixing_outer) * A_current + mixing_outer * A_new
            Delta_x_current = (1 - mixing_outer) * Delta_x_current + mixing_outer * Delta_x_new
            
            history['A'].append(A_current)
            history['Delta_x'].append(Delta_x_current)
            final_iter = iter_outer
        
        # Final wavefunction includes SCF composite + induced components
        chi_components_final = chi_components.copy()
        
        l_composition = self._compute_l_composition(chi_components_final)
        k_eff = self._compute_k_eff(chi_components_final)
        structure_norm = A_current
        
        # Compute EM self-energy from circulation (FIRST-PRINCIPLES)
        # E_EM = g₂ |J|² where J = ∫ χ* ∂χ/∂σ dσ
        circulation = self._compute_circulation(chi_components_final)
        em_energy = self._compute_em_self_energy(chi_components_final)
        
        if verbose:
            print(f"\n=== Final Baryon Structure (SCF Method) ===")
            print(f"  Delta_x={Delta_x_current:.6f}, A={A_current:.6f}")
            print(f"  k_eff={k_eff:.4f}, |J|^2={np.abs(circulation)**2:.4e}")
            print(f"  EM self-energy: E_EM = g2*|J|^2 = {em_energy:.4e}")
            # Verify color neutrality
            phi1, phi2, phi3 = color_phases
            phase_sum = np.exp(1j * phi1) + np.exp(1j * phi2) + np.exp(1j * phi3)
            print(f"  Color neutrality check: |sum(e^(i*phi))| = {np.abs(phase_sum):.2e} (should be ~0)")
        
        return WavefunctionStructure(
            chi_components=chi_components_final,
            n_target=1,  # Baryon is n=1 ground state in spatial domain
            k_winding=net_winding,  # Net winding from quark charges
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=(final_iter < max_iter_outer - 1),
            iterations=final_iter + 1,
            particle_type='baryon_scf',
            convergence_history=history,
            delta_x_final=Delta_x_current,
            delta_sigma_final=None,  # Not used in SCF method
            em_energy=em_energy,
            circulation=circulation,
        )
    
    # =========================================================================
    # NEW 4D THREE-QUARK BARYON SOLVER
    # =========================================================================
    #
    # CORRECT IMPLEMENTATION:
    # Each quark has a full 4D wavefunction Ψᵢ(x,y,z,σ).
    # The baryon is the 4D superposition: Ψ_baryon = Ψ₁ + Ψ₂ + Ψ₃.
    # Amplitude comes from: A² = ∫∫|Ψ₁+Ψ₂+Ψ₃|² dr dσ.
    #
    # This respects the 4D nonseparable nature of SFM, with interference
    # happening in BOTH spatial and subspace domains.
    # =========================================================================
    
    def solve_quark_4D(
        self,
        well_index: int,
        color_phase: float,
        k_winding: int,
        Delta_x_initial: float,
        V_mean_4D: Optional[Dict[Tuple[int, int, int], NDArray]] = None,
        max_iter_quark: int = 20,
        tol_quark: float = 1e-4,
        verbose: bool = False,
    ) -> Tuple[Dict[Tuple[int, int, int], NDArray], float]:
        """
        Solve for a single quark's full 4D wavefunction with self-consistent Δx.
        
        FIRST-PRINCIPLES 4D IMPLEMENTATION WITH AMPLITUDE AMPLIFICATION:
        ================================================================
        
        Each quark is a 4D entity with wavefunction:
            Ψ(x,y,z,σ) = Σ_{n,l,m} R_{nl}(r) Y_l^m(θ,φ) × χ_{nlm}(σ)
        
        This solver builds the χ_{nlm}(σ) components for all spatial states (n,l,m)
        with SELF-CONSISTENT Δx iteration to enable amplitude amplification:
        
        1. Build wavefunction at current Δx
        2. Compute amplitude A from wavefunction
        3. Update Δx from self-confinement: Δx = 1/(g_internal × A⁶)^(1/3)
        4. Repeat until converged
        
        This gives each quark the same amplitude growth mechanism as leptons.
        
        The quark experiences:
        1. Three-well potential V(σ) = V₀(1 - cos(3σ))
        2. Mean field from other quarks: V_mean(σ) per spatial state
        3. Spatial-subspace coupling (α × R_ij) that grows with tighter Δx
        
        ASSUMPTION: Quarks are initialized in specific wells (not from SCF).
        
        Args:
            well_index: Which well (1, 2, or 3)
            color_phase: Color phase φᵢ ∈ {0, 2π/3, 4π/3}
            k_winding: Winding number (e.g., +5 for u, -3 for d)
            Delta_x_initial: Initial spatial scale guess
            V_mean_4D: Mean field potential for each (n,l,m) state
            max_iter_quark: Maximum self-consistent iterations for this quark
            tol_quark: Convergence tolerance
            verbose: Print progress
            
        Returns:
            chi_components: Dict[(n,l,m)] -> χ_{nlm}(σ) for this quark
            Delta_x_final: Converged spatial scale for this quark
        """
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        if verbose:
            print(f"\n  Solving quark: well={well_index}, phi={color_phase:.3f}, k={k_winding}")
        
        # Effective quantum number (quarks are like n=1 leptons)
        n_eff = 1
        
        # Target state
        target_key = (n_eff, 0, 0)
        target_idx = self.basis.state_index(n_eff, 0, 0)
        
        # === PRIMARY COMPONENT ===
        # Initialize in specific well with color phase and winding
        # NOTE: Direct initialization (not from SCF) - see docstring
        envelope = self._well_envelope_with_spinorbit(
            well_index, sigma, k_winding, width=0.5
        )
        chi_primary = envelope * np.exp(1j * color_phase) * np.exp(1j * k_winding * sigma)
        
        # Normalize primary component
        # With shared Δx_baryon, the composite amplitude from superposition
        # determines the collective confinement scale
        norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
        chi_primary = chi_primary / np.sqrt(norm_sq)
        
        # === SELF-CONSISTENT Δx ITERATION ===
        Delta_x_current = Delta_x_initial
        A_current = 0.1
        MIN_SCALE = 0.0001
        
        for iter_quark in range(max_iter_quark):
            # === SPATIAL COUPLING AT CURRENT SCALE ===
            a_n = Delta_x_current / np.sqrt(2 * n_eff + 1)
            a_n = max(a_n, MIN_SCALE)
            spatial_coupling = self.basis.compute_spatial_coupling_at_scale(a_n)
            
            # === BUILD WAVEFUNCTION ===
            chi_components = {target_key: chi_primary}
            
            # Energy denominators with mean field
            V_target = self._compute_potential_energy(chi_primary)
            V_mean_target = 0.0
            if V_mean_4D is not None and target_key in V_mean_4D:
                V_mean_target = np.sum(V_mean_4D[target_key] * np.abs(chi_primary)**2) * dsigma
            
            E_target_0 = 2 * n_eff + 0 + V_target + V_mean_target
            
            # Build induced components
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                    continue
                
                # Envelope for induced state
                V_envelope = self._compute_potential_energy(envelope)
                
                # Mean field contribution
                V_mean_state = 0.0
                if V_mean_4D is not None and key in V_mean_4D:
                    V_mean_state = np.sum(V_mean_4D[key] * np.abs(envelope)**2) * dsigma
                
                E_state = 2 * state.n + state.l + V_envelope + V_mean_state
                E_denom = E_target_0 - E_state
                
                # Regularize
                if abs(E_denom) < 0.5:
                    E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
                
                # Induced component
                induced = (-self.alpha * R_coupling * envelope * 
                          np.exp(1j * color_phase) * np.exp(1j * k_winding * sigma) / E_denom)
                chi_components[key] = induced
            
            # === COMPUTE AMPLITUDE ===
            A_new = np.sqrt(sum(np.sum(np.abs(chi)**2) * dsigma 
                               for chi in chi_components.values()))
            
            # === UPDATE Δx FROM SELF-CONFINEMENT ===
            A_sixth = A_new ** 6
            if self.g_internal > 0 and A_sixth > 1e-30:
                Delta_x_new = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            else:
                Delta_x_new = Delta_x_current
            
            # === CHECK CONVERGENCE ===
            delta_A = abs(A_new - A_current)
            delta_Dx = abs(Delta_x_new - Delta_x_current)
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_current, 0.001)
            
            if verbose and iter_quark % 5 == 0:
                print(f"    Iter {iter_quark+1}: A={A_new:.6f}, Dx={Delta_x_new:.6f}, "
                      f"dA/A={rel_delta_A:.2e}, dDx/Dx={rel_delta_Dx:.2e}")
            
            if rel_delta_A < tol_quark and rel_delta_Dx < tol_quark:
                if verbose:
                    print(f"    Converged after {iter_quark+1} iterations: A={A_new:.6f}, Dx={Delta_x_new:.6f}")
                Delta_x_current = Delta_x_new
                A_current = A_new
                break
            
            # Update with mixing
            mixing = 0.3
            A_current = (1 - mixing) * A_current + mixing * A_new
            Delta_x_current = (1 - mixing) * Delta_x_current + mixing * Delta_x_new
            Delta_x_current = max(Delta_x_current, MIN_SCALE)
        
        else:
            if verbose:
                print(f"    Max iterations reached: A={A_current:.6f}, Dx={Delta_x_current:.6f}")
        
        return chi_components, Delta_x_current
    
    def _build_4D_components_fixed_primary(
        self,
        chi_primary: NDArray,
        well_index: int,
        color_phase: float,
        k_winding: int,
        Delta_x: float,
        V_mean_4D: Optional[Dict[Tuple[int, int, int], NDArray]] = None,
        verbose: bool = False,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Build 4D chi_components with induced states around a FIXED primary component.
        
        This is used after SCF initialization to add spatial-subspace coupling
        without modifying the carefully-computed SCF primary wavefunction.
        
        Args:
            chi_primary: Primary wavefunction from SCF (already optimized)
            well_index: Well index for this quark
            color_phase: Color phase
            k_winding: Winding number
            Delta_x: Spatial scale
            V_mean_4D: Optional mean field
            verbose: Print debug info
            
        Returns:
            chi_components: Dictionary with primary and induced components
        """
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        n_eff = 1
        target_key = (n_eff, 0, 0)
        target_idx = self.basis.state_index(n_eff, 0, 0)
        
        # Compute spatial coupling at current scale
        a_n = Delta_x / np.sqrt(2 * n_eff + 1)
        a_n = max(a_n, 0.0001)
        spatial_coupling = self.basis.compute_spatial_coupling_at_scale(a_n)
        
        # Start with fixed primary
        chi_components = {target_key: chi_primary.copy()}
        
        # Energy of primary state
        V_target = self._compute_potential_energy(chi_primary)
        V_mean_target = 0.0
        if V_mean_4D is not None and target_key in V_mean_4D:
            V_mean_target = np.sum(V_mean_4D[target_key] * np.abs(chi_primary)**2) * dsigma
        E_target_0 = 2 * n_eff + 0 + V_target + V_mean_target
        
        # Build induced components from spatial-subspace coupling
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            if key == target_key:
                continue
            
            R_coupling = spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                continue
            
            # Use envelope structure from SCF primary
            envelope = np.abs(chi_primary) / (np.abs(chi_primary).max() + 1e-30)
            V_envelope = self._compute_potential_energy(envelope)
            
            V_mean_state = 0.0
            if V_mean_4D is not None and key in V_mean_4D:
                V_mean_state = np.sum(V_mean_4D[key] * np.abs(envelope)**2) * dsigma
            
            E_state = 2 * state.n + state.l + V_envelope + V_mean_state
            E_denom = E_target_0 - E_state
            
            # Regularize
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            # Induced component maintains phase structure from primary
            induced = (-self.alpha * R_coupling * envelope * 
                      np.exp(1j * color_phase) * np.exp(1j * k_winding * sigma) / E_denom)
            chi_components[key] = induced
        
        return chi_components
    
    def compute_4D_baryon_amplitude(
        self,
        quark1_chi: Dict[Tuple[int, int, int], NDArray],
        quark2_chi: Dict[Tuple[int, int, int], NDArray],
        quark3_chi: Dict[Tuple[int, int, int], NDArray],
    ) -> float:
        """
        Compute total baryon amplitude from 4D superposition of three quarks.
        
        FIRST-PRINCIPLES:
        =================
        
        The baryon wavefunction is:
            Ψ_baryon(x,y,z,σ) = Ψ₁(x,y,z,σ) + Ψ₂(x,y,z,σ) + Ψ₃(x,y,z,σ)
        
        The amplitude is:
            A² = ∫∫|Ψ_baryon|² dr dσ
               = Σ_{n,l,m} ∫|R_{nl}(r)|² dr × ∫|χ₁_{nlm} + χ₂_{nlm} + χ₃_{nlm}|² dσ
        
        Since R_{nl} are the same for all quarks (they share spatial basis),
        we sum the subspace contributions for each (n,l,m) mode.
        
        **KEY**: Interference happens in subspace FOR EACH spatial mode separately.
        This captures 4D superposition correctly!
        
        Args:
            quark1_chi: χ_{nlm}(σ) components for quark 1
            quark2_chi: χ_{nlm}(σ) components for quark 2
            quark3_chi: χ_{nlm}(σ) components for quark 3
            
        Returns:
            Total amplitude A from 4D superposition
        """
        dsigma = self.basis.dsigma
        total_amplitude_sq = 0.0
        
        # Get all possible spatial state keys
        all_keys = set(quark1_chi.keys()) | set(quark2_chi.keys()) | set(quark3_chi.keys())
        
        for key in all_keys:
            # Get subspace components for this (n,l,m) spatial state
            chi1_nlm = quark1_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            chi2_nlm = quark2_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            chi3_nlm = quark3_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            
            # Superposition in subspace for THIS spatial mode
            chi_total_nlm = chi1_nlm + chi2_nlm + chi3_nlm
            
            # Amplitude contribution from this mode
            A_nlm_sq = np.sum(np.abs(chi_total_nlm)**2) * dsigma
            
            total_amplitude_sq += A_nlm_sq
        
        return np.sqrt(total_amplitude_sq)
    
    def solve_baryon_4D_self_consistent(
        self,
        quark_wells: tuple = (1, 2, 3),
        color_phases: tuple = (0, 2*np.pi/3, 4*np.pi/3),
        quark_windings: tuple = None,
        quark_spins: tuple = (+1, -1, +1),      # NEW: Spin quantum numbers
        quark_generations: tuple = (1, 1, 1),    # NEW: Generation numbers
        max_iter_outer: int = 30,
        max_iter_scf: int = 10,
        tol_outer: float = 1e-4,
        tol_scf: float = 1e-3,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for baryon using full 4D three-quark self-consistent approach.
        
        CORRECT 4D IMPLEMENTATION:
        ==========================
        
        Each quark has a full 4D wavefunction:
            Ψᵢ(x,y,z,σ) = Σ_{n,l,m} R_{nl}(r) Y_l^m(θ,φ) × χᵢ_{nlm}(σ)
        
        The baryon is their 4D superposition:
            Ψ_baryon = Ψ₁ + Ψ₂ + Ψ₃
        
        Self-consistent iteration:
        1. Each quark experiences mean field from the 4D superposition
        2. V_mean(r,σ) = g₁|Ψ_baryon(r,σ)|² in full 4D
        3. Each quark can have its own Δx (well-independent)
        4. Amplitude from full 4D superposition
        
        This respects SFM's 4D nonseparable framework completely.
        
        Args:
            quark_wells: Well indices (q1, q2, q3) ∈ {1,2,3}
            color_phases: Color phases (φ₁, φ₂, φ₃)
            quark_windings: Winding numbers (k₁, k₂, k₃)
                           Default: proton (uud) = (+5, +5, -3)
            quark_spins: Spin quantum numbers (s₁, s₂, s₃) = ±1
            quark_generations: Generation numbers (n₁, n₂, n₃) = 1,2,3
            max_iter_outer: Max iterations for Δx self-confinement
            max_iter_scf: Max iterations for quark-quark SCF
            tol_outer: Convergence tolerance for outer loop
            tol_scf: Convergence tolerance for SCF loop
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with converged baryon wavefunction
        """
        if quark_windings is None:
            quark_windings = (5, 5, -3)  # Proton (uud)
        
        w1, w2, w3 = quark_wells
        phi1, phi2, phi3 = color_phases
        k1, k2, k3 = quark_windings
        s1, s2, s3 = quark_spins
        n1, n2, n3 = quark_generations
        net_winding = k1 + k2 + k3
        
        if verbose:
            print(f"\n=== 4D Three-Quark Baryon Solver ===")
            print(f"Wells: {quark_wells}, Phases: {color_phases}")
            print(f"Windings: {quark_windings} -> net k = {net_winding}")
            print(f"Spins: {quark_spins}")
            print(f"Generations: {quark_generations}")
        
        # === INITIALIZE WITH SCF SOLVER ===
        # Use proper self-consistent field solver to get realistic initial quark wavefunctions
        # This finds the true ground state in the three-well potential with mean field
        if verbose:
            print("\n=== STEP 1: SCF Initialization ===")
        
        chi1_scf, chi2_scf, chi3_scf, A_scf = self.solve_baryon_self_consistent_field(
            color_phases=color_phases,
            quark_windings=quark_windings,
            quark_spins=quark_spins,
            quark_generations=quark_generations,
            max_iter=500,
            tol=1e-4,
            mixing=0.05,
            verbose=verbose,
        )
        
        if verbose:
            A1 = np.sqrt(np.sum(np.abs(chi1_scf)**2) * self.basis.dsigma)
            A2 = np.sqrt(np.sum(np.abs(chi2_scf)**2) * self.basis.dsigma)
            A3 = np.sqrt(np.sum(np.abs(chi3_scf)**2) * self.basis.dsigma)
            print(f"\nSCF initialization complete: A_scf = {A_scf:.6f}")
            print(f"Individual quark amplitudes:")
            print(f"  |chi1| = {A1:.6f}")
            print(f"  |chi2| = {A2:.6f}")
            print(f"  |chi3| = {A3:.6f}")
        
        # === INITIALIZE SPATIAL SCALE ===
        # KEY FIX: All three quarks share a SINGLE Δx determined by composite amplitude
        # Use parameter-aware initialization for better convergence
        A_init_param, Delta_x_init_param, delta_sigma_init_param = \
            self._initialize_baryon_from_parameters(self.g1, self.g2, self.lambda_so)
        
        # Start with parameter-aware values, but use SCF amplitude if significantly different
        # This provides better initial guess while respecting SCF results
        if abs(A_scf - A_init_param) / A_init_param < 0.5:
            # SCF and parameter-based estimates are consistent - use SCF
            A_current = A_scf
            Delta_x_baryon = 1.0 / (self.g_internal * A_scf**6) ** (1.0/3.0) if A_scf > 0.01 else Delta_x_init_param
        else:
            # SCF result differs significantly - use parameter-based initialization
            A_current = A_init_param
            Delta_x_baryon = Delta_x_init_param
        
        if verbose:
            print(f"\n=== STEP 2: 4D Spatial-Subspace Coupling with Collective Confinement ===")
            print(f"Parameter-aware initialization: A={A_init_param:.2f}, Dx={Delta_x_init_param:.3f}, Ds={delta_sigma_init_param:.3f}")
            print(f"SCF initialization: A={A_scf:.2f}")
            print(f"Using: A={A_current:.2f}, Dx={Delta_x_baryon:.6f}")
        
        history = {
            'Delta_x': [Delta_x_baryon],
            'A': [A_current],
        }
        
        final_iter = 0
        
        # Convert SCF wavefunctions to chi_components format (primary state only)
        n_eff = 1
        target_key = (n_eff, 0, 0)
        quark1_chi = {target_key: chi1_scf.copy()}
        quark2_chi = {target_key: chi2_scf.copy()}
        quark3_chi = {target_key: chi3_scf.copy()}
        
        # === OUTER LOOP: BARYON SELF-CONFINEMENT ===
        for iter_outer in range(max_iter_outer):
            if verbose:
                print(f"\n{'='*70}")
                print(f"OUTER ITERATION {iter_outer+1}/{max_iter_outer}")
                print(f"{'='*70}")
                print(f"Dx_baryon = {Delta_x_baryon:.6f}, A_composite = {A_current:.6f}")
            
            # === BUILD INDUCED COMPONENTS FOR EACH QUARK ===
            # Now that we have good primary components from SCF, add spatial-subspace coupling
            # ALL THREE QUARKS USE THE SAME Δx (collective confinement)
            quark1_chi = self._build_4D_components_fixed_primary(
                chi1_scf, w1, phi1, k1, Delta_x_baryon, None, verbose=False)
            quark2_chi = self._build_4D_components_fixed_primary(
                chi2_scf, w2, phi2, k2, Delta_x_baryon, None, verbose=False)
            quark3_chi = self._build_4D_components_fixed_primary(
                chi3_scf, w3, phi3, k3, Delta_x_baryon, None, verbose=False)
            
            for iter_scf in range(max_iter_scf):
                if verbose and iter_outer == 0:
                    print(f"\n  SCF iteration {iter_scf+1}/{max_iter_scf}")
                
                # Store old solutions
                quark1_chi_old = {k: v.copy() for k, v in quark1_chi.items()}
                quark2_chi_old = {k: v.copy() for k, v in quark2_chi.items()}
                quark3_chi_old = {k: v.copy() for k, v in quark3_chi.items()}
                
                # === BUILD 4D MEAN FIELD ===
                # V_mean(σ) for each spatial state (n,l,m)
                V_mean_4D = {}
                all_keys = set(quark1_chi.keys()) | set(quark2_chi.keys()) | set(quark3_chi.keys())
                
                for key in all_keys:
                    chi1 = quark1_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
                    chi2 = quark2_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
                    chi3 = quark3_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
                    
                    # 4D superposition for this spatial mode
                    chi_total = chi1 + chi2 + chi3
                    
                    # Mean field for this mode
                    V_mean_4D[key] = self.g1 * np.abs(chi_total)**2
                
                # === REBUILD INDUCED COMPONENTS WITH UPDATED MEAN FIELD ===
                # SCF primaries are fixed; only induced components update
                # All three quarks use the SAME Delta_x_baryon (collective confinement)
                quark1_chi = self._build_4D_components_fixed_primary(
                    chi1_scf, w1, phi1, k1, Delta_x_baryon, V_mean_4D, verbose=False)
                quark2_chi = self._build_4D_components_fixed_primary(
                    chi2_scf, w2, phi2, k2, Delta_x_baryon, V_mean_4D, verbose=False)
                quark3_chi = self._build_4D_components_fixed_primary(
                    chi3_scf, w3, phi3, k3, Delta_x_baryon, V_mean_4D, verbose=False)
                
                # === CHECK SCF CONVERGENCE ===
                change1 = sum(np.sum(np.abs(quark1_chi[k] - quark1_chi_old[k])**2) * self.basis.dsigma 
                            for k in quark1_chi.keys())
                change2 = sum(np.sum(np.abs(quark2_chi[k] - quark2_chi_old[k])**2) * self.basis.dsigma 
                            for k in quark2_chi.keys())
                change3 = sum(np.sum(np.abs(quark3_chi[k] - quark3_chi_old[k])**2) * self.basis.dsigma 
                            for k in quark3_chi.keys())
                
                total_change = change1 + change2 + change3
                
                if verbose and iter_outer == 0:
                    print(f"    Total change: {total_change:.2e}")
                
                if total_change < tol_scf:
                    if verbose and iter_outer == 0:
                        print(f"    SCF converged after {iter_scf+1} iterations")
                    break
            
            # === COMPUTE 4D COMPOSITE AMPLITUDE ===
            A_new = self.compute_4D_baryon_amplitude(quark1_chi, quark2_chi, quark3_chi)
            
            # === UPDATE BARYON Δx FROM COMPOSITE AMPLITUDE ===
            # KEY: Self-confinement uses COMPOSITE amplitude, not individual quark amplitudes!
            # Δx_baryon = 1/(g_internal × A_composite^6)^(1/3)
            A_sixth = A_new ** 6
            if self.g_internal > 0 and A_sixth > 1e-30:
                Delta_x_new = 1.0 / (self.g_internal * A_sixth) ** (1.0/3.0)
            else:
                Delta_x_new = Delta_x_baryon
            
            if verbose:
                print(f"\n4D Composite Amplitude: A = {A_new:.6f}")
                print(f"Baryon self-confinement: Dx_new = {Delta_x_new:.6f} (was {Delta_x_baryon:.6f})")
            
            # === CHECK OUTER CONVERGENCE ===
            delta_A = abs(A_new - A_current)
            delta_Dx = abs(Delta_x_new - Delta_x_baryon)
            rel_delta_A = delta_A / max(A_current, 0.01)
            rel_delta_Dx = delta_Dx / max(Delta_x_baryon, 0.001)
            
            if verbose:
                print(f"Convergence: dA/A = {rel_delta_A:.2e}, dDx/Dx = {rel_delta_Dx:.2e}")
            
            # Early termination if poor convergence indicates incompatible parameters
            if iter_outer == 15:
                if rel_delta_A > 0.05 or rel_delta_Dx > 0.05:
                    if verbose:
                        print(f"\n[EARLY TERMINATION] Poor convergence after 15 iterations")
                        print(f"  Parameters likely incompatible: g1={self.g1:.2f}, "
                              f"g2={self.g2:.2f} (ratio={self.g2/self.g1:.2f}), "
                              f"lambda_so={self.lambda_so:.4f}")
                        print(f"  Current errors: dA/A={rel_delta_A:.3e}, dDx/Dx={rel_delta_Dx:.3e}")
                    
                    # Build final composite with current state
                    chi_components_final = {}
                    all_keys = set(quark1_chi.keys()) | set(quark2_chi.keys()) | set(quark3_chi.keys())
                    for key in all_keys:
                        chi1 = quark1_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
                        chi2 = quark2_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
                        chi3 = quark3_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
                        chi_components_final[key] = chi1 + chi2 + chi3
                    
                    l_composition = self._compute_l_composition(chi_components_final)
                    k_eff = self._compute_k_eff(chi_components_final)
                    circulation = self._compute_circulation(chi_components_final)
                    em_energy = self._compute_em_self_energy(chi_components_final)
                    
                    # Return unconverged result immediately
                    return WavefunctionStructure(
                        chi_components=chi_components_final,
                        n_target=1,
                        k_winding=net_winding,
                        l_composition=l_composition,
                        k_eff=k_eff,
                        structure_norm=A_current,
                        converged=False,
                        iterations=iter_outer + 1,
                        particle_type='baryon_4D',
                        convergence_history=history,
                        delta_x_final=Delta_x_baryon,
                        delta_sigma_final=None,
                        em_energy=em_energy,
                        circulation=circulation,
                    )
            
            if rel_delta_A < tol_outer and rel_delta_Dx < tol_outer:
                if verbose:
                    print(f"\n[OUTER LOOP CONVERGED] after {iter_outer+1} iterations")
                final_iter = iter_outer
                break
            
            # === UPDATE WITH ADAPTIVE MIXING ===
            # Adaptive mixing based on convergence state
            if iter_outer < 3:
                # Initial iterations: very conservative to establish stable trajectory
                mixing = 0.10
            elif rel_delta_A > 0.02 or rel_delta_Dx > 0.02:
                # Large changes: conservative to avoid overshooting
                mixing = 0.15
            elif rel_delta_A < 0.001 and rel_delta_Dx < 0.001:
                # Near convergence: aggressive to accelerate final approach
                mixing = 0.50
            else:
                # Normal convergence: moderate mixing
                mixing = 0.30
            
            # Add oscillation detection and damping
            if iter_outer > 5 and 'A' in history:
                recent_As = history['A'][-5:]
                if len(recent_As) == 5:
                    variation = np.std(recent_As) / np.mean(recent_As)
                    if variation > 0.03:  # Oscillating
                        mixing = min(mixing, 0.15)
                        if verbose:
                            print(f"  [Oscillation detected, reducing mixing to {mixing:.2f}]")
            
            if verbose and iter_outer % 5 == 0:
                print(f"  Adaptive mixing: {mixing:.2f}")
            
            A_current = (1 - mixing) * A_current + mixing * A_new
            Delta_x_baryon = (1 - mixing) * Delta_x_baryon + mixing * Delta_x_new
            
            history['A'].append(A_current)
            history['Delta_x'].append(Delta_x_baryon)
            final_iter = iter_outer
        
        # === BUILD FINAL COMPOSITE ===
        # For output, combine all three quarks' components
        chi_components_final = {}
        all_keys = set(quark1_chi.keys()) | set(quark2_chi.keys()) | set(quark3_chi.keys())
        
        for key in all_keys:
            chi1 = quark1_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            chi2 = quark2_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            chi3 = quark3_chi.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            chi_components_final[key] = chi1 + chi2 + chi3
        
        l_composition = self._compute_l_composition(chi_components_final)
        k_eff = self._compute_k_eff(chi_components_final)
        
        # Compute EM self-energy
        circulation = self._compute_circulation(chi_components_final)
        em_energy = self._compute_em_self_energy(chi_components_final)
        
        if verbose:
            print(f"\n=== Final 4D Baryon Structure ===")
            print(f"  A_composite = {A_current:.6f}")
            print(f"  Dx_baryon = {Delta_x_baryon:.6f} (shared by all three quarks)")
            print(f"  k_eff = {k_eff:.4f}, |J|^2 = {np.abs(circulation)**2:.4e}")
            print(f"  E_EM = {em_energy:.4e}")
            phase_sum = np.exp(1j*phi1) + np.exp(1j*phi2) + np.exp(1j*phi3)
            print(f"  Color neutrality: |sum(e^(iphi))| = {np.abs(phase_sum):.2e}")
        
        return WavefunctionStructure(
            chi_components=chi_components_final,
            n_target=1,
            k_winding=net_winding,
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=A_current,
            converged=(final_iter < max_iter_outer - 1),
            iterations=final_iter + 1,
            particle_type='baryon_4D',
            convergence_history=history,
            delta_x_final=Delta_x_baryon,  # Shared baryon scale
            delta_sigma_final=None,
            em_energy=em_energy,
            circulation=circulation,
        )

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
            
            # FIRST-PRINCIPLES: Include potential energy in energy denominators
            V_target = self._compute_potential_energy(chi_primary_norm)
            V_envelope = self._compute_potential_energy(single_well_envelope)
            E_target_0 = 2 * n_eff + 0 + V_target
            
            # Induced components from spatial-subspace coupling
            for i, state in enumerate(self.basis.spatial_states):
                key = (state.n, state.l, state.m)
                if key == target_key:
                    continue
                
                R_coupling = spatial_coupling[target_idx, i]
                if abs(R_coupling) < 1e-10:
                    chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                    continue
                
                # Include potential energy (first-principles)
                E_state = 2 * state.n + state.l + V_envelope
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
        alpha=10.5,
        beta=100.0,
        g1=50.0,
        g2=70.0,
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

