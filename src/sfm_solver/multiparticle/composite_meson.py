"""
Composite Meson Solver for SFM.

COMPLETE ENERGY FUNCTIONAL (from "A Beautiful Balance" research notes):

    E_total = E_subspace + E_spatial + E_coupling + E_curvature

Where:
    E_subspace = kinetic + potential + nonlinear (in σ)
    E_spatial  = β×A²×c² (rest mass from amplitude)
    E_coupling = -α × n × k × A (stabilizes amplitude, creates mass hierarchy)
    E_curvature = κ × A⁴ / Δx (prevents collapse to A=0)

KEY PHYSICS:
1. WINDING NUMBERS FROM CHARGE:
   - Up-type (u, c, t): k = 5 
   - Down-type (d, s, b): k = 3
   - Meson composite winding: k_meson = k_q + k_qbar (signed sum)

2. AMPLITUDE QUANTIZATION:
   - The four-term energy balance creates discrete stable amplitudes
   - E_curvature ∝ A⁴ prevents collapse to vacuum
   - E_coupling ∝ A provides stabilization at finite amplitude
   - Different mesons have different equilibrium amplitudes → different masses

3. DESTRUCTIVE INTERFERENCE:
   - Meson (ud̄): opposite windings → smaller |k_meson|
   - Smaller k → weaker coupling → smaller equilibrium amplitude → lighter mass
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


# Quark winding numbers - DERIVED from charge relationship Q = k/3
# Up-type quarks: Q = +2/3 → k = 5 (with 3-fold symmetry, 5 mod 3 = 2 → 2/3)
# Down-type quarks: Q = -1/3 → k = 3 (with phase, gives -1/3)
QUARK_WINDING = {
    'u': 5, 'd': 3,
    'c': 5, 's': 3,
    'b': 5, 't': 3,
}

# Quark spatial mode number - FROM SFM framework (same as leptons)
# This is NOT artificial - it follows from the coupling mechanism
# that creates the lepton mass hierarchy (e, μ, τ → n=1,2,3)
# Quarks follow the same pattern by generation.
QUARK_SPATIAL_MODE = {
    'u': 1, 'd': 1,  # Generation 1 (like electron)
    'c': 2, 's': 2,  # Generation 2 (like muon)
    'b': 3, 't': 3,  # Generation 3 (like tau)
}

# Meson configurations (experimental masses for reference only - not used in calculation)
MESON_CONFIGS = {
    'pion_plus': {'quark': 'u', 'antiquark': 'd', 'mass_mev': 139.57},
    'pion_zero': {'quark': 'u', 'antiquark': 'u', 'mass_mev': 134.98},
    'pion_minus': {'quark': 'd', 'antiquark': 'u', 'mass_mev': 139.57},
    'kaon_plus': {'quark': 'u', 'antiquark': 's', 'mass_mev': 493.68},
    'jpsi': {'quark': 'c', 'antiquark': 'c', 'mass_mev': 3096.90},
    'upsilon_1s': {'quark': 'b', 'antiquark': 'b', 'mass_mev': 9460.30},
}

# NOTE: For heavy quarkonia (J/ψ, Υ), the full prediction requires
# solving the coupled spacetime-subspace problem to get the spatial
# mode number n, which determines the amplitude enhancement.
# Currently we only solve the subspace part.


@dataclass
class CompositeMesonState:
    """Result of composite meson solver."""
    chi_meson: NDArray[np.complexfloating]
    
    # Meson configuration
    meson_type: str
    quark: str
    antiquark: str
    
    # Amplitude (NOT normalized to 1!)
    amplitude_squared: float  # A² = ∫|χ|² dσ - determines mass via m = β×A²
    
    # Complete four-term energy breakdown
    energy_total: float
    energy_subspace: float     # Kinetic + potential + nonlinear + circulation
    energy_spatial: float      # Rest mass contribution: β×A²×c²
    energy_coupling: float     # Stabilizing term: -α×n×|k|×A
    energy_curvature: float    # Prevents collapse: κ×A⁴/Δx
    
    # Winding structure
    k_quark: int
    k_antiquark: int
    k_meson: int       # Net winding: k_q + k_qbar (for circulation/charge, can be 0)
    k_coupling: int    # Bare coupling winding: |k_q| + |k_qbar| (for reference)
    k_eff: float       # EMERGENT effective winding from wavefunction gradient
    
    # Generation (spatial mode number)
    generation: int
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeMesonSolver:
    """
    Solver for meson as a composite wavefunction.
    
    COMPLETE ENERGY FUNCTIONAL (four terms):
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
    
    This follows the "Beautiful Balance" mechanism that creates
    discrete stable amplitudes and prevents collapse to vacuum.
    """
    
    WELL_POSITIONS = [0.0, 2*np.pi/3, 4*np.pi/3]
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g1: float = 0.1,
        g2: float = 0.1,
        alpha: float = 2.0,         # Subspace-spacetime coupling strength
        beta: float = 1.0,          # Mass coupling (m = β × A²)
        kappa: float = 0.5,         # Curvature energy coefficient
        delta_x: float = 1.0,       # Characteristic spatial localization
        m_eff: float = 1.0,
        hbar: float = 1.0,
        c: float = 1.0,             # Speed of light
    ):
        self.grid = grid
        self.potential = potential
        self.g1 = g1
        self.g2 = g2
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.delta_x = delta_x
        self.m_eff = m_eff
        self.hbar = hbar
        self.c = c
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
    
    def _get_quark_params(self, quark: str, antiquark: str) -> Tuple[int, int, int, int, int, int, int]:
        """
        Get winding numbers and spatial mode numbers.
        
        KEY INSIGHT: Antiparticles have opposite circulation (charge) but same
        magnitude kinetic energy and coupling strength.
        
        Returns:
            k_q: Quark winding (positive)
            k_qbar: Antiquark winding (negative)
            k_net: Net winding = k_q + k_qbar (for circulation/charge)
            k_sq_total: k_q² + k_qbar² (for kinetic energy, never zero)
            k_coupling: |k_q| + |k_qbar| (for coupling energy, never zero)
            n_q: Quark spatial mode (1, 2, 3 for generations)
            n_qbar: Antiquark spatial mode
        """
        k_q = QUARK_WINDING.get(quark, 3)
        k_qbar = -QUARK_WINDING.get(antiquark, 3)  # Antiquark has negative winding
        
        # For circulation/charge: signed sum (can be zero for cc̄, bb̄)
        k_net = k_q + k_qbar
        
        # For kinetic energy: sum of squares (never zero)
        k_sq_total = k_q**2 + k_qbar**2
        
        # For coupling energy: sum of magnitudes (never zero!)
        # This is crucial for J/ψ (cc̄) where k_net=0 but |k_q|+|k_qbar|=10
        k_coupling = abs(k_q) + abs(k_qbar)
        
        # Spatial mode number (from SFM - same mechanism as leptons)
        n_q = QUARK_SPATIAL_MODE.get(quark, 1)
        n_qbar = QUARK_SPATIAL_MODE.get(antiquark, 1)
        
        return k_q, k_qbar, k_net, k_sq_total, k_coupling, n_q, n_qbar
    
    def _compute_circulation(self, chi: NDArray[np.complexfloating]) -> complex:
        """
        Compute actual circulation from the wavefunction.
        
        J = ∫ χ* (dχ/dσ) dσ
        
        For a single winding k: J = ik × A²
        For meson (q + q̄): J includes INTERFERENCE terms that create cancellation!
        
        This is the CORRECT way to get the effective coupling - from the
        actual wavefunction structure, not from assumed parameters.
        """
        dchi = self.grid.first_derivative(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        return J
    
    def _compute_k_eff_from_wavefunction(self, chi: NDArray[np.complexfloating]) -> float:
        """
        Compute effective winding number from actual wavefunction gradient.
        
        k²_eff = ∫|∂χ/∂σ|² dσ / ∫|χ|² dσ
        
        This AUTOMATICALLY accounts for:
        ✅ Destructive interference (reduces k_eff)
        ✅ Constructive interference (increases k_eff)
        ✅ Mixed winding numbers
        ✅ All composite structures
        
        For pion (ud̄): opposite windings → destructive → small k_eff
        For baryon-like: same-sign windings → constructive → large k_eff
        """
        # Compute wavefunction gradient
        dchi_dsigma = self.grid.first_derivative(chi)
        
        # ∫|∂χ/∂σ|² dσ (measures "waviness")
        numerator = np.sum(np.abs(dchi_dsigma)**2) * self.grid.dsigma
        
        # ∫|χ|² dσ (normalization)
        denominator = np.sum(np.abs(chi)**2) * self.grid.dsigma
        
        if denominator < 1e-10:
            return 0.0
        
        # k_eff = sqrt(⟨|∇χ|²⟩ / ⟨|χ|²⟩)
        k_eff = np.sqrt(numerator / denominator)
        return float(k_eff)
    
    def _initialize_meson(
        self,
        quark: str,
        antiquark: str,
        initial_amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Initialize meson as a SINGLE composite wavefunction.
        
        Like the baryon solver, we use a SINGLE winding for the composite:
            k_meson = k_q + k_qbar (net winding)
        
        The composite has TWO peaks (quark and antiquark regions) with
        the same net winding but different color phases.
        
        The peak positions and relative phases will adjust during
        energy minimization to find the equilibrium configuration.
        """
        sigma = self.grid.sigma
        N = len(sigma)
        
        k_q, k_qbar, k_net, _, _, _, _ = self._get_quark_params(quark, antiquark)
        
        chi = np.zeros(N, dtype=complex)
        width = 0.4  # Localization width
        
        # Initialize with INDIVIDUAL windings for quark and antiquark
        # This creates the actual interference pattern that determines k_eff
        # Quark at well 0 with winding k_q, antiquark at well 1 with winding k_qbar
        
        well_q = self.WELL_POSITIONS[0]
        well_qbar = self.WELL_POSITIONS[1]
        
        # Quark contribution: Gaussian at well 0 with winding k_q
        dist_q = np.angle(np.exp(1j * (sigma - well_q)))
        envelope_q = np.exp(-0.5 * (dist_q / width)**2)
        chi_q = envelope_q * np.exp(1j * k_q * sigma)
        
        # Antiquark contribution: Gaussian at well 1 with winding k_qbar (negative!)
        dist_qbar = np.angle(np.exp(1j * (sigma - well_qbar)))
        envelope_qbar = np.exp(-0.5 * (dist_qbar / width)**2)
        chi_qbar = envelope_qbar * np.exp(1j * k_qbar * sigma)  # k_qbar is already negative
        
        # Composite wavefunction: sum of quark and antiquark
        # The interference between e^{ik_q σ} and e^{ik_qbar σ} is what creates
        # the effective winding k_eff and determines the coupling energy
        chi = chi_q + chi_qbar
        
        # Scale to desired initial amplitude
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        if current_amp_sq > 1e-10:
            chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def _compute_interference_factor(
        self,
        chi: NDArray[np.complexfloating],
        quark: str,
        antiquark: str
    ) -> float:
        """
        Compute interference factor: ratio of actual amplitude to maximum possible.
        
        interference_factor = |χ_q + χ_qbar|² / (|χ_q|² + |χ_qbar|²)
        
        For constructive: factor ≈ 2 (or up to 4 with perfect phase)
        For destructive: factor << 1
        
        This is EMERGENT - we measure the actual interference.
        """
        sigma = self.grid.sigma
        k_q, k_qbar, _, _, _ = self._get_quark_params(quark, antiquark)
        
        width = 0.5
        
        # Reconstruct individual contributions (without interference)
        envelope = np.exp(-0.5 * (np.angle(np.exp(1j * sigma))**2 / width**2))
        chi_q_alone = envelope * np.exp(1j * k_q * sigma)
        chi_qbar_alone = envelope * np.exp(1j * (k_qbar * sigma + np.pi))
        
        # What we'd have without interference
        amp_separate = np.sum(np.abs(chi_q_alone)**2 + np.abs(chi_qbar_alone)**2) * self.grid.dsigma
        
        # Actual amplitude with interference
        amp_actual = np.sum(np.abs(chi)**2) * self.grid.dsigma
        
        if amp_separate > 1e-10:
            return amp_actual / amp_separate
        return 1.0
    
    def _compute_energy(
        self,
        chi: NDArray[np.complexfloating],
        generation: int
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Compute total energy with COMPLETE four-term functional.
        
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
        
        KEY INSIGHT: k_eff is computed FROM THE ACTUAL WAVEFUNCTION:
            k²_eff = ∫|∂χ/∂σ|² dσ / ∫|χ|² dσ
        
        This AUTOMATICALLY accounts for interference effects:
        - Destructive interference (opposite windings) → small k_eff
        - Constructive interference → large k_eff
        
        Where (from "A Beautiful Balance" research notes):
            E_subspace = kinetic + potential + nonlinear + circulation (∝ A²)
            E_spatial  = ℏ² / (2 × β × A² × Δx²) (∝ 1/A² - PREVENTS COLLAPSE!)
            E_coupling = -α × n × k_eff(χ) × A (EMERGENT k_eff from wavefunction)
            E_curvature = κ × β² × A⁴ / Δx (∝ A⁴ - limits maximum)
        """
        # Amplitude
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        
        # === COMPUTE k_eff FROM ACTUAL WAVEFUNCTION ===
        # This is the key insight: k_eff emerges from the composite structure
        k_eff = self._compute_k_eff_from_wavefunction(chi)
        
        # === SUBSPACE ENERGY COMPONENTS ===
        
        # Kinetic: ∫(ℏ²/2m)|∇χ|² dσ
        T_chi = self.operators.apply_kinetic(chi)
        E_kin = np.real(self.grid.inner_product(chi, T_chi))
        
        # Potential: ∫V(σ)|χ|² dσ (scaled by generation for heavier quarks)
        E_pot = generation * np.real(np.sum(self._V_grid * np.abs(chi)**2) * self.grid.dsigma)
        
        # Nonlinear: (g₁/2)∫|χ|⁴ dσ
        E_nl = (self.g1 / 2) * np.sum(np.abs(chi)**4) * self.grid.dsigma
        
        # Circulation: g₂|J|² where J = ∫χ*∂χ/∂σ dσ
        J = self._compute_circulation(chi)
        E_circ = self.g2 * np.abs(J)**2
        
        E_subspace = E_kin + E_pot + E_nl + E_circ
        
        # === SPATIAL ENERGY (LOCALIZATION) ===
        # From Beautiful Balance: E_spatial = ℏ² / (2 × m × Δx²)
        # where m = β × A², so E_spatial = ℏ² / (2 × β × A² × Δx²)
        # This scales as 1/A² - PREVENTS COLLAPSE by making E → ∞ as A → 0
        E_spatial = self.hbar**2 / (2 * self.beta * max(A_sq, 1e-10) * self.delta_x**2)
        
        # === COUPLING ENERGY (from H_coupling = -α ∂²/∂r∂σ) ===
        # E_coupling = -α × n × k_eff(χ) × A
        # k_eff is EMERGENT from the actual wavefunction gradient!
        # This automatically captures interference effects
        E_coupling = -self.alpha * generation * k_eff * A
        
        # === CURVATURE ENERGY (gravitational self-energy) ===
        # E_curvature = κ × m² / Δx = κ × (β×A²)² / Δx = κ×β²×A⁴/Δx
        # This is POSITIVE and limits maximum amplitude
        E_curvature = self.kappa * (self.beta * A_sq)**2 / self.delta_x
        
        # Total energy from all four components
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
        
        return E_total, E_subspace, E_spatial, E_coupling, E_curvature, E_circ, A_sq, k_eff
    
    def _compute_gradient(
        self,
        chi: NDArray[np.complexfloating],
        generation: int
    ) -> NDArray[np.complexfloating]:
        """
        Compute energy gradient for the complete four-term functional.
        
        Uses k_eff computed from the actual wavefunction gradient.
        
        δE/δχ* for each term:
            δE_subspace/δχ* = T_chi + V_chi + g₁|χ|²χ + circ_grad
            δE_spatial/δχ*  = -ℏ²/(β×A⁴×Δx²) × χ
            δE_coupling/δχ* = -α×n×k_eff/(2A) × χ (k_eff from wavefunction)
            δE_curvature/δχ* = 4κβ²A²/Δx × χ
        """
        # Amplitude
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        
        # k_eff from actual wavefunction (emergent!)
        k_eff = self._compute_k_eff_from_wavefunction(chi)
        
        # === SUBSPACE GRADIENT ===
        
        # Kinetic
        T_chi = self.operators.apply_kinetic(chi)
        
        # Potential (scaled by generation)
        V_chi = generation * self._V_grid * chi
        
        # Nonlinear
        NL_chi = self.g1 * np.abs(chi)**2 * chi
        
        # Circulation gradient: δ(g₂|J|²)/δχ*
        dchi = self.grid.first_derivative(chi)
        J = self._compute_circulation(chi)
        circ_grad = 2 * self.g2 * np.real(np.conj(J)) * dchi
        
        grad_subspace = T_chi + V_chi + NL_chi + circ_grad
        
        # === SPATIAL GRADIENT (1/A² term) ===
        # E_spatial = ℏ²/(2βA²Δx²)
        # δE_spatial/δA² = -ℏ²/(2βA⁴Δx²)
        # δA²/δχ* = χ
        # So: δE_spatial/δχ* = -ℏ²/(2βA⁴Δx²) × χ
        grad_spatial = -self.hbar**2 / (2 * self.beta * max(A_sq, 1e-10)**2 * self.delta_x**2) * chi
        
        # === COUPLING GRADIENT ===
        # δ(-α×n×k_eff×A)/δχ* ≈ -α×n×k_eff/(2A) × χ
        # Note: k_eff also depends on χ, but we treat it as approximately constant
        # for the gradient step (quasi-Newton approximation)
        grad_coupling = -self.alpha * generation * k_eff / (2 * A + 1e-10) * chi
        
        # === CURVATURE GRADIENT ===
        # δ(κβ²A⁴/Δx)/δχ* = 4×κ×β²×A²/Δx × χ
        grad_curvature = 4 * self.kappa * self.beta**2 * A_sq / self.delta_x * chi
        
        return grad_subspace + grad_spatial + grad_coupling + grad_curvature
    
    def solve(
        self,
        meson_type: str = 'pion_plus',
        max_iter: int = 5000,
        tol: float = 1e-8,
        dt: float = 0.0005,
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> CompositeMesonState:
        """
        Solve for meson ground state using complete four-term energy functional.
        
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
        
        KEY PHYSICS:
        Antiparticles have opposite circulation but same magnitude kinetic energy
        and coupling strength.
        
        1. k_net = k_q + k_qbar (for circulation/charge - can be zero for cc̄, bb̄)
        2. k_sq_total = k_q² + k_qbar² (for kinetic energy - never zero)
        3. k_coupling = |k_q| + |k_qbar| (for coupling - NEVER ZERO!)
        4. E_coupling = -α × n × k_coupling × A (stabilizes amplitude)
        5. E_curvature = κ × A⁴ / Δx (prevents collapse to vacuum)
        """
        config = MESON_CONFIGS.get(meson_type, MESON_CONFIGS['pion_plus'])
        quark = config['quark']
        antiquark = config['antiquark']
        
        k_q, k_qbar, k_net, k_sq_total, k_coupling, n_q, n_qbar = self._get_quark_params(quark, antiquark)
        
        # Use maximum generation for the meson
        generation = max(n_q, n_qbar)
        
        if verbose:
            print("=" * 60)
            print(f"COMPOSITE MESON SOLVER: {meson_type.upper()}")
            print(f"  Quark: {quark} (k={k_q}, n={n_q})")
            print(f"  Antiquark: {antiquark}-bar (k={k_qbar}, n={n_qbar})")
            print(f"  k_net = {k_net} (for circulation/charge)")
            print(f"  k_sq_total = {k_sq_total} (for kinetic energy)")
            print(f"  k_coupling_bare = {k_coupling} (sum of magnitudes)")
            print(f"  Generation: {generation}")
            print(f"  Parameters: alpha={self.alpha}, beta={self.beta}, kappa={self.kappa}")
            print("  k_eff computed from actual wavefunction gradient (EMERGENT)")
            print("  Energy: E = E_subspace + E_spatial + E_coupling + E_curvature")
            print("=" * 60)
        
        # Initialize as composite with individual quark/antiquark windings
        chi = self._initialize_meson(quark, antiquark, initial_amplitude)
        E_old, _, _, _, _, _, A_sq, k_eff = self._compute_energy(chi, generation)
        
        converged = False
        final_residual = float('inf')
        
        for iteration in range(max_iter):
            # Gradient descent with complete energy functional
            gradient = self._compute_gradient(chi, generation)
            chi_new = chi - dt * gradient
            
            # Compute new energy with all four terms (k_eff is emergent!)
            E_new, E_sub, E_spat, E_coup, E_curv, E_circ, A_sq_new, k_eff_new = self._compute_energy(
                chi_new, generation
            )
            
            # Adaptive step size
            if E_new > E_old:
                dt *= 0.5
                continue
            else:
                dt = min(dt * 1.05, 0.005)
            
            # Convergence check
            dE = abs(E_new - E_old)
            final_residual = dE
            
            if verbose and iteration % 500 == 0:
                print(f"  Iter {iteration}: E={E_new:.4f}, A²={A_sq_new:.4f}, "
                      f"k_eff={k_eff_new:.2f}, E_coup={E_coup:.4f}")
            
            if dE < tol:
                converged = True
                chi = chi_new
                break
            
            chi = chi_new
            E_old = E_new
        
        # Final results with complete energy breakdown
        E_total, E_subspace, E_spatial, E_coupling, E_curvature, E_circ, A_sq, k_eff = self._compute_energy(
            chi, generation
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS (from four-term energy balance):")
            print(f"  Amplitude A² = {A_sq:.6f}")
            print(f"  k_eff = {k_eff:.4f} (EMERGENT from wavefunction)")
            print(f"  E_total = {E_total:.6f}")
            print(f"  E_subspace = {E_subspace:.6f}")
            print(f"  E_spatial = {E_spatial:.6f}")
            print(f"  E_coupling = {E_coupling:.6f} (uses emergent k_eff)")
            print(f"  E_curvature = {E_curvature:.6f} (prevents collapse)")
            print(f"  Converged: {converged} ({iteration+1} iterations)")
            print("=" * 60)
        
        return CompositeMesonState(
            chi_meson=chi,
            meson_type=meson_type,
            quark=quark,
            antiquark=antiquark,
            amplitude_squared=float(A_sq),
            energy_total=float(E_total),
            energy_subspace=float(E_subspace),
            energy_spatial=float(E_spatial),
            energy_coupling=float(E_coupling),
            energy_curvature=float(E_curvature),
            k_quark=k_q,
            k_antiquark=k_qbar,
            k_meson=k_net,        # Net winding for circulation/charge
            k_coupling=k_coupling, # Bare sum of magnitudes (for reference)
            k_eff=float(k_eff),    # EMERGENT from wavefunction gradient
            generation=generation,
            converged=converged,
            iterations=iteration + 1,
            final_residual=float(final_residual),
        )
    
    def solve_pion(self, **kwargs) -> CompositeMesonState:
        """Solve for pion (π⁺ = ud̄)."""
        return self.solve(meson_type='pion_plus', **kwargs)
    
    def solve_jpsi(self, **kwargs) -> CompositeMesonState:
        """Solve for J/ψ (cc̄)."""
        return self.solve(meson_type='jpsi', **kwargs)
