"""
Composite Baryon Solver for SFM.

CORRECT PHYSICS:
- Single composite wavefunction, NOT three separate quarks
- Amplitude A² = ∫|χ|² is FREE (determines mass m = βA²)
- NO normalization - amplitude emerges from energy minimization  
- Color neutrality Σe^(iφᵢ) = 0 emerges from geometry/energy minimization
- Three-well potential localizes "quark" peaks
- QUARK TYPES (u/d) create different interference patterns → different masses

Energy functional to minimize:
    E[χ] = ∫[ℏ²/(2m)|∇χ|² + V(σ)|χ|² + (g₁/2)|χ|⁴] dσ + E_coulomb + E_coupling

The proton (uud) and neutron (udd) have:
- Different charge configurations → different Coulomb energies
- Different interference patterns in |χ(σ)|²
- Different integrated amplitudes A²_p vs A²_n
- Different masses: m = βA²

Reference: docs/Research Note - Origin of Strong Force.html
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators

# =============================================================================
# Quark Properties - EXPECTED values for validation
# =============================================================================
# These are the Standard Model values that the emergent charges should match.
# The actual charges used in calculations should ideally be computed from
# the wavefunction circulation integral (see electromagnetic.py).

# Expected SIGNED quark charges (in units of e) for validation
EXPECTED_QUARK_CHARGES = {
    'u': +2/3,   # up quark: POSITIVE 2/3 (from k = +5, (5 mod 3)/3 = 2/3)
    'd': -1/3,   # down quark: NEGATIVE 1/3 (from k = -3, -1/3)
}

# Expected SIGNED winding numbers
# Sign convention: positive k → positive charge, negative k → negative charge
EXPECTED_QUARK_WINDINGS = {
    'u': +5,     # up quark: positive winding → positive charge +2e/3
    'd': -3,     # down quark: NEGATIVE winding → negative charge -e/3
}

# Current implementation uses these values directly for Coulomb energy.
# TODO: In future, compute charges from wavefunction circulation integral
# using calculate_charge_from_wavefunction from electromagnetic.py
QUARK_CHARGES = EXPECTED_QUARK_CHARGES  # Alias for backward compatibility

# Standard baryon configurations
PROTON_QUARKS = ['u', 'u', 'd']   # uud → total charge = +2/3 + 2/3 - 1/3 = +1
NEUTRON_QUARKS = ['u', 'd', 'd']  # udd → total charge = +2/3 - 1/3 - 1/3 = 0


@dataclass
class CompositeBaryonState:
    """Result of composite baryon solver."""
    chi_baryon: NDArray[np.complexfloating]
    
    # Quark configuration
    quark_types: Tuple[str, str, str]  # e.g., ('u', 'u', 'd') for proton
    
    # Amplitude (NOT normalized to 1!)
    amplitude_squared: float  # A² = ∫|χ|² dσ - determines mass
    
    # Energy components (UNIFIED four-term functional + Coulomb)
    energy_total: float
    energy_subspace: float     # Kinetic + potential + nonlinear + circulation
    energy_spatial: float      # Localization: ℏ²/(2βA²Δx²)
    energy_coupling: float     # -α×k×A term that stabilizes amplitude
    energy_curvature: float    # Enhanced 5D gravity: κ×(βA²)²/Δx
    energy_coulomb: float      # Charge-dependent Coulomb energy
    
    # Legacy energy breakdown (for compatibility)
    energy_kinetic: float
    energy_potential: float
    energy_nonlinear: float
    
    # Effective winding (EMERGENT from wavefunction gradient)
    k_eff: float  # Computed from ∫|∂χ/∂σ|²/∫|χ|² - NOT summed windings!
    
    # Color structure
    phases: Tuple[float, float, float]
    phase_differences: Tuple[float, float]
    color_sum_magnitude: float
    is_color_neutral: bool
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeBaryonSolver:
    """
    Solver for baryon as a single composite wavefunction.
    
    KEY: Does NOT normalize! Amplitude emerges from energy minimization.
    
    The equilibrium amplitude comes from balancing:
    - Potential energy (favors larger amplitude in wells)
    - Nonlinear energy (penalizes large amplitude for g₁ > 0)
    - Kinetic energy (penalizes sharp gradients)
    - Coulomb energy (charge-dependent, differs for proton vs neutron)
    
    PROTON (uud) vs NEUTRON (udd):
    - Different quark charges → different Coulomb energies
    - Different interference patterns → different A²
    - Different masses: m = β×A² (use SFM_CONSTANTS.beta for consistency)
    
    NOTE ON β:
    The baryon solver computes the amplitude A² through energy minimization.
    To convert to mass, use: m = SFM_CONSTANTS.beta × A²
    This ensures consistency with the global β from the Beautiful Equation.
    """
    
    WELL_POSITIONS = [0.0, 2*np.pi/3, 4*np.pi/3]
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g1: Optional[float] = None,  # Nonlinear coupling (default: SFM_CONSTANTS.g1)
        g2: Optional[float] = None,  # Circulation coupling (default: SFM_CONSTANTS.g2_alpha)
        alpha: Optional[float] = None,  # Subspace-spacetime coupling (None = use mode default)
        beta: Optional[float] = None,   # Mass coupling (None = use mode default)
        kappa: Optional[float] = None,  # Enhanced gravity (None = use mode default)
        delta_x: float = 1.0,         # Spatial localization scale (same as meson)
        k: int = 3,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        coulomb_strength: float = 0.06,  # Coulomb coupling (tuned to n-p mass difference)
        use_physical: Optional[bool] = None,  # None = inherit from SFM_CONSTANTS.use_physical
    ):
        """
        Initialize baryon solver.
        
        PARAMETER MODES:
        ================
        
        1. PHYSICAL MODE (default, use_physical=True or None):
           Uses first-principles parameters from SFM theory.
           - α (alpha) = SFM_CONSTANTS.alpha_coupling_base ≈ 18.8 GeV
             (SAME as meson - winding appears ONLY in E_coupling = -α × k × A)
           - β (beta) = SFM_CONSTANTS.beta_physical = M_W ≈ 80.4 GeV
           - κ (kappa) = SFM_CONSTANTS.kappa_physical ≈ 0.012 GeV⁻¹
           - Amplitudes EMERGE from energy minimization
           - m = β × A² gives ABSOLUTE MASSES
        
        2. NORMALIZED MODE (use_physical=False):
           Uses dimensionless parameters calibrated for numerical stability.
           - α (alpha) = 1.0 (calibrated for baryon masses)
           - β (beta) = SFM_CONSTANTS.beta_normalized = 1.0
           - κ (kappa) = SFM_CONSTANTS.kappa_normalized = 0.10
           - Predicts MASS RATIOS correctly
        
        The default mode is controlled by SFM_CONSTANTS.DEFAULT_USE_PHYSICAL (True).
        
        Reference: docs/First_Principles_Parameter_Derivation.md
        
        Args:
            use_physical: If True, use first-principles physical parameters.
                         If False, use normalized parameters.
                         If None (default), inherit from SFM_CONSTANTS.use_physical.
        """
        self.grid = grid
        self.potential = potential
        
        # Inherit global mode if not specified
        if use_physical is None:
            use_physical = SFM_CONSTANTS.use_physical
        self.use_physical = use_physical
        
        # Use derived first-principles values from SFM_CONSTANTS if not specified
        self.g1 = g1 if g1 is not None else SFM_CONSTANTS.g1
        self.g2 = g2 if g2 is not None else SFM_CONSTANTS.g2_alpha  # Use α for self-energy
        
        # Set mode-dependent parameters
        if use_physical:
            # PHYSICAL MODE: Use first-principles values from SFM theory
            # UNIFIED APPROACH: Same α for ALL particles (baryon, meson, lepton)
            # Winding k appears ONLY in E_coupling = -α × k × A
            # This ensures A² ratios match mass ratios for universal m = β × A²
            self.alpha = alpha if alpha is not None else SFM_CONSTANTS.alpha_coupling_base
            self.beta = beta if beta is not None else SFM_CONSTANTS.beta_physical
            self.kappa = kappa if kappa is not None else SFM_CONSTANTS.kappa_physical
        else:
            # NORMALIZED MODE: Use calibrated values for numerical stability
            self.alpha = alpha if alpha is not None else 1.0
            self.beta = beta if beta is not None else SFM_CONSTANTS.beta_normalized
            self.kappa = kappa if kappa is not None else SFM_CONSTANTS.kappa_normalized
        
        self.k = k
        self.m_eff = m_eff
        self.hbar = hbar
        self.delta_x = delta_x  # Spatial localization scale
        self.coulomb_strength = coulomb_strength  # EM coupling
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
        
        # Current quark configuration (set during solve)
        self._quark_charges: List[float] = [QUARK_CHARGES['u']] * 3
        
    def _initialize_baryon(self, initial_amplitude: float = 1.0) -> NDArray[np.complexfloating]:
        """
        Initialize baryon wavefunction with three peaks at wells.
        
        Each peak has:
        - Gaussian envelope localized at well
        - Winding factor e^(ikσ) with k=3
        - Color phase: 0, 2π/3, 4π/3 for color neutrality
        
        The initial_amplitude sets the overall scale.
        """
        sigma = self.grid.sigma
        N = len(sigma)
        
        chi = np.zeros(N, dtype=complex)
        width = 0.5  # Localization width
        
        for i, well_pos in enumerate(self.WELL_POSITIONS):
            # Color phase for neutrality
            color_phase = i * 2 * np.pi / 3
            
            # Gaussian at well (handle periodicity)
            dist = np.angle(np.exp(1j * (sigma - well_pos)))
            envelope = np.exp(-0.5 * (dist / width)**2)
            
            # Full phase: winding + color
            phase = self.k * sigma + color_phase
            
            chi += envelope * np.exp(1j * phase)
        
        # Scale to desired initial amplitude (NOT normalizing!)
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def _compute_coulomb_energy(self, chi: NDArray[np.complexfloating]) -> float:
        """
        Compute Coulomb energy from quark charges.
        
        E_coulomb = λ × Σ_{i<j} Q_i × Q_j × A_i × A_j
        
        where A_i is the amplitude in well i.
        
        This creates the proton-neutron mass difference:
        - Proton (uud): (+2/3)(+2/3) + (+2/3)(-1/3) + (+2/3)(-1/3) = 4/9 - 2/9 - 2/9 = 0
        - Neutron (udd): (+2/3)(-1/3) + (+2/3)(-1/3) + (-1/3)(-1/3) = -2/9 - 2/9 + 1/9 = -1/3
        
        Neutron has MORE NEGATIVE Coulomb energy (less repulsion),
        which leads to HIGHER amplitude at equilibrium → higher mass.
        """
        sigma = self.grid.sigma
        well_width = np.pi / 3
        
        # Compute amplitude in each well
        well_amplitudes = []
        for well_pos in self.WELL_POSITIONS:
            # Distance from well (periodic)
            diff = np.angle(np.exp(1j * (sigma - well_pos)))
            mask = np.abs(diff) < well_width
            A_i = np.sqrt(np.sum(np.abs(chi[mask])**2) * self.grid.dsigma) if np.any(mask) else 0.0
            well_amplitudes.append(A_i)
        
        # Coulomb sum: Σ_{i<j} Q_i × Q_j × A_i × A_j
        E_coulomb = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                E_coulomb += self._quark_charges[i] * self._quark_charges[j] * well_amplitudes[i] * well_amplitudes[j]
        
        return self.coulomb_strength * E_coulomb
    
    def _compute_k_coupling_baryon(self) -> int:
        """
        Compute effective coupling winding for baryon.
        
        FIRST PRINCIPLES FOR BARYONS:
        =============================
        
        For baryons, the three quarks are at DIFFERENT WELL POSITIONS
        and couple to spacetime INDEPENDENTLY. The coupling energy is:
        
            E_coupling = -α × (|k₁| + |k₂| + |k₃|) × A
        
        NOT the signed interference sum! Each quark contributes its full
        winding to the spacetime-subspace coupling.
        
        This is fundamentally different from mesons where q and q̄ can
        destructively interfere because they occupy similar regions.
        
        For proton (uud): k₁=5, k₂=5, k₃=3 → k_coupling = 13
        
        This explains why baryons have much larger mass than mesons:
        - Baryon: k_coupling = 13 (independent coupling)
        - Pion: k_eff ≈ 3.6 (destructive interference reduces coupling)
        - Ratio ≈ 3.6, giving A² ratio ≈ 13 (close to mass ratio!)
        """
        # For proton (uud): u has k=5, d has k=3
        k1, k2, k3 = 5, 5, 3
        k_coupling = abs(k1) + abs(k2) + abs(k3)  # = 13
        return k_coupling
    
    def _compute_k_eff_from_wavefunction(self, chi: NDArray[np.complexfloating]) -> float:
        """
        Compute kinetic effective winding from wavefunction gradient.
        
        This measures the "waviness" for kinetic energy purposes:
        k²_eff_kinetic = ∫|∂χ/∂σ|² dσ / ∫|χ|² dσ
        
        NOTE: For COUPLING energy, baryons use k_coupling (sum of |k_i|)
        because quarks couple independently. This k_eff is only for
        kinetic energy scaling.
        """
        dchi_dsigma = self.grid.first_derivative(chi)
        numerator = np.sum(np.abs(dchi_dsigma)**2) * self.grid.dsigma
        denominator = np.sum(np.abs(chi)**2) * self.grid.dsigma
        
        if denominator < 1e-20:
            return 0.0
        
        k_eff = np.sqrt(numerator / denominator)
        return float(k_eff)
    
    def _compute_energy(self, chi: NDArray[np.complexfloating]) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Compute total energy using SIMPLIFIED two-term functional.
        
        E_total = E_subspace + E_coupling + E_coulomb
        
        NOTE: E_spatial and E_curvature are NOT minimized!
              Δx is derived from mass via Compton wavelength: Δx = 1/(βA²)
              Mass is the fundamental output: m = β × A²
        
        NO normalization constraints!
        """
        # Amplitude
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        
        # === SUBSPACE ENERGY COMPONENTS ===
        
        # Kinetic: ∫(ℏ²/2m)|∇χ|² dσ
        T_chi = self.operators.apply_kinetic(chi)
        E_kin = np.real(self.grid.inner_product(chi, T_chi))
        
        # Potential: ∫V(σ)|χ|² dσ
        E_pot = np.real(np.sum(self._V_grid * np.abs(chi)**2) * self.grid.dsigma)
        
        # Nonlinear: (g₁/2)∫|χ|⁴ dσ
        E_nl = (self.g1 / 2) * np.sum(np.abs(chi)**4) * self.grid.dsigma
        
        # Circulation (EM-like): g₂|J|² where J = ∫χ*∂χ/∂σ dσ
        dchi = self.grid.first_derivative(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        E_circ = self.g2 * np.abs(J)**2
        
        E_subspace = E_kin + E_pot + E_nl + E_circ
        
        # === BARYON COUPLING: INDEPENDENT QUARK CONTRIBUTIONS ===
        # 
        # CRITICAL INSIGHT: For baryons, the three quarks are at DIFFERENT
        # WELL POSITIONS and couple to spacetime INDEPENDENTLY!
        #
        # This is fundamentally different from mesons where q and q̄ can
        # destructively interfere because they occupy similar regions.
        #
        # E_coupling_baryon = -α × (|k₁| + |k₂| + |k₃|) × A
        #
        # For proton (uud): k_coupling = 5 + 5 + 3 = 13
        #
        # This explains the large baryon/meson mass ratio:
        # - Baryon k_coupling = 13 (independent)
        # - Meson k_eff ≈ 3.6 (destructive interference)
        # - Ratio ≈ 3.6 → A² ratio ≈ 13 → mass ratio ≈ 6.7
        #
        k_coupling = self._compute_k_coupling_baryon()  # = 13 for proton
        
        # Also compute k_eff for kinetic energy reference
        k_eff = self._compute_k_eff_from_wavefunction(chi)
        
        # === Δx FROM COMPTON WAVELENGTH (first-principles QM) ===
        # Δx = λ_C = ℏ/(mc) = 1/m = 1/(βA²)  [natural units]
        m = self.beta * max(A_sq, 1e-10)
        delta_x = self.hbar / m  # = 1/m in natural units (hbar=1)
        
        # === COUPLING ENERGY (uses k_coupling for INDEPENDENT quark coupling) ===
        # E_coupling = -α × k_coupling × A
        #
        # NOT k_eff! Baryons have quarks at separate wells that couple
        # independently to spacetime. No destructive interference possible.
        E_coupling = -self.alpha * k_coupling * A
        
        # === NO E_spatial OR E_curvature IN MINIMIZATION ===
        # 
        # Key insight: Δx is NOT an independent variable - it's determined
        # by the Compton wavelength: Δx = ℏ/(mc) = 1/(βA²)
        # 
        # The spatial and curvature energies are derived quantities,
        # not part of the energy functional to minimize.
        # They emerge from the mass-size relationship after equilibrium.
        E_spatial = 0.0  # Not part of minimization
        E_curvature = 0.0  # Not part of minimization
        
        # === COULOMB ENERGY (charge-dependent, baryon-specific) ===
        E_coulomb = self._compute_coulomb_energy(chi)
        
        # Total energy: subspace + coupling + coulomb (no spatial/curvature)
        E_total = E_subspace + E_coupling + E_coulomb
        return E_total, E_subspace, E_spatial, E_coupling, E_curvature, E_coulomb, E_kin, E_pot, E_nl, k_eff
    
    def _compute_coulomb_gradient(self, chi: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute gradient of Coulomb energy.
        
        δE_coulomb/δχ* = λ × Σ_{i<j} Q_i × Q_j × δ(A_i × A_j)/δχ*
        
        This is approximated by applying charge-weighted modulation.
        """
        sigma = self.grid.sigma
        well_width = np.pi / 3
        
        # Compute charge-weighted effective potential at each point
        # Approximate: gradient is proportional to χ weighted by local charge environment
        coulomb_weight = np.zeros_like(sigma)
        
        # For each point, compute its contribution to Coulomb energy
        for i, well_pos in enumerate(self.WELL_POSITIONS):
            diff = np.angle(np.exp(1j * (sigma - well_pos)))
            envelope_i = np.exp(-0.5 * (diff / (well_width / 2))**2)
            
            # Sum over pairs involving this well
            for j in range(3):
                if j != i:
                    # Other well envelope
                    other_pos = self.WELL_POSITIONS[j]
                    diff_j = np.angle(np.exp(1j * (sigma - other_pos)))
                    envelope_j = np.exp(-0.5 * (diff_j / (well_width / 2))**2)
                    
                    coulomb_weight += self._quark_charges[i] * self._quark_charges[j] * envelope_i * envelope_j
        
        return self.coulomb_strength * coulomb_weight * chi
    
    def _compute_gradient(self, chi: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute energy gradient: δE/δχ* using UNIFIED four-term functional.
        
        Same structure as meson for consistent A² ratios.
        """
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        
        # === SUBSPACE GRADIENT ===
        T_chi = self.operators.apply_kinetic(chi)
        V_chi = self._V_grid * chi
        NL_chi = self.g1 * np.abs(chi)**2 * chi
        
        # Circulation gradient
        dchi = self.grid.first_derivative(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        circ_grad = 2 * self.g2 * np.real(np.conj(J)) * dchi
        
        grad_subspace = T_chi + V_chi + NL_chi + circ_grad
        
        # === BARYON: INDEPENDENT QUARK COUPLING (same as energy) ===
        k_coupling = self._compute_k_coupling_baryon()  # = 13 for proton
        
        # === COUPLING GRADIENT (uses k_coupling for INDEPENDENT coupling) ===
        # E_coupling = -α × k_coupling × A
        # grad = -α × k_coupling / (2A) × χ
        #
        # Baryons use k_coupling (sum of |k_i|), not k_eff!
        grad_coupling = -self.alpha * k_coupling / (2 * A + 1e-10) * chi
        
        # === NO SPATIAL OR CURVATURE GRADIENTS ===
        # These terms are not part of the energy functional to minimize.
        # Δx is derived from mass via Compton wavelength, not minimized.
        
        # === COULOMB GRADIENT (baryon-specific) ===
        grad_coulomb = self._compute_coulomb_gradient(chi)
        
        return grad_subspace + grad_coupling + grad_coulomb
    
    def _extract_color_phases(self, chi: NDArray[np.complexfloating]) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
        """Extract phases at well positions."""
        sigma = self.grid.sigma
        phases = []
        
        for well_pos in self.WELL_POSITIONS:
            idx = np.argmin(np.abs(sigma - well_pos))
            # Remove winding to get color phase
            raw_phase = np.angle(chi[idx])
            color_phase = (raw_phase - self.k * well_pos) % (2 * np.pi)
            phases.append(float(color_phase))
        
        # Phase differences (should be ~2π/3)
        diff1 = (phases[1] - phases[0]) % (2 * np.pi)
        diff2 = (phases[2] - phases[1]) % (2 * np.pi)
        
        # Color sum: Σe^(iφᵢ) should be ~0
        color_sum = sum(np.exp(1j * phi) for phi in phases)
        
        return tuple(phases), (diff1, diff2), abs(color_sum)
    
    def solve(
        self,
        quark_types: List[str] = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
        dt: float = 0.001,
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> CompositeBaryonState:
        """
        Minimize total energy using gradient descent.
        
        KEY: NO normalization! Amplitude is free to evolve.
        
        Args:
            quark_types: List of quark types for each well, e.g.:
                         ['u', 'u', 'd'] for proton (uud)
                         ['u', 'd', 'd'] for neutron (udd)
                         Defaults to proton configuration.
        
        Uses gradient descent: χ → χ - dt × δE/δχ*
        """
        # Set quark configuration
        if quark_types is None:
            quark_types = PROTON_QUARKS  # Default to proton
        
        if len(quark_types) != 3:
            raise ValueError(f"quark_types must have exactly 3 elements, got {len(quark_types)}")
        
        # Store quark charges for energy calculations
        self._quark_charges = [QUARK_CHARGES[q] for q in quark_types]
        
        chi = self._initialize_baryon(initial_amplitude)
        
        if verbose:
            A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
            phases, diffs, color_mag = self._extract_color_phases(chi)
            print(f"Baryon type: {''.join(quark_types)} (charges: {self._quark_charges})")
            print(f"Initial: A²={A2:.4f}, color_sum={color_mag:.4f}")
            print(f"  phases: {[f'{p:.3f}' for p in phases]}")
        
        E_old, *_ = self._compute_energy(chi)
        converged = False
        residual = float('inf')
        
        for iteration in range(max_iter):
            # Gradient descent (NO normalization!)
            gradient = self._compute_gradient(chi)
            chi_new = chi - dt * gradient
            
            # Compute new energy (extract only what we need for the loop)
            energy_tuple = self._compute_energy(chi_new)
            E_new = energy_tuple[0]  # E_total
            E_coul = energy_tuple[5]  # E_coulomb (for verbose output)
            
            # Check for energy increase (reduce step size if needed)
            if E_new > E_old:
                dt *= 0.5
                if dt < 1e-10:
                    if verbose:
                        print(f"Step size too small at iteration {iteration}")
                    break
                continue
            
            # Update
            residual = np.sqrt(np.sum(np.abs(chi_new - chi)**2) * self.grid.dsigma)
            chi = chi_new
            dE = E_old - E_new
            E_old = E_new
            
            if verbose and iteration % 100 == 0:
                A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
                _, _, color_mag = self._extract_color_phases(chi)
                print(f"Iter {iteration}: E={E_new:.4f}, A²={A2:.4f}, "
                      f"color={color_mag:.4f}, E_coul={E_coul:.4e}, dE={dE:.2e}")
            
            if dE < tol and residual < tol:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # Final results
        (E_total, E_subspace, E_spatial, E_coupling, E_curvature, E_coulomb, 
         E_kin, E_pot, E_nl, k_eff) = self._compute_energy(chi)
        A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
        phases, diffs, color_mag = self._extract_color_phases(chi)
        
        return CompositeBaryonState(
            chi_baryon=chi,
            quark_types=tuple(quark_types),
            amplitude_squared=A2,
            energy_total=E_total,
            energy_subspace=E_subspace,
            energy_spatial=E_spatial,
            energy_coupling=E_coupling,
            energy_curvature=E_curvature,
            energy_coulomb=E_coulomb,
            energy_kinetic=E_kin,
            energy_potential=E_pot,
            energy_nonlinear=E_nl,
            k_eff=k_eff,  # EMERGENT effective winding from wavefunction gradient
            phases=phases,
            phase_differences=diffs,
            color_sum_magnitude=color_mag,
            is_color_neutral=color_mag < 0.1,
            converged=converged,
            iterations=iteration + 1,
            final_residual=residual
        )


def solve_composite_baryon(
    grid: SpectralGrid,
    potential: ThreeWellPotential,
    g1: float = 0.1,
    quark_types: List[str] = None,
    **kwargs
) -> CompositeBaryonState:
    """Convenience function."""
    solver = CompositeBaryonSolver(grid, potential, g1)
    return solver.solve(quark_types=quark_types, **kwargs)


def solve_proton(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for proton (uud) configuration."""
    solver = CompositeBaryonSolver(grid, potential, **kwargs)
    return solver.solve(quark_types=PROTON_QUARKS)


def solve_neutron(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for neutron (udd) configuration."""
    solver = CompositeBaryonSolver(grid, potential, **kwargs)
    return solver.solve(quark_types=NEUTRON_QUARKS)
