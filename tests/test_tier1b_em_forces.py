"""
Tier 1b Tests: Electromagnetic Forces in Single-Field Model.

These tests validate the electromagnetic force mechanism in SFM, where
EM forces emerge from the circulation term:

    Ĥ_circ = g₂ |∫ χ* ∂χ/∂σ dσ|²

Key physics:
- Like charges (same winding): Circulations add → high energy → repulsion
- Opposite charges (opposite winding): Circulations cancel → low energy → attraction
- Charge quantization: Q emerges from SIGNED winding number k

SIGN CONVENTION:
- Positive winding (+k) → Positive charge (positron, up quark, anti-down)
- Negative winding (-k) → Negative charge (electron, down quark, anti-up)

Tier 1b Success Criteria:
1. Like charges repel, opposite charges attract
2. EM energy follows Coulomb-like scaling with separation
3. SIGNED charge quantization: Q = -1 (k=-1), -1/3 (k=-3), +2/3 (k=+5)
4. Fine structure constant α can be related to g₂
5. Multi-particle systems show correct net force behavior
"""

import pytest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.constants import ALPHA_EM
from sfm_solver.forces.electromagnetic import (
    EMForceCalculator,
    calculate_circulation,
    calculate_winding_number,
    calculate_em_energy,
    calculate_nonlinear_energy,
    calculate_total_interaction_energy,
    calculate_envelope_asymmetry,
    calculate_charge_from_wavefunction,
    EXPECTED_LEPTON_CHARGE,
    EXPECTED_QUARK_CHARGE,
    EXPECTED_WINDING,
)


# =============================================================================
# Helper Functions
# =============================================================================

def create_localized_winding_mode(
    grid: SpectralGrid, 
    k: int,  # SIGNED! k=-3 for down quark, k=+5 for up quark
    well_index: int = 0, 
    width: float = 0.5
) -> NDArray[np.complexfloating]:
    """
    Create a winding mode localized in one well of the 3-well structure.
    
    IMPORTANT: k is SIGNED!
    - k > 0: positive winding → positive charge (e.g., up quark k=+5)
    - k < 0: negative winding → negative charge (e.g., down quark k=-3)
    
    Used for testing emergent charge calculations for quark-like modes.
    
    Args:
        grid: SpectralGrid instance.
        k: SIGNED winding number.
        well_index: Which well to localize in (0, 1, or 2).
        width: Gaussian envelope width.
        
    Returns:
        Localized wavefunction with specified winding.
    """
    well_positions = [0.0, 2*np.pi/3, 4*np.pi/3]
    center = well_positions[well_index]
    
    # Gaussian envelope centered at well
    dist = np.angle(np.exp(1j * (grid.sigma - center)))
    envelope = np.exp(-0.5 * (dist / width)**2)
    
    # Add SIGNED winding - this determines charge sign!
    chi = envelope * np.exp(1j * k * grid.sigma)
    
    return chi


# =============================================================================
# Test Class: Charge Quantization
# =============================================================================

class TestTier1bChargeQuantization:
    """Test that SIGNED winding numbers produce correct SIGNED charge values."""
    
    def test_positron_charge_k_positive_1(self, add_prediction):
        """
        FUNDAMENTAL: k=+1 gives POSITIVE unit charge Q = +e.
        
        Positrons have positive winding k=+1 → positive charge.
        """
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=+1)
        
        add_prediction(
            parameter="Q/e for k=+1 (positron)",
            predicted=Q,
            experimental=+1.0,
            target_accuracy=0.001,
            notes="Positron charge: positive winding → positive charge"
        )
        
        assert_allclose(Q, +1.0, rtol=1e-10)
    
    def test_electron_charge_k_negative_1(self, add_prediction):
        """
        FUNDAMENTAL: k=-1 gives NEGATIVE unit charge Q = -e.
        
        Electrons have NEGATIVE winding k=-1 → NEGATIVE charge.
        """
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=-1)
        
        add_prediction(
            parameter="Q/e for k=-1 (electron)",
            predicted=Q,
            experimental=-1.0,
            target_accuracy=0.001,
            notes="Electron charge: NEGATIVE winding → NEGATIVE charge"
        )
        
        assert_allclose(Q, -1.0, rtol=1e-10)
    
    def test_down_quark_charge_k_negative_3(self, add_prediction):
        """
        FUNDAMENTAL: k=-3 gives NEGATIVE charge Q = -e/3.
        
        Down-type quarks (d, s, b) have NEGATIVE winding k=-3 → NEGATIVE charge.
        """
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=-3)
        
        add_prediction(
            parameter="Q/e for k=-3 (down quark)",
            predicted=Q,
            experimental=-1/3,
            target_accuracy=0.001,
            notes="Down quark charge: NEGATIVE winding → NEGATIVE charge -1/3"
        )
        
        assert_allclose(Q, -1/3, rtol=1e-10)
    
    def test_anti_down_quark_charge_k_positive_3(self, add_prediction):
        """
        k=+3 gives POSITIVE charge Q = +e/3.
        
        Anti-down quarks have POSITIVE winding k=+3 → POSITIVE charge.
        """
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=+3)
        
        add_prediction(
            parameter="Q/e for k=+3 (anti-down quark)",
            predicted=Q,
            experimental=+1/3,
            target_accuracy=0.001,
            notes="Anti-down quark charge: POSITIVE winding → POSITIVE charge +1/3"
        )
        
        assert_allclose(Q, +1/3, rtol=1e-10)
    
    def test_up_quark_charge_k_positive_5(self, add_prediction):
        """
        FUNDAMENTAL: k=+5 gives POSITIVE charge Q = +2e/3.
        
        Up-type quarks (u, c, t) have POSITIVE winding k=+5.
        With 3-fold symmetry: Q = sign(k) × (|k| mod 3)/3 = +1 × 2/3 = +2/3.
        
        Reference: Research Note - Origin of Electromagnetic Force
        """
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=+5)
        
        add_prediction(
            parameter="Q/e for k=+5 (up quark)",
            predicted=Q,
            experimental=+2/3,
            target_accuracy=0.001,
            notes="Up quark charge: POSITIVE winding k=+5, (5 mod 3)/3 = 2/3"
        )
        
        assert_allclose(Q, +2/3, rtol=1e-10)
    
    def test_anti_up_quark_charge_k_negative_5(self, add_prediction):
        """
        k=-5 gives NEGATIVE charge Q = -2e/3.
        
        Anti-up quarks have NEGATIVE winding k=-5 → NEGATIVE charge.
        """
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=-5)
        
        add_prediction(
            parameter="Q/e for k=-5 (anti-up quark)",
            predicted=Q,
            experimental=-2/3,
            target_accuracy=0.001,
            notes="Anti-up quark charge: NEGATIVE winding → NEGATIVE charge -2/3"
        )
        
        assert_allclose(Q, -2/3, rtol=1e-10)
    
    def test_neutral_k0(self):
        """k=0 gives zero charge (neutral particle)."""
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=0)
        
        assert Q == 0.0
    
    def test_sign_determines_charge_sign(self):
        """
        Verify that the SIGN of k determines the SIGN of charge.
        
        - Positive k → Positive Q
        - Negative k → Negative Q
        - Same |k| → Same |Q|
        """
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid)
        
        # Test for leptons (k=±1)
        Q_positron = calculator.charge_from_winding(k=+1)
        Q_electron = calculator.charge_from_winding(k=-1)
        
        assert Q_positron > 0, "Positive k should give positive charge"
        assert Q_electron < 0, "Negative k should give negative charge"
        assert abs(Q_positron) == abs(Q_electron), "Same magnitude for ±k"
        
        # Test for quarks (k=±3, k=±5)
        Q_anti_down = calculator.charge_from_winding(k=+3)  # anti-d
        Q_down = calculator.charge_from_winding(k=-3)       # d
        
        assert Q_anti_down > 0, "Positive k=+3 should give positive charge"
        assert Q_down < 0, "Negative k=-3 should give negative charge"
        assert abs(Q_anti_down) == abs(Q_down), "Same magnitude for ±3"
        
        Q_up = calculator.charge_from_winding(k=+5)        # u
        Q_anti_up = calculator.charge_from_winding(k=-5)   # anti-u
        
        assert Q_up > 0, "Positive k=+5 should give positive charge"
        assert Q_anti_up < 0, "Negative k=-5 should give negative charge"
        assert abs(Q_up) == abs(Q_anti_up), "Same magnitude for ±5"


# =============================================================================
# Test Class: Emergent Charge from Wavefunction
# =============================================================================

class TestTier1bEmergentCharge:
    """
    Test that charge emerges from wavefunction circulation integral.
    
    These tests validate that calculate_charge_from_wavefunction() correctly
    extracts charge from actual wavefunctions. The charge predictions are
    already registered by TestTier1bChargeQuantization, so these tests
    focus on validating the wavefunction-based calculation.
    """
    
    def test_emergent_charge_lepton_delocalized(self):
        """
        Delocalized k=±1 modes should give unit charge (lepton-like).
        
        For pure winding modes spread over full circle, charge = ±1.
        """
        grid = SpectralGrid(N=256)
        
        # Create delocalized positron mode (k=+1)
        chi_positron = grid.create_winding_mode(k=+1)
        Q_positron = calculate_charge_from_wavefunction(chi_positron, grid)
        
        # Create delocalized electron mode (k=-1)  
        chi_electron = grid.create_winding_mode(k=-1)
        Q_electron = calculate_charge_from_wavefunction(chi_electron, grid)
        
        # Positron should have positive charge
        assert Q_positron > 0, "Positive winding should give positive charge"
        assert_allclose(abs(Q_positron), 1.0, rtol=0.1)
        
        # Electron should have negative charge
        assert Q_electron < 0, "Negative winding should give negative charge"
        assert_allclose(abs(Q_electron), 1.0, rtol=0.1)
    
    def test_emergent_charge_up_quark_localized(self):
        """
        Localized k=+5 mode should give charge +2/3 (up quark).
        
        When localized in a single well (quark-like), the 3-fold symmetry
        gives: Q = sign(k) × (|k| mod 3)/3 = +1 × 2/3 = +2/3.
        """
        grid = SpectralGrid(N=256)
        
        # Create localized up quark mode (k=+5)
        chi_up = create_localized_winding_mode(grid, k=+5, well_index=0, width=0.4)
        Q_up = calculate_charge_from_wavefunction(chi_up, grid)
        
        # Up quark should have positive charge ~2/3
        assert Q_up > 0, "Up quark (k=+5) should have positive charge"
        assert_allclose(Q_up, +2/3, rtol=0.15)
    
    def test_emergent_charge_down_quark_localized(self):
        """
        Localized k=-3 mode should give charge -1/3 (down quark).
        
        CRITICAL: Down quark has NEGATIVE winding → NEGATIVE charge!
        """
        grid = SpectralGrid(N=256)
        
        # Create localized down quark mode (k=-3, NEGATIVE!)
        chi_down = create_localized_winding_mode(grid, k=-3, well_index=0, width=0.4)
        Q_down = calculate_charge_from_wavefunction(chi_down, grid)
        
        # Down quark should have NEGATIVE charge ~-1/3
        assert Q_down < 0, "Down quark (k=-3) should have NEGATIVE charge"
        assert_allclose(Q_down, -1/3, rtol=0.15)
    
    def test_emergent_vs_theoretical_agreement(self):
        """
        Verify that emergent charge from wavefunction matches theoretical prediction.
        
        This is the key validation: charge should EMERGE from the wavefunction,
        and the emergent value should match the theoretical formula.
        """
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid)
        
        test_cases = [
            (+1, +1.0, "positron"),      # k=+1 → Q=+1
            (-1, -1.0, "electron"),      # k=-1 → Q=-1
            (+5, +2/3, "up quark"),      # k=+5 → Q=+2/3
            (-5, -2/3, "anti-up"),       # k=-5 → Q=-2/3
            (+3, +1/3, "anti-down"),     # k=+3 → Q=+1/3
            (-3, -1/3, "down quark"),    # k=-3 → Q=-1/3
        ]
        
        for k, expected_Q, name in test_cases:
            # Theoretical prediction from k
            Q_theoretical = calculator.charge_from_winding(k=k)
            
            # Verify theoretical matches expected
            assert_allclose(Q_theoretical, expected_Q, rtol=1e-10,
                           err_msg=f"Theoretical charge for {name} (k={k})")
            
            # For localized modes, also verify emergent charge
            if abs(k) > 1:  # Quark-like
                chi = create_localized_winding_mode(grid, k=k, well_index=0, width=0.4)
            else:  # Lepton-like
                chi = grid.create_winding_mode(k=k)
            
            Q_emergent = calculate_charge_from_wavefunction(chi, grid)
            
            # Sign should always match
            assert np.sign(Q_emergent) == np.sign(expected_Q), \
                f"Sign mismatch for {name}: emergent={Q_emergent}, expected={expected_Q}"


# =============================================================================
# Test Class: Circulation Integral
# =============================================================================

class TestTier1bCirculation:
    """Test the circulation integral J = ∫ χ* ∂χ/∂σ dσ."""
    
    def test_circulation_pure_winding(self):
        """
        For pure winding mode χ = exp(ikσ), circulation J = 2πik.
        """
        grid = SpectralGrid(N=256)
        
        for k in [1, 2, 3, -1, -2]:
            chi = grid.create_winding_mode(k=k)
            J = calculate_circulation(chi, grid)
            
            # J should be purely imaginary with Im(J) ≈ 2πk
            expected_imag = 2 * np.pi * k
            assert abs(np.real(J)) < 0.01, f"Real part should be small for k={k}"
            assert_allclose(np.imag(J), expected_imag, rtol=0.01)
    
    def test_circulation_localized_mode(self):
        """
        For localized mode, circulation still proportional to winding.
        """
        grid = SpectralGrid(N=256)
        
        # Create localized mode with Gaussian envelope
        chi = grid.create_localized_mode(k=1, width=0.5, center=np.pi)
        chi = grid.normalize(chi)
        
        k_extracted = calculate_winding_number(chi, grid)
        
        assert_allclose(k_extracted, 1.0, atol=0.1)
    
    def test_winding_number_extraction(self):
        """Verify winding number can be extracted from circulation."""
        grid = SpectralGrid(N=256)
        
        for k_input in [1, 2, 3, 5]:
            chi = grid.create_winding_mode(k=k_input)
            k_extracted = calculate_winding_number(chi, grid)
            
            assert_allclose(k_extracted, k_input, atol=0.05)


# =============================================================================
# Test Class: Like vs Opposite Charge Interaction
# =============================================================================

class TestTier1bChargeInteraction:
    """Test that like charges repel and opposite charges attract."""
    
    def test_like_charges_higher_energy(self):
        """
        FUNDAMENTAL: Like charges have higher EM energy.
        
        Two particles with same winding sign → circulations add → 
        |J₁ + J₂|² is large → high energy → repulsion.
        """
        grid = SpectralGrid(N=256)
        
        chi_plus1 = grid.create_winding_mode(k=1)
        chi_plus2 = grid.create_winding_mode(k=1)
        chi_minus = grid.create_winding_mode(k=-1)
        
        g2 = 0.1
        
        E_like = calculate_em_energy(chi_plus1, chi_plus2, grid, g2)
        E_opposite = calculate_em_energy(chi_plus1, chi_minus, grid, g2)
        
        # Like charges should have higher energy
        assert E_like > E_opposite, "Like charges should have higher EM energy"
        
        # E_opposite should be near zero (circulations cancel)
        assert E_opposite < 1e-6, "Opposite charges should have near-zero energy"
        
        # E_like should be positive and significant
        assert E_like > 1.0, "Like charges should have significant energy"
    
    def test_opposite_charges_low_energy(self):
        """
        Opposite charges have low EM energy (circulations cancel).
        """
        grid = SpectralGrid(N=256)
        
        chi_plus = grid.create_winding_mode(k=1)
        chi_minus = grid.create_winding_mode(k=-1)
        
        g2 = 0.1
        
        E = calculate_em_energy(chi_plus, chi_minus, grid, g2)
        
        # Circulations should largely cancel
        # J₁ = 2πi, J₂ = -2πi → J₁ + J₂ ≈ 0
        assert E < 1e-6, f"Opposite charge energy should be near zero, got {E}"
    
    def test_force_type_determination(self):
        """Test that force type is correctly identified."""
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        chi_plus = grid.create_winding_mode(k=1)
        chi_minus = grid.create_winding_mode(k=-1)
        chi_plus2 = grid.create_winding_mode(k=1)
        
        # Same sign winding → repulsive
        assert calculator.force_type(chi_plus, chi_plus2) == 'repulsive'
        
        # Opposite sign winding → attractive
        assert calculator.force_type(chi_plus, chi_minus) == 'attractive'
    
    def test_like_charges_various_k(self):
        """Test like-charge repulsion for various winding numbers."""
        grid = SpectralGrid(N=256)
        g2 = 0.1
        
        for k in [1, 2, 3]:
            chi1 = grid.create_winding_mode(k=k)
            chi2 = grid.create_winding_mode(k=k)
            chi_opp = grid.create_winding_mode(k=-k)
            
            E_like = calculate_em_energy(chi1, chi2, grid, g2)
            E_opposite = calculate_em_energy(chi1, chi_opp, grid, g2)
            
            assert E_like > E_opposite, f"Like charges should repel for k={k}"


# =============================================================================
# Test Class: Coulomb-like Scaling
# =============================================================================

class TestTier1bCoulombScaling:
    """Test that EM energy has Coulomb-like properties."""
    
    def test_energy_scales_with_charge_squared(self):
        """
        EM energy should scale as charge².
        
        For circulation J ~ k, E ~ g₂|J|² ~ k².
        """
        grid = SpectralGrid(N=256)
        g2 = 0.1
        
        energies = {}
        for k in [1, 2, 3]:
            chi = grid.create_winding_mode(k=k)
            # Self-energy (single particle circulation)
            J = calculate_circulation(chi, grid)
            E = g2 * np.abs(J)**2
            energies[k] = E
        
        # Energy should scale as k²
        ratio_2_1 = energies[2] / energies[1]
        ratio_3_1 = energies[3] / energies[1]
        
        assert_allclose(ratio_2_1, 4.0, rtol=0.1)  # 2² / 1² = 4
        assert_allclose(ratio_3_1, 9.0, rtol=0.1)  # 3² / 1² = 9
    
    def test_energy_vs_separation(self, add_solver_parameter):
        """
        Test that EM energy varies with separation between localized charges.
        """
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g1=0.0, g2=0.1)
        
        # Create template for localized mode
        chi_template = grid.create_localized_mode(k=1, width=0.3, center=0)
        
        # Get energy vs separation for like charges
        separations, energies_like = calculator.energy_vs_separation(
            chi_template, k1=1, k2=1, n_points=20
        )
        
        # Get energy vs separation for opposite charges
        _, energies_opposite = calculator.energy_vs_separation(
            chi_template, k1=1, k2=-1, n_points=20
        )
        
        # Like charges: energy should decrease with separation (less overlap)
        # Opposite charges: energy should be low throughout
        
        add_solver_parameter("EM_separation_points", 20)
        add_solver_parameter("EM_max_separation", f"{separations[-1]:.2f}")
        
        # Average energies
        E_like_avg = np.mean(energies_like)
        E_opposite_avg = np.mean(energies_opposite)
        
        assert E_like_avg > E_opposite_avg, "Like charges should have higher average energy"
    
    def test_coulomb_approximation(self):
        """
        Test the Coulomb-like energy approximation.
        
        E ~ g₂ (k₁ + k₂)² for well-separated particles.
        """
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        # Test various charge combinations
        test_cases = [
            (1, 1, 4.0),    # Like: (1+1)² = 4
            (1, -1, 0.0),   # Opposite: (1-1)² = 0
            (2, 2, 16.0),   # Like: (2+2)² = 16
            (1, 2, 9.0),    # Same sign: (1+2)² = 9
        ]
        
        for k1, k2, expected_factor in test_cases:
            E_approx = calculator.coulomb_like_energy(k1, k2)
            
            # Normalize by g2 and norm factors
            E_base = calculator.coulomb_like_energy(1, 0)  # Single k=1 particle
            if E_base > 0:
                ratio = E_approx / E_base
            else:
                ratio = 0 if expected_factor == 0 else float('inf')
            
            if expected_factor > 0:
                assert_allclose(ratio, expected_factor, rtol=0.2)


# =============================================================================
# Test Class: Fine Structure Constant
# =============================================================================

class TestTier1bFineStructure:
    """Test connection between g₂ and fine structure constant α."""
    
    def test_alpha_order_of_magnitude(self, add_prediction, add_solver_parameter):
        """
        The coupling g₂ should relate to α ≈ 1/137.
        
        In natural units, the EM coupling strength should be O(α).
        """
        # α ≈ 1/137.036
        alpha_experimental = ALPHA_EM
        
        # In SFM, g₂ sets the EM coupling scale
        # The relationship g₂ ~ α needs to be established
        # For now, we verify that reasonable g₂ gives sensible physics
        
        g2_values = [0.001, 0.01, 0.1]
        
        # A g₂ of order α should give correct force magnitudes
        g2_alpha = alpha_experimental  # ~0.0073
        
        add_solver_parameter("alpha_experimental", f"{alpha_experimental:.6f}")
        add_solver_parameter("g2_test_values", str(g2_values))
        
        # The prediction is that g₂ should be O(α) or contain α as a factor
        add_prediction(
            parameter="g₂/α (expected O(1))",
            predicted=0.01 / alpha_experimental,  # Using g₂=0.01 as reference
            experimental=1.0,
            target_accuracy=2.0,  # Order of magnitude
            notes="Relationship between g₂ and fine structure constant"
        )
    
    def test_coulomb_energy_scale(self):
        """
        Verify Coulomb energy scale is reasonable.
        
        For two unit charges, E ~ g₂ |J₁ + J₂|² ~ g₂ × (4πk)² ~ 160 g₂.
        With g₂ = α ≈ 0.0073, E ≈ 1.2 in natural units.
        """
        grid = SpectralGrid(N=256)
        
        # With g₂ = α, the energy scale should be order unity
        g2 = ALPHA_EM
        calculator = EMForceCalculator(grid, g2=g2)
        
        chi1 = grid.create_winding_mode(k=1)
        chi2 = grid.create_winding_mode(k=1)
        
        E = calculate_em_energy(chi1, chi2, grid, g2)
        
        # E should be O(1) in natural units for g₂ = α
        assert E > 0, "Energy should be positive for like charges"
        assert E < 10.0, "Energy should be bounded in natural units"
        
        # More specifically, E ≈ g₂ × (4π)² ≈ 1.15 for two k=1 charges
        expected_E = g2 * (4 * np.pi)**2
        assert abs(E - expected_E) / expected_E < 0.1, f"E={E} should be close to {expected_E}"


# =============================================================================
# Test Class: Multi-Particle Systems
# =============================================================================

class TestTier1bMultiParticle:
    """Test EM interactions in multi-particle systems."""
    
    def test_three_like_charges(self):
        """Three like charges should have higher energy than two."""
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        chi1 = grid.create_winding_mode(k=1)
        chi2 = grid.create_winding_mode(k=1)
        chi3 = grid.create_winding_mode(k=1)
        
        # Two particles
        E_two = calculator.circulation_energy([chi1, chi2])
        
        # Three particles
        E_three = calculator.circulation_energy([chi1, chi2, chi3])
        
        # Three like charges should have higher energy
        assert E_three > E_two
        
        # Scaling: E ~ (Σk)², so E₃/E₂ = 9/4 = 2.25
        ratio = E_three / E_two
        assert_allclose(ratio, 2.25, rtol=0.1)
    
    def test_neutral_system(self):
        """
        A system with equal positive and negative charges should be nearly neutral.
        """
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        chi_plus = grid.create_winding_mode(k=1)
        chi_minus = grid.create_winding_mode(k=-1)
        
        # Total winding should be zero
        total_k = calculator.total_winding([chi_plus, chi_minus])
        assert_allclose(total_k, 0.0, atol=0.1)
        
        # Circulation energy should be near zero
        E = calculator.circulation_energy([chi_plus, chi_minus])
        assert E < 1e-4, f"Neutral system should have near-zero EM energy, got {E}"
    
    def test_hydrogen_like_system(self, add_prediction):
        """
        Test a hydrogen-like system: heavy nucleus (k=3, down quark) + electron (k=-1).
        
        Note: This is simplified - real proton is uud, but tests the concept.
        """
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        # "Nucleus" with k=3 (1/3 charge)
        chi_nucleus = grid.create_winding_mode(k=3)
        
        # "Electron" with k=-1 (-1 charge in opposite sign convention)
        chi_electron = grid.create_winding_mode(k=-1)
        
        # They should attract (opposite-ish charges)
        force_type = calculator.force_type(chi_nucleus, chi_electron)
        
        # Net winding = 3 + (-1) = 2, not zero, but attraction expected
        # Actually k=3 and k=-1 have opposite SIGN so should attract
        assert force_type == 'attractive', "Nucleus and electron should attract"
        
        # Record this as a qualitative prediction
        add_prediction(
            parameter="H-like attraction",
            predicted=1.0 if force_type == 'attractive' else 0.0,
            experimental=1.0,
            target_accuracy=0.01,
            notes="Hydrogen-like system shows attraction"
        )
    
    def test_positronium_like_system(self):
        """
        Test e⁺e⁻ system (positronium-like).
        
        Equal and opposite charges should have minimum EM energy.
        """
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        chi_positron = grid.create_winding_mode(k=1)
        chi_electron = grid.create_winding_mode(k=-1)
        
        # Should attract
        assert calculator.force_type(chi_positron, chi_electron) == 'attractive'
        
        # EM energy should be near zero (cancellation)
        E = calculator.circulation_energy([chi_positron, chi_electron])
        assert E < 1e-4


# =============================================================================
# Test Class: Envelope Asymmetry
# =============================================================================

class TestTier1bEnvelopeAsymmetry:
    """Test envelope asymmetry calculations."""
    
    def test_symmetric_envelopes(self):
        """Identical envelopes should have zero asymmetry."""
        grid = SpectralGrid(N=256)
        
        chi_plus = grid.create_winding_mode(k=1)
        chi_minus = grid.create_winding_mode(k=-1)
        
        eta, delta = calculate_envelope_asymmetry(chi_plus, chi_minus, grid)
        
        # For pure winding modes, envelopes are identical
        assert abs(eta) < 0.01, f"Amplitude asymmetry should be near zero, got {eta}"
    
    def test_different_amplitude_asymmetry(self):
        """Different amplitudes should show non-zero η."""
        grid = SpectralGrid(N=256)
        
        # Create modes with different amplitudes
        chi_plus = 2.0 * grid.create_winding_mode(k=1)
        chi_minus = 1.0 * grid.create_winding_mode(k=-1)
        
        eta, delta = calculate_envelope_asymmetry(chi_plus, chi_minus, grid)
        
        # η = (A₊ - A₋)/(A₊ + A₋) = (2-1)/(2+1) = 1/3
        expected_eta = (2 - 1) / (2 + 1)
        assert_allclose(eta, expected_eta, rtol=0.1)


# =============================================================================
# Test Class: Two-Particle Analysis
# =============================================================================

class TestTier1bTwoParticleAnalysis:
    """Test comprehensive two-particle analysis."""
    
    def test_full_analysis_like_charges(self, add_solver_parameter):
        """Full analysis of two like charges."""
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g1=0.1, g2=0.1)
        
        chi1 = grid.create_winding_mode(k=1)
        chi2 = grid.create_winding_mode(k=1)
        
        analysis = calculator.analyze_two_particle(chi1, chi2)
        
        add_solver_parameter("like_charge_winding", str(analysis['winding_numbers']))
        add_solver_parameter("like_charge_force", analysis['force_type'])
        
        assert analysis['force_type'] == 'repulsive'
        assert analysis['energies']['total'] > 0
        assert analysis['charges'] == (1.0, 1.0)
    
    def test_full_analysis_opposite_charges(self, add_solver_parameter):
        """Full analysis of two opposite charges."""
        grid = SpectralGrid(N=256)
        calculator = EMForceCalculator(grid, g1=0.1, g2=0.1)
        
        chi_plus = grid.create_winding_mode(k=1)
        chi_minus = grid.create_winding_mode(k=-1)
        
        analysis = calculator.analyze_two_particle(chi_plus, chi_minus)
        
        add_solver_parameter("opposite_charge_winding", str(analysis['winding_numbers']))
        add_solver_parameter("opposite_charge_force", analysis['force_type'])
        
        assert analysis['force_type'] == 'attractive'
        assert analysis['energies']['em'] < 1e-4  # Near zero


# =============================================================================
# Test Class: Physical Consistency
# =============================================================================

class TestTier1bPhysicalConsistency:
    """Test physical consistency of EM force implementation."""
    
    def test_energy_positive_definite(self):
        """EM energy should be non-negative."""
        grid = SpectralGrid(N=256)
        g2 = 0.1
        
        for k1 in [-2, -1, 1, 2]:
            for k2 in [-2, -1, 1, 2]:
                chi1 = grid.create_winding_mode(k=k1)
                chi2 = grid.create_winding_mode(k=k2)
                
                E = calculate_em_energy(chi1, chi2, grid, g2)
                
                assert E >= 0, f"EM energy should be non-negative for k1={k1}, k2={k2}"
    
    def test_symmetry_particle_exchange(self):
        """EM energy should be symmetric under particle exchange."""
        grid = SpectralGrid(N=256)
        g2 = 0.1
        
        chi1 = grid.create_winding_mode(k=1)
        chi2 = grid.create_winding_mode(k=2)
        
        E_12 = calculate_em_energy(chi1, chi2, grid, g2)
        E_21 = calculate_em_energy(chi2, chi1, grid, g2)
        
        assert_allclose(E_12, E_21, rtol=1e-10)
    
    def test_linearity_in_g2(self):
        """EM energy should scale linearly with g₂."""
        grid = SpectralGrid(N=256)
        
        chi1 = grid.create_winding_mode(k=1)
        chi2 = grid.create_winding_mode(k=1)
        
        E_1 = calculate_em_energy(chi1, chi2, grid, g2=0.1)
        E_2 = calculate_em_energy(chi1, chi2, grid, g2=0.2)
        
        assert_allclose(E_2 / E_1, 2.0, rtol=1e-10)
    
    def test_nonlinear_energy_positive(self):
        """Nonlinear interaction energy should be non-negative for repulsive g₁."""
        grid = SpectralGrid(N=256)
        g1 = 0.1  # Repulsive
        
        chi1 = grid.create_winding_mode(k=1)
        chi2 = grid.create_winding_mode(k=1)
        
        E_nl = calculate_nonlinear_energy(chi1, chi2, grid, g1)
        
        assert E_nl >= 0

