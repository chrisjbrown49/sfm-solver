"""
Tests for potential functions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.legacy.effective import EffectivePotential, SpinOrbitPotential


class TestThreeWellPotential:
    """Test the three-well potential."""
    
    def test_creation(self):
        """Test potential creation."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        assert pot.V0 == 1.0
        assert pot.V1 == 0.1
    
    def test_negative_V0_raises(self):
        """Test that negative V0 raises error."""
        with pytest.raises(ValueError):
            ThreeWellPotential(V0=-1.0)
    
    def test_negative_V1_raises(self):
        """Test that negative V1 raises error."""
        with pytest.raises(ValueError):
            ThreeWellPotential(V0=1.0, V1=-0.1)
    
    def test_well_positions(self):
        """Test well positions are at 0, 2π/3, 4π/3."""
        pot = ThreeWellPotential()
        positions = pot.well_positions
        
        expected = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        assert_allclose(positions, expected)
    
    def test_minimum_at_wells(self):
        """Test that potential is minimum at well positions."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        for sigma_well in pot.well_positions:
            V_well = pot(sigma_well)
            assert_allclose(V_well, 0, atol=1e-10)
    
    def test_barrier_positions(self):
        """Test barrier positions."""
        pot = ThreeWellPotential()
        barriers = pot.barrier_positions
        
        expected = np.array([np.pi / 3, np.pi, 5 * np.pi / 3])
        assert_allclose(barriers, expected)
    
    def test_potential_maximum(self):
        """Test that potential has maximum at barriers."""
        pot = ThreeWellPotential(V0=1.0, V1=0.0)
        
        # With V1=0, max is exactly 2V0 at barriers
        for sigma_barrier in pot.barrier_positions:
            V_barrier = pot(sigma_barrier)
            assert_allclose(V_barrier, 2.0, atol=1e-10)
    
    def test_periodicity(self):
        """Test that potential is 2π-periodic."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        sigma = np.linspace(0, 2 * np.pi, 100)
        V1 = pot(sigma)
        V2 = pot(sigma + 2 * np.pi)
        
        assert_allclose(V1, V2, atol=1e-10)
    
    def test_derivative_at_wells(self):
        """Test that derivative is zero at well positions."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        for sigma_well in pot.well_positions:
            dV = pot.derivative(sigma_well)
            assert_allclose(dV, 0, atol=1e-10)
    
    def test_second_derivative_at_wells(self):
        """Test curvature at well positions."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        # At wells, d²V/dσ² = 9V₀ + 36V₁
        expected = 9 * 1.0 + 36 * 0.1
        
        for sigma_well in pot.well_positions:
            d2V = pot.second_derivative(sigma_well)
            assert_allclose(d2V, expected, atol=1e-10)
    
    def test_well_curvature(self):
        """Test well_curvature method."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        curvature = pot.well_curvature(0)
        expected = 9 * 1.0 + 36 * 0.1
        
        assert_allclose(curvature, expected, atol=1e-10)
    
    def test_array_evaluation(self):
        """Test that potential can be evaluated on arrays."""
        pot = ThreeWellPotential(V0=1.0, V1=0.0)
        
        sigma = np.array([0, np.pi / 3, np.pi])
        V = pot(sigma)
        
        assert len(V) == 3
        assert_allclose(V[0], 0)  # Well
        assert_allclose(V[1], 2.0)  # Barrier
        assert_allclose(V[2], 2.0)  # Barrier


class TestSpinOrbitPotential:
    """Test the spin-orbit coupling."""
    
    def test_creation(self):
        """Test spin-orbit creation."""
        so = SpinOrbitPotential(lambda_so=0.2)
        assert so.lambda_so == 0.2
    
    def test_negative_raises(self):
        """Test that negative coupling raises error."""
        with pytest.raises(ValueError):
            SpinOrbitPotential(lambda_so=-0.1)
    
    def test_positive_charge(self):
        """Test spin-orbit for positive charge."""
        so = SpinOrbitPotential(lambda_so=0.1)
        
        # V_so,+ = -λk for positive charge
        V = so(k=3, charge_sign=+1)
        assert_allclose(V, -0.3)
    
    def test_negative_charge(self):
        """Test spin-orbit for negative charge."""
        so = SpinOrbitPotential(lambda_so=0.1)
        
        # V_so,- = +λk for negative charge
        V = so(k=3, charge_sign=-1)
        assert_allclose(V, +0.3)


class TestEffectivePotential:
    """Test the effective potential with spin-orbit."""
    
    def test_creation_from_params(self):
        """Test creation from parameters."""
        eff = EffectivePotential(V0=1.0, V1=0.1, lambda_so=0.2)
        
        assert eff.V0 == 1.0
        assert eff.V1 == 0.1
        assert eff.lambda_so == 0.2
    
    def test_creation_from_potential(self):
        """Test creation from existing potential."""
        base = ThreeWellPotential(V0=1.5, V1=0.2)
        eff = EffectivePotential(base_potential=base, lambda_so=0.1)
        
        assert eff.V0 == 1.5
        assert eff.V1 == 0.2
    
    def test_at_well_with_spin_orbit(self):
        """Test effective potential at well with spin-orbit."""
        eff = EffectivePotential(V0=1.0, V1=0.0, lambda_so=0.1)
        
        # At well, base V=0
        # For positive charge k=1: V_eff = 0 - 0.1 × 1 = -0.1
        V_plus = eff.for_positive_charge(0, k=1)
        assert_allclose(V_plus, -0.1)
        
        # For negative charge k=1: V_eff = 0 + 0.1 × 1 = +0.1
        V_minus = eff.for_negative_charge(0, k=1)
        assert_allclose(V_minus, +0.1)
    
    def test_spinor_potentials(self):
        """Test spinor component potentials."""
        eff = EffectivePotential(V0=1.0, V1=0.0, lambda_so=0.1)
        
        V_plus, V_minus = eff.spinor_potentials(0, k=2)
        
        # At well: V = 0
        # V_+ = 0 - λk = -0.2
        # V_- = 0 + λk = +0.2
        assert_allclose(V_plus, -0.2)
        assert_allclose(V_minus, +0.2)
    
    def test_energy_splitting(self):
        """Test spin-orbit energy splitting."""
        eff = EffectivePotential(lambda_so=0.1)
        
        splitting = eff.energy_splitting(k=3)
        expected = 2 * 0.1 * 3  # 2λk
        
        assert_allclose(splitting, expected)
    
    def test_well_positions(self):
        """Test that effective potential has same well positions."""
        eff = EffectivePotential(V0=1.0, V1=0.1)
        
        expected = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        assert_allclose(eff.well_positions, expected)

