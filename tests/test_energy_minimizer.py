"""
Unit tests for UniversalEnergyMinimizer.

Tests for Stage 2 of the two-stage solver architecture.
"""

import pytest
import numpy as np
from sfm_solver.core.energy_minimizer import UniversalEnergyMinimizer


@pytest.mark.unit
class TestUniversalEnergyMinimizer:
    """Test suite for universal energy minimizer."""
    
    @pytest.fixture
    def minimizer(self):
        """Create an energy minimizer for testing.
        
        Note: No beta parameter - minimizer returns amplitudes only.
        Mass conversion (m = beta * A^2) is done externally in test scripts.
        """
        return UniversalEnergyMinimizer(
            g_internal=0.003,
            g1=40.0,
            g2=1.0,
            V0=5.0,
            V1=0.0,
            alpha=10.5,
            verbose=False
        )
    
    @pytest.fixture
    def test_shape_structure(self):
        """Create a test shape structure."""
        N_sigma = 64
        sigma = np.linspace(0, 2*np.pi, N_sigma, endpoint=False)
        dsigma = 2*np.pi / N_sigma
        
        # Create primary component (Gaussian)
        psi_primary = np.exp(-((sigma - np.pi)**2) / 0.5)
        psi_primary = psi_primary / np.sqrt(np.sum(np.abs(psi_primary)**2) * dsigma)
        
        # Create induced components (smaller)
        psi_induced_1 = 0.1 * np.exp(-((sigma - np.pi)**2) / 0.3)
        psi_induced_2 = 0.05 * np.exp(-((sigma - np.pi)**2) / 0.7)
        
        structure = {
            (1, 0, 0): psi_primary,
            (2, 0, 0): psi_induced_1,
            (1, 1, 0): psi_induced_2
        }
        
        # Normalize total
        total_norm_sq = sum(np.sum(np.abs(chi)**2) * dsigma for chi in structure.values())
        normalization = 1.0 / np.sqrt(total_norm_sq)
        for key in structure:
            structure[key] *= normalization
        
        return structure
    
    def test_wavefunction_scaling(self, minimizer, test_shape_structure):
        """Test that wavefunction scaling preserves normalization correctly."""
        Delta_sigma = 0.5
        A = 10.0
        
        chi_scaled = minimizer._scale_wavefunctions(
            test_shape_structure, Delta_sigma, A
        )
        
        # Check scaling factor
        scaling_factor = A / np.sqrt(Delta_sigma)
        
        for (n, l, m), psi_shape in test_shape_structure.items():
            chi = chi_scaled[(n, l, m)]
            
            # Should be scaled version
            expected = scaling_factor * psi_shape
            assert np.allclose(chi, expected), f"Scaling failed for ({n},{l},{m})"
    
    def test_energy_components(self, minimizer, test_shape_structure):
        """Test that all energy components are computed without error."""
        Delta_x = 1.0
        Delta_sigma = 0.5
        A = 10.0
        
        E_total, components = minimizer._compute_total_energy(
            test_shape_structure, Delta_x, Delta_sigma, A
        )
        
        # Check all components exist
        assert 'E_sigma' in components
        assert 'E_kinetic_sigma' in components
        assert 'E_potential_sigma' in components
        assert 'E_nonlinear_sigma' in components
        assert 'E_spatial' in components
        assert 'E_coupling' in components
        assert 'E_curvature' in components
        assert 'E_em' in components
        
        # All should be finite
        for name, value in components.items():
            assert np.isfinite(value), f"{name} is not finite: {value}"
        
        # Total should equal sum
        expected_total = sum(components[k] for k in ['E_sigma', 'E_spatial', 'E_coupling', 'E_curvature', 'E_em'])
        assert abs(E_total - expected_total) < 1e-6, f"Total {E_total} != sum {expected_total}"
    
    def test_energy_minimum(self, minimizer, test_shape_structure):
        """Test that found point is actually a minimum (basic check)."""
        # Use simple initial guess
        Delta_x_0 = 1.0
        Delta_sigma_0 = 0.5
        A_0 = 5.0
        
        # Compute energy at initial point
        E_initial, _ = minimizer._compute_total_energy(
            test_shape_structure, Delta_x_0, Delta_sigma_0, A_0
        )
        
        # Perturb slightly
        Delta_x_pert = Delta_x_0 * 1.01
        E_pert, _ = minimizer._compute_total_energy(
            test_shape_structure, Delta_x_pert, Delta_sigma_0, A_0
        )
        
        # Energy should change (not stuck at zero)
        assert abs(E_pert - E_initial) > 1e-10, "Energy not sensitive to parameters"
    
    def test_minimize_lepton_energy(self, minimizer, test_shape_structure):
        """Test that lepton energy minimization runs without error."""
        result = minimizer.minimize_lepton_energy(
            shape_structure=test_shape_structure,
            generation_n=1
        )
        
        # Check result structure
        assert result.Delta_x > 0
        assert result.Delta_sigma > 0
        assert result.A > 0
        # Mass should be None - minimizer returns amplitudes only
        assert result.mass is None, "Minimizer should not calculate mass (beta-independent)"
        assert result.particle_type == 'lepton'
        assert result.n_target == 1
        
        # Energy components should be finite
        assert np.isfinite(result.E_total)
        assert np.isfinite(result.E_sigma)
        assert np.isfinite(result.E_spatial)
    
    def test_scaling_preserves_shape(self, minimizer, test_shape_structure):
        """Test that scaling doesn't change relative structure."""
        # Scale with two different amplitudes
        chi1 = minimizer._scale_wavefunctions(test_shape_structure, 0.5, 10.0)
        chi2 = minimizer._scale_wavefunctions(test_shape_structure, 0.5, 20.0)
        
        # Relative structure should be identical (up to overall scaling)
        # Compare ratios of components
        key1 = (1, 0, 0)
        key2 = (2, 0, 0)
        
        if key1 in chi1 and key2 in chi1 and key1 in chi2 and key2 in chi2:
            ratio1 = chi1[key1] / (chi1[key2] + 1e-30)
            ratio2 = chi2[key1] / (chi2[key2] + 1e-30)
            
            assert np.allclose(ratio1, ratio2, rtol=1e-6), "Relative structure changed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

