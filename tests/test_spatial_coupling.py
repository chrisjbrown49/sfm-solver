"""
Unit tests for SpatialCouplingBuilder.

Tests for spatial coupling in Stage 1 of the two-stage architecture.
"""

import pytest
import numpy as np
from sfm_solver.core.spatial_coupling import SpatialCouplingBuilder, SpatialState


@pytest.mark.unit
class TestSpatialCouplingBuilder:
    """Test suite for spatial coupling builder."""
    
    @pytest.fixture
    def builder(self):
        """Create a spatial coupling builder for testing."""
        return SpatialCouplingBuilder(
            alpha_dimensionless=2.1,  # alpha=10.5, V0=5.0
            n_max=3,
            l_max=2,
            verbose=False
        )
    
    @pytest.fixture
    def test_shape(self):
        """Create a test subspace shape (Gaussian)."""
        N_sigma = 64
        sigma = np.linspace(0, 2*np.pi, N_sigma, endpoint=False)
        dsigma = 2*np.pi / N_sigma
        
        # Gaussian shape
        psi = np.exp(-((sigma - np.pi)**2) / 0.5)
        
        # Normalize
        norm_sq = np.sum(np.abs(psi)**2) * dsigma
        psi = psi / np.sqrt(norm_sq)
        
        return psi
    
    def test_4d_structure_normalization(self, builder, test_shape):
        """Test that 4D structure is normalized to integral_Sigma |chi_nlm|^2 = 1."""
        structure = builder.build_4d_structure(
            subspace_shape=test_shape,
            n_target=1,
            l_target=0,
            m_target=0
        )
        
        # Compute total norm
        N_sigma = len(test_shape)
        dsigma = 2*np.pi / N_sigma
        
        total_norm_sq = 0.0
        for (n, l, m), chi in structure.items():
            norm_sq = np.sum(np.abs(chi)**2) * dsigma
            total_norm_sq += norm_sq
        
        # Should be normalized to 1
        assert abs(total_norm_sq - 1.0) < 1e-8, f"Total norm is {total_norm_sq}, expected 1.0"
    
    def test_induced_components(self, builder, test_shape):
        """Test that induced components are created."""
        structure = builder.build_4d_structure(
            subspace_shape=test_shape,
            n_target=1,
            l_target=0,
            m_target=0
        )
        
        # Should have primary component
        assert (1, 0, 0) in structure
        
        # Should have at least a few components
        assert len(structure) > 1, "No induced components created"
        
        # Primary component should dominate
        primary_norm = np.sum(np.abs(structure[(1, 0, 0)])**2) * (2*np.pi / len(test_shape))
        assert primary_norm > 0.5, f"Primary component norm too small: {primary_norm}"
    
    def test_coupling_matrix(self, builder):
        """Test that coupling matrix is computed."""
        R = builder.R_coupling
        
        # Should be square matrix
        assert R.shape[0] == R.shape[1]
        assert R.shape[0] == builder.N_spatial
        
        # Should have some non-zero elements
        assert np.max(np.abs(R)) > 0, "Coupling matrix is all zeros"
    
    def test_spatial_states(self, builder):
        """Test that spatial states are properly constructed."""
        # Should have states for n=1,2,3 with appropriate l values
        assert builder.N_spatial > 0
        
        # Check some specific states exist
        state_100 = SpatialState(n=1, l=0, m=0)
        assert state_100 in builder.spatial_states
        
        # n=2 should have l=0,1
        state_200 = SpatialState(n=2, l=0, m=0)
        state_210 = SpatialState(n=2, l=1, m=0)
        assert state_200 in builder.spatial_states
        assert state_210 in builder.spatial_states
    
    def test_energy_denominators(self, builder, test_shape):
        """Test that energy denominators are reasonable."""
        structure = builder.build_4d_structure(
            subspace_shape=test_shape,
            n_target=1,
            l_target=0,
            m_target=0
        )
        
        # All induced components should exist
        for (n, l, m), chi in structure.items():
            assert chi is not None
            assert len(chi) == len(test_shape)
            
            # Check for NaN or Inf
            assert not np.any(np.isnan(chi)), f"NaN in component ({n},{l},{m})"
            assert not np.any(np.isinf(chi)), f"Inf in component ({n},{l},{m})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

