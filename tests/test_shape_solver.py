"""
Unit tests for DimensionlessShapeSolver.

Tests for Stage 1 of the two-stage solver architecture.
"""

import pytest
import numpy as np
from sfm_solver.core.shape_solver import DimensionlessShapeSolver


@pytest.mark.unit
class TestDimensionlessShapeSolver:
    """Test suite for dimensionless shape solver."""
    
    @pytest.fixture
    def solver(self):
        """Create a shape solver instance for testing."""
        return DimensionlessShapeSolver(
            g1_dimensionless=8.0,  # g1=40, V0=5.0
            g2_dimensionless=0.2,  # g2=1.0, V0=5.0
            V0=5.0,
            V1=0.0,
            N_sigma=64,
            verbose=False
        )
    
    def test_shape_normalization_lepton(self, solver):
        """Test that lepton shape is properly normalized to integral|psi|^2 = 1."""
        result = solver.solve_lepton_shape(winding_k=1, generation_n=1, max_iter=100)
        
        # Compute norm
        psi = result.composite_shape
        dsigma = 2*np.pi / len(psi)
        norm_sq = np.sum(np.abs(psi)**2) * dsigma
        
        # Should be normalized to 1
        assert abs(norm_sq - 1.0) < 1e-8, f"Norm is {norm_sq}, expected 1.0"
    
    def test_shape_normalization_baryon(self, solver):
        """Test that baryon shape is properly normalized."""
        result = solver.solve_baryon_shape(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            max_iter=100
        )
        
        # Compute norm of composite
        psi = result.composite_shape
        dsigma = 2*np.pi / len(psi)
        norm_sq = np.sum(np.abs(psi)**2) * dsigma
        
        # Should be normalized to 1
        assert abs(norm_sq - 1.0) < 1e-8, f"Norm is {norm_sq}, expected 1.0"
        
        # Individual quarks should also be normalized
        for i, chi in enumerate([result.chi1_shape, result.chi2_shape, result.chi3_shape]):
            norm_sq_i = np.sum(np.abs(chi)**2) * dsigma
            assert abs(norm_sq_i - 1.0) < 1e-8, f"Quark {i+1} norm is {norm_sq_i}, expected 1.0"
    
    def test_em_in_hamiltonian(self, solver):
        """Test that EM potential affects eigenstate structure."""
        # Solve with EM
        result_with_em = solver.solve_lepton_shape(winding_k=1, generation_n=1, max_iter=100)
        
        # Solve without EM (g2=0)
        solver_no_em = DimensionlessShapeSolver(
            g1_dimensionless=8.0,
            g2_dimensionless=0.0,  # No EM
            V0=5.0,
            V1=0.0,
            N_sigma=64,
            verbose=False
        )
        result_no_em = solver_no_em.solve_lepton_shape(winding_k=1, generation_n=1, max_iter=100)
        
        # Shapes should differ due to EM
        psi_em = result_with_em.composite_shape
        psi_no_em = result_no_em.composite_shape
        
        diff = np.sum(np.abs(psi_em - psi_no_em)**2) * (2*np.pi / len(psi_em))
        
        # Should be significantly different
        assert diff > 1e-6, f"EM effect is too small: {diff}"
    
    def test_dimensionless_couplings(self, solver):
        """Test that all energies are properly dimensionless ratios."""
        # The solver should work with dimensionless couplings
        assert solver.g1_dimless == 8.0
        assert solver.g2_dimless == 0.2
        
        # V_well should be dimensionless (normalized by V0)
        assert np.max(solver.V_well_dimless) < 10.0  # Reasonable range
        assert np.min(solver.V_well_dimless) >= 0.0
    
    def test_scf_convergence_baryon(self, solver):
        """Test that SCF converges for baryon."""
        result = solver.solve_baryon_shape(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            max_iter=500,
            tol=1e-4
        )
        
        # Should converge
        assert result.converged, "SCF did not converge"
        assert result.iterations < 500, f"Used all {result.iterations} iterations"
    
    def test_lepton_convergence(self, solver):
        """Test that lepton solver converges."""
        result = solver.solve_lepton_shape(winding_k=1, generation_n=1, max_iter=200)
        
        assert result.converged
        assert result.particle_type == 'lepton'
        assert result.winding_k == 1
        assert result.generation_n == 1
    
    def test_meson_convergence(self, solver):
        """Test that meson solver converges."""
        result = solver.solve_meson_shape(
            quark_winding=5,
            antiquark_winding=-5,
            max_iter=300
        )
        
        # Should have two components
        assert result.chi1_shape is not None
        assert result.chi2_shape is not None
        assert result.composite_shape is not None
        assert result.particle_type == 'meson'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

