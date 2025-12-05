"""
Tests for the SpectralGrid class.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sfm_solver.core.grid import SpectralGrid


class TestSpectralGridBasics:
    """Test basic grid functionality."""
    
    def test_grid_creation(self):
        """Test that grid is created correctly."""
        grid = SpectralGrid(N=64)
        
        assert grid.N == 64
        assert len(grid.sigma) == 64
        assert grid.sigma[0] == 0
        assert grid.sigma[-1] < 2 * np.pi
        assert_allclose(grid.dsigma, 2 * np.pi / 64)
    
    def test_grid_periodicity(self):
        """Test periodic grid points."""
        grid = SpectralGrid(N=128)
        
        # Grid should not include 2π (same as 0)
        assert grid.sigma[-1] < 2 * np.pi
        
        # Difference between last and first + dsigma should give 2π
        span = grid.sigma[-1] + grid.dsigma
        assert_allclose(span, 2 * np.pi)
    
    def test_small_grid_error(self):
        """Test that small grids raise errors."""
        with pytest.raises(ValueError):
            SpectralGrid(N=2)


class TestSpectralDerivatives:
    """Test derivative operations."""
    
    def test_first_derivative_sin(self):
        """Test first derivative of sin(σ)."""
        grid = SpectralGrid(N=128)
        
        f = np.sin(grid.sigma)
        df = grid.first_derivative(f)
        expected = np.cos(grid.sigma)
        
        assert_allclose(np.real(df), expected, atol=1e-10)
    
    def test_first_derivative_exp(self):
        """Test first derivative of exp(ikσ)."""
        grid = SpectralGrid(N=128)
        
        k = 3
        f = np.exp(1j * k * grid.sigma)
        df = grid.first_derivative(f)
        expected = 1j * k * f
        
        assert_allclose(df, expected, atol=1e-10)
    
    def test_second_derivative_sin(self):
        """Test second derivative of sin(3σ)."""
        grid = SpectralGrid(N=128)
        
        f = np.sin(3 * grid.sigma)
        d2f = grid.second_derivative(f)
        expected = -9 * np.sin(3 * grid.sigma)
        
        assert_allclose(np.real(d2f), expected, atol=1e-10)
    
    def test_laplacian_equals_second_derivative(self):
        """Test that laplacian gives same result as second_derivative."""
        grid = SpectralGrid(N=64)
        
        f = np.cos(2 * grid.sigma) + np.sin(5 * grid.sigma)
        d2f = grid.second_derivative(f)
        lap = grid.laplacian(f)
        
        assert_allclose(d2f, lap)


class TestIntegration:
    """Test integration methods."""
    
    def test_integrate_constant(self):
        """Test integration of constant function."""
        grid = SpectralGrid(N=64)
        
        f = np.ones(grid.N) * 2.5
        integral = grid.integrate(f)
        expected = 2.5 * 2 * np.pi
        
        assert_allclose(integral, expected, rtol=1e-10)
    
    def test_integrate_sin_squared(self):
        """Test integration of sin²(σ)."""
        grid = SpectralGrid(N=128)
        
        f = np.sin(grid.sigma)**2
        integral = grid.integrate(f)
        expected = np.pi  # ∫₀^(2π) sin²(σ) dσ = π
        
        assert_allclose(integral, expected, rtol=1e-10)
    
    def test_integrate_sin(self):
        """Test that ∫sin(σ) = 0 over full period."""
        grid = SpectralGrid(N=64)
        
        f = np.sin(grid.sigma)
        integral = grid.integrate(f)
        
        assert_allclose(integral, 0, atol=1e-10)


class TestNormalization:
    """Test normalization methods."""
    
    def test_norm_of_constant(self):
        """Test L² norm of constant function."""
        grid = SpectralGrid(N=64)
        
        # f = 1 everywhere → ||f||² = 2π, ||f|| = √(2π)
        f = np.ones(grid.N)
        norm = grid.norm(f)
        expected = np.sqrt(2 * np.pi)
        
        assert_allclose(norm, expected, rtol=1e-10)
    
    def test_normalize_function(self):
        """Test normalization of a function."""
        grid = SpectralGrid(N=64)
        
        f = np.sin(grid.sigma)**2
        target_norm = 1.0
        f_norm = grid.normalize(f, target_norm)
        
        assert_allclose(grid.norm(f_norm), target_norm, rtol=1e-10)
    
    def test_normalize_zero_raises(self):
        """Test that normalizing zero function raises error."""
        grid = SpectralGrid(N=64)
        
        f = np.zeros(grid.N)
        with pytest.raises(ValueError):
            grid.normalize(f)


class TestInnerProduct:
    """Test inner product calculations."""
    
    def test_inner_product_orthogonal(self):
        """Test orthogonality of sin and cos."""
        grid = SpectralGrid(N=128)
        
        f = np.sin(grid.sigma)
        g = np.cos(grid.sigma)
        ip = grid.inner_product(f, g)
        
        assert_allclose(ip, 0, atol=1e-10)
    
    def test_inner_product_self(self):
        """Test <f|f> = ||f||²."""
        grid = SpectralGrid(N=64)
        
        f = np.sin(2 * grid.sigma) + np.cos(3 * grid.sigma)
        ip = grid.inner_product(f, f)
        norm_sq = grid.norm(f)**2
        
        assert_allclose(ip, norm_sq, rtol=1e-10)


class TestCirculation:
    """Test circulation and winding number calculations."""
    
    def test_circulation_pure_winding(self):
        """Test circulation for pure exp(ikσ) mode."""
        grid = SpectralGrid(N=128)
        
        k = 3
        chi = np.exp(1j * k * grid.sigma)
        J = grid.circulation(chi)
        
        # J = ∫ χ* dχ/dσ = ∫ exp(-ikσ) × ik × exp(ikσ) = ik × 2π
        expected = 1j * k * 2 * np.pi
        
        assert_allclose(J, expected, rtol=1e-10)
    
    def test_winding_number_k1(self):
        """Test winding number extraction for k=1."""
        grid = SpectralGrid(N=128)
        
        chi = np.exp(1j * grid.sigma)
        k = grid.winding_number(chi)
        
        assert_allclose(k, 1.0, atol=0.01)
    
    def test_winding_number_k5(self):
        """Test winding number extraction for k=5."""
        grid = SpectralGrid(N=128)
        
        chi = np.exp(5j * grid.sigma)
        k = grid.winding_number(chi)
        
        assert_allclose(k, 5.0, atol=0.01)
    
    def test_winding_negative(self):
        """Test negative winding number."""
        grid = SpectralGrid(N=128)
        
        chi = np.exp(-3j * grid.sigma)
        k = grid.winding_number(chi)
        
        assert_allclose(k, -3.0, atol=0.01)


class TestModeCreation:
    """Test helper functions for creating modes."""
    
    def test_winding_mode(self):
        """Test creation of pure winding mode."""
        grid = SpectralGrid(N=64)
        
        mode = grid.create_winding_mode(k=2, amplitude=1.5)
        expected = 1.5 * np.exp(2j * grid.sigma)
        
        assert_allclose(mode, expected)
    
    def test_gaussian_envelope(self):
        """Test Gaussian envelope creation."""
        grid = SpectralGrid(N=128)
        
        env = grid.create_gaussian_envelope(center=0, width=0.5)
        
        # Should peak at center
        idx_center = 0
        assert np.argmax(env) == idx_center
        
        # Should be symmetric around center
        assert_allclose(env[1], env[-1], rtol=1e-10)
    
    def test_localized_mode(self):
        """Test localized mode has correct winding."""
        grid = SpectralGrid(N=128)
        
        mode = grid.create_localized_mode(k=3, center=0, width=0.5)
        k_extracted = grid.winding_number(mode)
        
        assert_allclose(k_extracted, 3.0, atol=0.1)

