"""
Tests for SpectralOperators - Core FFT-based infrastructure.

SpectralOperators provides the FFT-based differentiation and
Hamiltonian operations used by all physics-based solvers.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.eigensolver.spectral import SpectralOperators


class TestSpectralOperators:
    """Test spectral operators."""
    
    def test_kinetic_energy_coefficient(self):
        """Test kinetic energy coefficient in Fourier space."""
        grid = SpectralGrid(N=64)
        ops = SpectralOperators(grid, m_eff=1.0, hbar=1.0)
        
        # For k=0 mode, kinetic energy should be 0
        assert_allclose(ops.kinetic_fourier[0], 0)
        
        # For k=1 mode, T = ℏ²k²/(2m) = 0.5
        idx_k1 = 1  # k=1 is at index 1
        assert_allclose(ops.kinetic_fourier[idx_k1], 0.5, rtol=1e-10)
    
    def test_apply_kinetic_plane_wave(self):
        """Test kinetic energy applied to plane wave."""
        grid = SpectralGrid(N=64)
        ops = SpectralOperators(grid, m_eff=1.0, hbar=1.0)
        
        # ψ = exp(ikσ) → T ψ = (ℏ²k²/2m) ψ
        k = 3
        psi = np.exp(1j * k * grid.sigma)
        T_psi = ops.apply_kinetic(psi)
        
        expected_eigenvalue = 0.5 * k**2  # ℏ=m=1
        expected = expected_eigenvalue * psi
        
        assert_allclose(T_psi, expected, atol=1e-10)
    
    def test_hamiltonian_hermitian(self):
        """Test that Hamiltonian is Hermitian."""
        grid = SpectralGrid(N=32)
        ops = SpectralOperators(grid)
        
        V = np.sin(3 * grid.sigma)**2  # Some potential
        H = ops.build_hamiltonian_matrix(V)
        
        # Check Hermitian: H = H†
        assert_allclose(H, np.conj(H.T), atol=1e-10)
    
    def test_expectation_value(self):
        """Test energy expectation value calculation."""
        grid = SpectralGrid(N=64)
        ops = SpectralOperators(grid, m_eff=1.0, hbar=1.0)
        
        # For plane wave exp(ikσ) with V=0, E = k²/2 per unit norm
        # With normalization ∫|ψ|² = 2π, <H> = (k²/2) × 2π
        k = 2
        psi = np.exp(1j * k * grid.sigma)
        psi = grid.normalize(psi)  # ||ψ|| = √(2π)
        V = np.zeros(grid.N)
        
        E = ops.expectation_value(psi, V)
        # Expected: (k²/2) × ∫|ψ|² = (k²/2) × 2π
        expected = 0.5 * k**2 * 2 * np.pi
        
        assert_allclose(E, expected, rtol=1e-5)

