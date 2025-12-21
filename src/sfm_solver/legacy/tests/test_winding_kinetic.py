"""
Unit tests for winding-dependent kinetic energy terms.

Tests that the _build_winding_kinetic_terms() method produces correct
matrix structure with proper k^2 scaling for isospin physics.
"""

import numpy as np
import pytest
from sfm_solver.core.nonseparable_wavefunction_solver import (
    NonSeparableWavefunctionSolver
)


def test_winding_kinetic_terms_shape():
    """Test that winding kinetic terms have correct matrix dimensions."""
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
    )
    
    # Build winding terms for k=3
    T_k3 = solver._build_winding_kinetic_terms(k=3)
    
    # Check shape matches basis size
    N_sigma = solver.basis.N_sigma
    assert T_k3.shape == (N_sigma, N_sigma), f"Expected shape ({N_sigma}, {N_sigma}), got {T_k3.shape}"
    
    # Check that matrix is complex (due to -2ik d/dsigma term)
    assert T_k3.dtype == complex, "Winding kinetic terms should be complex"
    
    print(f"[PASS] Winding kinetic terms have correct shape: {T_k3.shape}")


def test_winding_kinetic_k_squared_scaling():
    """Test that k^2 term dominates diagonal and scales correctly."""
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
    )
    
    # Build winding terms for k=3 and k=5
    T_k3 = solver._build_winding_kinetic_terms(k=3)
    T_k5 = solver._build_winding_kinetic_terms(k=5)
    
    # Extract diagonal (k^2 term)
    k3_diagonal = np.diag(T_k3).real
    k5_diagonal = np.diag(T_k5).real
    
    # Check that k=5 has higher energy than k=3
    assert np.mean(k5_diagonal) > np.mean(k3_diagonal), \
        "k=5 should have higher kinetic energy than k=3"
    
    # Check that difference scales as k^2
    # E(k=5) / E(k=3) should be proportional to 25 / 9
    ratio = np.mean(k5_diagonal) / np.mean(k3_diagonal)
    expected_ratio = 25.0 / 9.0  # ≈ 2.78
    
    # Allow 10% tolerance due to off-diagonal contributions
    assert abs(ratio - expected_ratio) < 0.3, \
        f"Got ratio {ratio:.2f}, expected {expected_ratio:.2f}"
    
    print(f"[PASS] k^2 scaling verified:")
    print(f"  k=3 diagonal mean: {np.mean(k3_diagonal):.6f}")
    print(f"  k=5 diagonal mean: {np.mean(k5_diagonal):.6f}")
    print(f"  Ratio k5/k3: {ratio:.2f} (expected {expected_ratio:.2f})")


def test_winding_kinetic_off_diagonal_structure():
    """Test that -2ik d/dsigma term produces correct off-diagonal structure."""
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
    )
    
    # Build winding terms for k=5
    T_k5 = solver._build_winding_kinetic_terms(k=5)
    
    # Check that matrix has non-zero imaginary parts (from -2ik term)
    off_diagonal_imag = np.imag(T_k5[0, 1])
    assert abs(off_diagonal_imag) > 1e-10, \
        "Off-diagonal elements should have imaginary parts from -2ik d/dsigma term"
    
    # Check that off-diagonal structure respects periodic BC
    # Should have non-zero elements at [i, i+1] and [i, i-1]
    N = T_k5.shape[0]
    assert abs(T_k5[0, 1]) > 1e-10, "Should have coupling to next element"
    assert abs(T_k5[0, N-1]) > 1e-10, "Should have periodic coupling to last element"
    
    print(f"[PASS] Off-diagonal structure verified:")
    print(f"  T[0,1] = {T_k5[0, 1]:.6e}")
    print(f"  T[0,{N-1}] = {T_k5[0, N-1]:.6e}")


def test_winding_kinetic_sign_convention():
    """Test that signed k values are handled correctly."""
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
    )
    
    # Build winding terms for k=-3 (down quark) and k=+5 (up quark)
    T_k_minus3 = solver._build_winding_kinetic_terms(k=-3)
    T_k_plus5 = solver._build_winding_kinetic_terms(k=5)
    
    # Both should have positive k^2 diagonal terms
    k_minus3_diagonal = np.diag(T_k_minus3).real
    k_plus5_diagonal = np.diag(T_k_plus5).real
    
    assert np.all(k_minus3_diagonal > 0), "k^2 term should be positive for k=-3"
    assert np.all(k_plus5_diagonal > 0), "k^2 term should be positive for k=5"
    
    # The -2ik term should have opposite signs
    off_diag_minus3 = np.imag(T_k_minus3[0, 1])
    off_diag_plus5 = np.imag(T_k_plus5[0, 1])
    
    # Signs should be opposite
    assert np.sign(off_diag_minus3) != np.sign(off_diag_plus5), \
        "Off-diagonal imaginary parts should have opposite signs for opposite k"
    
    print(f"[PASS] Sign convention verified:")
    print(f"  k=-3: diagonal = {np.mean(k_minus3_diagonal):.6f}, off-diag = {off_diag_minus3:.6e}")
    print(f"  k=+5: diagonal = {np.mean(k_plus5_diagonal):.6f}, off-diag = {off_diag_plus5:.6e}")


def test_winding_kinetic_zero_winding():
    """Test that k=0 produces minimal kinetic term (should be nearly zero)."""
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
    )
    
    # Build winding terms for k=0
    T_k0 = solver._build_winding_kinetic_terms(k=0)
    
    # For k=0, both k^2 and -2ik terms should vanish
    assert np.allclose(T_k0, 0.0, atol=1e-10), \
        "k=0 should produce zero winding kinetic terms"
    
    print(f"[PASS] k=0 case verified: max|T_k0| = {np.max(np.abs(T_k0)):.2e}")


if __name__ == "__main__":
    print("\n=== Testing Winding Kinetic Energy Terms ===\n")
    
    test_winding_kinetic_terms_shape()
    print()
    
    test_winding_kinetic_k_squared_scaling()
    print()
    
    test_winding_kinetic_off_diagonal_structure()
    print()
    
    test_winding_kinetic_sign_convention()
    print()
    
    test_winding_kinetic_zero_winding()
    print()
    
    print("\n[PASS] All winding kinetic tests passed!")

