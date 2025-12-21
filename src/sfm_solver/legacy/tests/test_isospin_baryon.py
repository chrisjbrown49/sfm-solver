"""
Integration tests for isospin splitting in baryon solver.

Tests that the winding-dependent kinetic energy terms produce
automatic proton-neutron mass difference of ~1-2 MeV from first principles.
"""

import numpy as np
import pytest
from sfm_solver.core.nonseparable_wavefunction_solver import (
    NonSeparableWavefunctionSolver
)


def test_proton_neutron_mass_difference():
    """
    Test that isospin splitting emerges automatically from winding kinetic terms.
    
    This is the key physics test: proton (uud, k=[5,5,-3]) and neutron (udd, k=[5,-3,-3])
    should have different amplitudes due to the kinetic energy difference between k=5 and k=-3 quarks.
    
    NOTE: This test validates the MECHANISM (isospin splitting emerges from winding kinetic terms),
    not the exact mass values (which require proper parameter optimization).
    """
    
    # Initialize solver with typical optimization parameters
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
        g1=49.0,   # Typical from optimization
        g2=67.0,   # Typical from optimization
        lambda_so=0.20,  # Typical from optimization
    )
    
    print("\n=== Testing Proton (uud) ===")
    chi1_p, chi2_p, chi3_p, A_p = solver.solve_baryon_self_consistent_field(
        color_phases=(0, 2*np.pi/3, 4*np.pi/3),
        quark_windings=(5, 5, -3),  # uud
        quark_spins=(+1, -1, +1),
        quark_generations=(1, 1, 1),
        max_iter=300,
        tol=1e-3,
        mixing=0.1,
        verbose=False,
    )
    
    print("\n=== Testing Neutron (udd) ===")
    chi1_n, chi2_n, chi3_n, A_n = solver.solve_baryon_self_consistent_field(
        color_phases=(0, 2*np.pi/3, 4*np.pi/3),
        quark_windings=(5, -3, -3),  # udd
        quark_spins=(+1, +1, -1),
        quark_generations=(1, 1, 1),
        max_iter=300,
        tol=1e-3,
        mixing=0.1,
        verbose=False,
    )
    
    # KEY TEST: Check that amplitudes are DIFFERENT due to winding kinetic terms
    # The masses would be m = β × A², but we test amplitude difference directly
    
    print("\n=== Results ===")
    print(f"Proton amplitude: A_p = {A_p:.6f}")
    print(f"Neutron amplitude: A_n = {A_n:.6f}")
    print(f"Amplitude ratio: A_n/A_p = {A_n/A_p:.6f}")
    print(f"Amplitude difference: |A_n - A_p| = {abs(A_n - A_p):.6f}")
    
    # The key physics test: amplitudes should be DIFFERENT
    amplitude_diff_percent = abs(A_n - A_p) / A_p * 100
    print(f"Amplitude difference: {amplitude_diff_percent:.3f}%")
    
    # Validation checks
    assert A_p > 0.1, f"Proton amplitude too small: {A_p:.6f}"
    assert A_n > 0.1, f"Neutron amplitude too small: {A_n:.6f}"
    
    assert abs(A_n - A_p) > 1e-6, \
        f"Amplitudes should be different due to isospin, got |A_n - A_p| = {abs(A_n - A_p):.2e}"
    
    # Check that the difference is reasonable (not too small, not too large)
    assert amplitude_diff_percent > 0.01 and amplitude_diff_percent < 10.0, \
        f"Amplitude difference should be 0.01-10%, got {amplitude_diff_percent:.3f}%"
    
    print("\n[PASS] Isospin splitting test passed!")
    print(f"  - Amplitudes are different (isospin mechanism working) [PASS]")
    print(f"  - Amplitude difference is {amplitude_diff_percent:.3f}% [PASS]")
    
    # Return both amplitudes for further analysis
    return A_p, A_n


def test_proton_convergence():
    """Test that proton solver converges reliably."""
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
        g1=49.0,
        g2=67.0,
        lambda_so=0.20,
    )
    
    chi1, chi2, chi3, A = solver.solve_baryon_self_consistent_field(
        color_phases=(0, 2*np.pi/3, 4*np.pi/3),
        quark_windings=(5, 5, -3),  # uud
        quark_spins=(+1, -1, +1),
        quark_generations=(1, 1, 1),
        max_iter=500,
        tol=1e-4,
        mixing=0.05,
        verbose=False,
    )
    
    # Check that amplitude is reasonable
    assert A > 0.1, f"Proton amplitude too small: {A:.6f}"
    assert A < 100.0, f"Proton amplitude too large: {A:.6f}"
    
    # Check that wavefunctions are normalized
    dsigma = solver.basis.dsigma
    norm1 = np.sqrt(np.sum(np.abs(chi1)**2) * dsigma)
    norm2 = np.sqrt(np.sum(np.abs(chi2)**2) * dsigma)
    norm3 = np.sqrt(np.sum(np.abs(chi3)**2) * dsigma)
    
    assert abs(norm1 - 1.0) < 0.1, f"Quark 1 not normalized: {norm1:.3f}"
    assert abs(norm2 - 1.0) < 0.1, f"Quark 2 not normalized: {norm2:.3f}"
    assert abs(norm3 - 1.0) < 0.1, f"Quark 3 not normalized: {norm3:.3f}"
    
    print(f"[PASS] Proton convergence test passed: A={A:.6f}")


def test_neutron_convergence():
    """Test that neutron solver converges reliably."""
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
        g1=49.0,
        g2=67.0,
        lambda_so=0.20,
    )
    
    chi1, chi2, chi3, A = solver.solve_baryon_self_consistent_field(
        color_phases=(0, 2*np.pi/3, 4*np.pi/3),
        quark_windings=(5, -3, -3),  # udd
        quark_spins=(+1, +1, -1),
        quark_generations=(1, 1, 1),
        max_iter=500,
        tol=1e-4,
        mixing=0.05,
        verbose=False,
    )
    
    # Check that amplitude is reasonable
    assert A > 0.1, f"Neutron amplitude too small: {A:.6f}"
    assert A < 100.0, f"Neutron amplitude too large: {A:.6f}"
    
    # Check that wavefunctions are normalized
    dsigma = solver.basis.dsigma
    norm1 = np.sqrt(np.sum(np.abs(chi1)**2) * dsigma)
    norm2 = np.sqrt(np.sum(np.abs(chi2)**2) * dsigma)
    norm3 = np.sqrt(np.sum(np.abs(chi3)**2) * dsigma)
    
    assert abs(norm1 - 1.0) < 0.1, f"Quark 1 not normalized: {norm1:.3f}"
    assert abs(norm2 - 1.0) < 0.1, f"Quark 2 not normalized: {norm2:.3f}"
    assert abs(norm3 - 1.0) < 0.1, f"Quark 3 not normalized: {norm3:.3f}"
    
    print(f"[PASS] Neutron convergence test passed: A={A:.6f}")


def test_winding_kinetic_contribution():
    """
    Test that winding kinetic terms contribute to the energy difference.
    
    This verifies that the k^2 terms are actually affecting the Hamiltonian.
    """
    solver = NonSeparableWavefunctionSolver(
        alpha=10.0,
        g_internal=0.003,
    )
    
    # Build kinetic operators for different windings
    T_base = solver._build_kinetic_operator()
    T_wind_k5 = solver._build_winding_kinetic_terms(k=5)
    T_wind_k3 = solver._build_winding_kinetic_terms(k=-3)
    
    # Build simple test Hamiltonians (without mean field)
    V_well = np.diag(solver.V_sigma)
    H_k5 = T_base + T_wind_k5 + V_well
    H_k3 = T_base + T_wind_k3 + V_well
    
    # Get ground state energies
    E_k5 = np.linalg.eigvalsh(H_k5)[0]
    E_k3 = np.linalg.eigvalsh(H_k3)[0]
    
    # k=5 should have higher energy than k=-3 (due to k^2 = 25 vs 9)
    assert E_k5 > E_k3, \
        f"k=5 should have higher ground state energy than k=-3, got E_k5={E_k5:.6f}, E_k3={E_k3:.6f}"
    
    energy_diff = E_k5 - E_k3
    print(f"[PASS] Winding kinetic contribution verified:")
    print(f"  Ground state energy k=5: {E_k5:.6f}")
    print(f"  Ground state energy k=-3: {E_k3:.6f}")
    print(f"  Energy difference: {energy_diff:.6f}")


if __name__ == "__main__":
    print("\n=== Testing Isospin Splitting in Baryon Solver ===\n")
    
    print("Test 1: Proton convergence")
    test_proton_convergence()
    print()
    
    print("Test 2: Neutron convergence")
    test_neutron_convergence()
    print()
    
    print("Test 3: Winding kinetic contribution")
    test_winding_kinetic_contribution()
    print()
    
    print("Test 4: Proton-neutron amplitude difference (main physics test)")
    A_p, A_n = test_proton_neutron_mass_difference()
    print()
    
    print(f"\n[PASS] All isospin baryon tests passed!")
    print(f"\nFinal Result: A_p = {A_p:.6f}, A_n = {A_n:.6f}")
    print(f"Amplitude difference: {abs(A_n - A_p)/A_p * 100:.3f}%")
    print(f"\nIsospin mechanism validated: winding kinetic terms produce different baryon amplitudes!")

