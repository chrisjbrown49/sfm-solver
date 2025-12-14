#!/usr/bin/env python3
"""
Test baryon solver using 4D nonseparable framework.

This tests that the baryon solver uses the SAME 4D framework as
leptons and mesons, ensuring consistent treatment across all particle types.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1


def test_color_neutrality():
    """Test: Verify color neutrality condition."""
    print("\n" + "=" * 60)
    print("TEST: Color Neutrality")
    print("=" * 60)
    
    phi1, phi2, phi3 = 0, 2*np.pi/3, 4*np.pi/3
    phase_sum = np.exp(1j*phi1) + np.exp(1j*phi2) + np.exp(1j*phi3)
    
    print(f"  sum(e^(i*phi)) = {phase_sum}")
    print(f"  |sum(e^(i*phi))| = {np.abs(phase_sum):.2e}")
    
    assert np.abs(phase_sum) < 1e-10, "Color neutrality violated"
    
    print("  [PASSED] Color neutrality satisfied")
    return True


def test_baryon_4d_framework():
    """Test: Verify baryon solver uses 4D framework with chi_components."""
    print("\n" + "=" * 60)
    print("TEST: Baryon 4D Framework Structure")
    print("=" * 60)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    result = solver.solve_baryon_self_consistent(verbose=True)
    
    # Check that result has chi_components (4D framework)
    print(f"\n  Result type: {type(result).__name__}")
    print(f"  Has chi_components: {hasattr(result, 'chi_components')}")
    print(f"  Number of spatial states: {len(result.chi_components)}")
    
    assert hasattr(result, 'chi_components'), "Baryon result should have chi_components"
    assert len(result.chi_components) > 1, "Baryon should have multiple spatial states (induced components)"
    
    # Check for amplitude differentiation
    print(f"\n  Amplitude A = {result.structure_norm:.6f}")
    assert result.structure_norm > 1.5, "Amplitude should be > 1.5 (not stuck at ~1)"
    
    # Check l_composition
    print(f"  l_composition: {result.l_composition}")
    
    print("\n  [PASSED] Baryon uses 4D framework with induced components")
    return result


def test_proton_mass():
    """Test: Proton mass prediction using 4D framework."""
    print("\n" + "=" * 60)
    print("TEST: Proton Mass Prediction (4D Framework)")
    print("=" * 60)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    # Electron for beta calibration
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
    beta = 0.000511 / (e.structure_norm ** 2)
    print(f"  Beta from electron: {beta:.6f} GeV")
    
    # Proton with 4D framework
    proton = solver.solve_baryon_self_consistent(verbose=True)
    
    A = proton.structure_norm
    m_predicted = beta * A**2 * 1000  # MeV
    m_experimental = 938.27  # MeV
    
    error = (m_predicted - m_experimental) / m_experimental * 100
    
    print(f"\n  Proton amplitude: A = {A:.6f}")
    print(f"  Predicted mass: {m_predicted:.2f} MeV")
    print(f"  Experimental:   {m_experimental:.2f} MeV")
    print(f"  Error:          {error:+.2f}%")
    
    if abs(error) < 5:
        print("  [EXCELLENT] Error < 5%")
    elif abs(error) < 10:
        print("  [GOOD] Error < 10%")
    elif abs(error) < 30:
        print("  [MODERATE] Error < 30%")
    else:
        print("  [NEEDS WORK] Error >= 30%")
    
    return m_predicted, m_experimental, error


def test_all_particles():
    """Test: Compare all particles using consistent 4D framework."""
    print("\n" + "=" * 60)
    print("TEST: All Particle Comparison (4D Framework)")
    print("=" * 60)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    print(f"\n  Parameters: ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}, G1={G1}")
    
    # Leptons (max_iter_outer=30 matches optimizer settings)
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
    mu = solver.solve_lepton_self_consistent(n_target=2, max_iter_outer=30, max_iter_nl=0)
    tau = solver.solve_lepton_self_consistent(n_target=3, max_iter_outer=30, max_iter_nl=0)
    
    # Pion (meson)
    pion = solver.solve_meson_self_consistent(quark_wells=(1, 2))
    
    # Proton (baryon) - now uses 4D framework
    proton = solver.solve_baryon_self_consistent()
    
    beta = 0.000511 / (e.structure_norm ** 2)
    
    exp_masses = {
        'Electron': 0.511,
        'Muon': 105.66,
        'Tau': 1776.86,
        'Pion': 139.57,
        'Proton': 938.27,
    }
    
    particles = [
        ('Electron', e),
        ('Muon', mu),
        ('Tau', tau),
        ('Pion', pion),
        ('Proton', proton),
    ]
    
    print(f"\n  Beta = {beta:.6f} GeV\n")
    print(f"  {'Particle':<14} {'A':>8} {'Pred (MeV)':>12} {'Exp (MeV)':>12} {'Error%':>10} {'Conv':>6}")
    print("  " + "-" * 66)
    
    results = []
    for name, p in particles:
        pred = beta * p.structure_norm**2 * 1000
        exp = exp_masses[name]
        err = (pred - exp) / exp * 100
        conv = "Y" if p.converged else "N"
        print(f"  {name:<14} {p.structure_norm:>8.4f} {pred:>12.2f} {exp:>12.2f} {err:>+10.1f}% {conv:>6}")
        results.append((name, err))
    
    return results


def test_consistency_with_meson():
    """Test: Verify baryon solver is consistent with meson solver structure."""
    print("\n" + "=" * 60)
    print("TEST: Consistency with Meson Solver")
    print("=" * 60)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    meson = solver.solve_meson_self_consistent()
    baryon = solver.solve_baryon_self_consistent()
    
    print(f"\n  Meson (pi+):")
    print(f"    Amplitude A = {meson.structure_norm:.6f}")
    print(f"    n_target = {meson.n_target}")
    print(f"    Num chi_components = {len(meson.chi_components)}")
    print(f"    l_composition = {meson.l_composition}")
    
    print(f"\n  Baryon (proton):")
    print(f"    Amplitude A = {baryon.structure_norm:.6f}")
    print(f"    n_target = {baryon.n_target}")
    print(f"    Num chi_components = {len(baryon.chi_components)}")
    print(f"    l_composition = {baryon.l_composition}")
    
    # Both should have same number of chi_components (same basis)
    assert len(meson.chi_components) == len(baryon.chi_components), \
        "Meson and baryon should use same spatial basis"
    
    # Baryon should have n_target=3, meson should have n_target=2
    assert meson.n_target == 2, "Meson n_target should be 2"
    assert baryon.n_target == 3, "Baryon n_target should be 3"
    
    # Baryon amplitude should be larger than meson (more quarks)
    assert baryon.structure_norm > meson.structure_norm, \
        "Baryon amplitude should be larger than meson"
    
    print("\n  [PASSED] Baryon solver is structurally consistent with meson solver")
    return True


def main():
    print("=" * 60)
    print("BARYON 4D FRAMEWORK TEST SUITE")
    print("=" * 60)
    print(f"\nUsing ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}, G1={G1}")
    
    # Test color neutrality (analytical check)
    test_color_neutrality()
    
    # Test 4D framework structure
    test_baryon_4d_framework()
    
    # Test consistency with meson solver
    test_consistency_with_meson()
    
    # Test proton mass
    m_pred, m_exp, error = test_proton_mass()
    
    # Compare all particles
    results = test_all_particles()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Proton mass prediction: {m_pred:.2f} MeV (error: {error:+.1f}%)")
    
    print("\n  All particle errors:")
    for name, err in results:
        status = "OK" if abs(err) < 30 else "NEEDS WORK"
        print(f"    {name:<14} {err:+7.1f}%  [{status}]")
    
    if abs(error) < 5:
        print("\n  *** SUCCESS: Proton mass within 5% of experiment! ***")
    elif abs(error) < 10:
        print("\n  *** GOOD: Proton mass within 10% of experiment ***")
    elif abs(error) < 30:
        print("\n  *** MODERATE: Proton mass within 30% of experiment ***")
    else:
        print("\n  *** NEEDS WORK: Proton mass error > 30% ***")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
