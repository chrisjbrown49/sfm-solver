#!/usr/bin/env python3
"""
Test baryon solver fixes from baryon_solver_fix.md

Phase 1A: n_eff scan - test different effective quantum numbers
Phase 1B: quark_width scan - test different quark envelope widths
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1

# Experimental values
EXP_MASSES = {
    'electron': 0.511,   # MeV
    'muon': 105.66,      # MeV
    'tau': 1776.86,      # MeV
    'pion': 139.57,      # MeV
    'proton': 938.27,    # MeV
}


def run_baseline():
    """Run baseline test to establish current behavior."""
    print("=" * 70)
    print("BASELINE TEST (current implementation)")
    print("=" * 70)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    # Solve electron to derive beta
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
    beta_out = 0.000511 / (e.structure_norm ** 2)
    
    # Solve proton with default settings
    proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3))
    
    pred_mass = beta_out * proton.structure_norm**2 * 1000
    error = (pred_mass - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100
    
    print(f"\nBeta derived from electron: {beta_out:.6f} GeV")
    print(f"\nProton (default n_eff=3, width=0.5):")
    print(f"  A = {proton.structure_norm:.4f}")
    print(f"  Predicted: {pred_mass:.1f} MeV")
    print(f"  Experimental: {EXP_MASSES['proton']:.1f} MeV")
    print(f"  Error: {error:.1f}%")
    
    return beta_out


def test_n_eff_scan(beta_out: float):
    """Phase 1A: Systematic n_eff scan."""
    print("\n" + "=" * 70)
    print("PHASE 1A: n_eff SCAN")
    print("=" * 70)
    print("\nTesting different effective quantum numbers for baryon.")
    print("n_eff = 3 corresponds to 3 quarks (current assumption)")
    print("Higher n_eff may be needed to capture binding energy contributions.\n")
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    results = {}
    n_eff_values = np.arange(2.5, 5.5, 0.5)
    
    print(f"{'n_eff':<8} {'A':>10} {'Mass (MeV)':>12} {'Error%':>10}")
    print("-" * 44)
    
    for n_eff in n_eff_values:
        try:
            proton = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3),
                n_eff=n_eff,
                quark_width=0.5,  # Keep default width
            )
            pred_mass = beta_out * proton.structure_norm**2 * 1000
            error = (pred_mass - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100
            results[n_eff] = {'A': proton.structure_norm, 'm': pred_mass, 'error': error}
            print(f"{n_eff:<8.1f} {proton.structure_norm:>10.4f} {pred_mass:>12.1f} {error:>10.1f}%")
        except Exception as e:
            print(f"{n_eff:<8.1f} {'ERROR':>10} {str(e)}")
    
    # Find best match
    if results:
        best_n_eff = min(results.items(), key=lambda x: abs(x[1]['error']))[0]
        print(f"\n*** Best n_eff = {best_n_eff:.1f} (error = {results[best_n_eff]['error']:.1f}%) ***")
    
    return results


def test_quark_width_scan(beta_out: float):
    """Phase 1B: Systematic quark width scan."""
    print("\n" + "=" * 70)
    print("PHASE 1B: QUARK WIDTH SCAN")
    print("=" * 70)
    print("\nTesting different quark envelope widths.")
    print("Larger widths = more overlap between quarks.\n")
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    results = {}
    width_values = np.arange(0.3, 1.3, 0.1)
    
    print(f"{'width':<8} {'A':>10} {'Mass (MeV)':>12} {'Error%':>10}")
    print("-" * 44)
    
    for width in width_values:
        try:
            proton = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3),
                n_eff=3.0,  # Keep default n_eff
                quark_width=width,
            )
            pred_mass = beta_out * proton.structure_norm**2 * 1000
            error = (pred_mass - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100
            results[width] = {'A': proton.structure_norm, 'm': pred_mass, 'error': error}
            print(f"{width:<8.2f} {proton.structure_norm:>10.4f} {pred_mass:>12.1f} {error:>10.1f}%")
        except Exception as e:
            print(f"{width:<8.2f} {'ERROR':>10} {str(e)}")
    
    # Find best match
    if results:
        best_width = min(results.items(), key=lambda x: abs(x[1]['error']))[0]
        print(f"\n*** Best width = {best_width:.2f} (error = {results[best_width]['error']:.1f}%) ***")
    
    return results


def test_combined_scan(beta_out: float):
    """Combined n_eff and width optimization."""
    print("\n" + "=" * 70)
    print("COMBINED SCAN: n_eff + quark_width")
    print("=" * 70)
    print("\nSearching for optimal combination...\n")
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    best_error = float('inf')
    best_params = None
    
    n_eff_values = np.arange(3.0, 5.0, 0.25)
    width_values = np.arange(0.4, 1.0, 0.1)
    
    print(f"{'n_eff':<8} {'width':<8} {'A':>10} {'Mass (MeV)':>12} {'Error%':>10}")
    print("-" * 52)
    
    for n_eff in n_eff_values:
        for width in width_values:
            try:
                proton = solver.solve_baryon_self_consistent(
                    quark_wells=(1, 2, 3),
                    n_eff=n_eff,
                    quark_width=width,
                )
                pred_mass = beta_out * proton.structure_norm**2 * 1000
                error = (pred_mass - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100
                
                if abs(error) < abs(best_error):
                    best_error = error
                    best_params = (n_eff, width, proton.structure_norm, pred_mass)
                    print(f"{n_eff:<8.2f} {width:<8.2f} {proton.structure_norm:>10.4f} {pred_mass:>12.1f} {error:>10.1f}% *")
            except Exception as e:
                pass
    
    if best_params:
        print("\n" + "=" * 52)
        print(f"*** BEST RESULT ***")
        print(f"  n_eff = {best_params[0]:.2f}")
        print(f"  width = {best_params[1]:.2f}")
        print(f"  A = {best_params[2]:.4f}")
        print(f"  Mass = {best_params[3]:.1f} MeV")
        print(f"  Error = {best_error:.1f}%")
    
    return best_params


def test_nonlinear_iteration(beta_out: float):
    """Phase 2: Test nonlinear iteration effect on baryon mass."""
    print("\n" + "=" * 70)
    print("PHASE 2: NONLINEAR ITERATION")
    print("=" * 70)
    print("\nTesting effect of nonlinear feedback g1|chi|^4 on energy denominators.")
    print("This should enhance induced components and increase amplitude.\n")
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    print(f"{'max_iter_nl':<12} {'A':>10} {'Mass (MeV)':>12} {'Error%':>10}")
    print("-" * 48)
    
    results = {}
    for max_iter_nl in [0, 1, 3, 5, 10]:
        try:
            proton = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3),
                n_eff=3.0,
                quark_width=0.5,
                max_iter_nl=max_iter_nl,
            )
            pred_mass = beta_out * proton.structure_norm**2 * 1000
            error = (pred_mass - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100
            results[max_iter_nl] = {'A': proton.structure_norm, 'm': pred_mass, 'error': error}
            print(f"{max_iter_nl:<12} {proton.structure_norm:>10.4f} {pred_mass:>12.1f} {error:>10.1f}%")
        except Exception as e:
            print(f"{max_iter_nl:<12} {'ERROR':>10} {str(e)}")
    
    return results


def test_combined_with_nl(beta_out: float):
    """Test n_eff + nonlinear iteration combined."""
    print("\n" + "=" * 70)
    print("PHASE 2B: n_eff + NONLINEAR ITERATION COMBINED")
    print("=" * 70)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    best_error = float('inf')
    best_params = None
    
    print(f"{'n_eff':<8} {'nl_iter':<8} {'A':>10} {'Mass (MeV)':>12} {'Error%':>10}")
    print("-" * 52)
    
    for n_eff in [3.0, 3.5, 4.0, 4.5]:
        for max_iter_nl in [0, 5, 10]:
            try:
                proton = solver.solve_baryon_self_consistent(
                    quark_wells=(1, 2, 3),
                    n_eff=n_eff,
                    max_iter_nl=max_iter_nl,
                )
                pred_mass = beta_out * proton.structure_norm**2 * 1000
                error = (pred_mass - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100
                
                if abs(error) < abs(best_error):
                    best_error = error
                    best_params = (n_eff, max_iter_nl, proton.structure_norm, pred_mass)
                    print(f"{n_eff:<8.1f} {max_iter_nl:<8} {proton.structure_norm:>10.4f} {pred_mass:>12.1f} {error:>10.1f}% *")
                else:
                    print(f"{n_eff:<8.1f} {max_iter_nl:<8} {proton.structure_norm:>10.4f} {pred_mass:>12.1f} {error:>10.1f}%")
            except Exception as e:
                pass
    
    if best_params:
        print("\n" + "=" * 52)
        print(f"*** BEST RESULT ***")
        print(f"  n_eff = {best_params[0]:.1f}")
        print(f"  max_iter_nl = {best_params[1]}")
        print(f"  A = {best_params[2]:.4f}")
        print(f"  Mass = {best_params[3]:.1f} MeV")
        print(f"  Error = {best_error:.1f}%")
    
    return best_params


def main():
    print("=" * 70)
    print("BARYON SOLVER FIX - Phase 1 & 2 Tests")
    print("=" * 70)
    print(f"\nUsing ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}, G1={G1}")
    
    # Run baseline
    beta_out = run_baseline()
    
    # Phase 1A: n_eff scan
    n_eff_results = test_n_eff_scan(beta_out)
    
    # Phase 1B: quark width scan
    width_results = test_quark_width_scan(beta_out)
    
    # Combined optimization
    best_phase1 = test_combined_scan(beta_out)
    
    # Phase 2: Nonlinear iteration
    nl_results = test_nonlinear_iteration(beta_out)
    
    # Phase 2B: Combined n_eff + nonlinear
    best_phase2 = test_combined_with_nl(beta_out)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n--- Phase 1 (n_eff + width) ---")
    if best_phase1:
        print(f"  Best: n_eff={best_phase1[0]:.2f}, width={best_phase1[1]:.2f}")
        print(f"  Mass = {best_phase1[3]:.1f} MeV, Error = {(best_phase1[3] - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100:.1f}%")
    
    print("\n--- Phase 2 (n_eff + nonlinear) ---")
    if best_phase2:
        print(f"  Best: n_eff={best_phase2[0]:.1f}, max_iter_nl={best_phase2[1]}")
        print(f"  Mass = {best_phase2[3]:.1f} MeV, Error = {(best_phase2[3] - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100:.1f}%")
    
    # Determine overall outcome
    best_overall = None
    if best_phase1 and best_phase2:
        err1 = abs((best_phase1[3] - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100)
        err2 = abs((best_phase2[3] - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100)
        if err2 < err1:
            best_overall = ('Phase 2', best_phase2, err2)
        else:
            best_overall = ('Phase 1', best_phase1, err1)
    
    if best_overall:
        print(f"\n*** OVERALL BEST: {best_overall[0]} with error = {best_overall[2]:.1f}% ***")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

