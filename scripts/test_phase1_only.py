#!/usr/bin/env python3
"""
Test if Phase 1 (no nonlinear iteration) can produce correct proton mass
by tuning alpha, g_internal, or n_eff - without needing the NL iteration.

This answers: Did we over-engineer the solution?
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1

EXP = {
    'electron': 0.511,
    'muon': 105.66,
    'tau': 1776.86,
    'pion': 139.57,
    'proton': 938.27,
}


def test_alpha_scan():
    """Can we fix proton by tuning alpha alone (no NL iteration)?"""
    print("=" * 70)
    print("TEST 1: Can tuning ALPHA fix proton (no NL iteration)?")
    print("=" * 70)
    
    print(f"\n{'alpha':<10} {'e err%':>8} {'pi err%':>8} {'p err%':>8}")
    print("-" * 40)
    
    for alpha in [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        solver = NonSeparableWavefunctionSolver(
            alpha=alpha, g_internal=G_INTERNAL, g1=G1,
            g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
        )
        
        e = solver.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
        pion = solver.solve_meson_self_consistent(quark_wells=(1, 2))
        proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3), max_iter_nl=0)
        
        beta = 0.000511 / (e.structure_norm ** 2)
        
        e_m = beta * e.structure_norm**2 * 1000
        pi_m = beta * pion.structure_norm**2 * 1000
        p_m = beta * proton.structure_norm**2 * 1000
        
        e_err = (e_m - EXP['electron']) / EXP['electron'] * 100
        pi_err = (pi_m - EXP['pion']) / EXP['pion'] * 100
        p_err = (p_m - EXP['proton']) / EXP['proton'] * 100
        
        print(f"{alpha:<10.1f} {e_err:>8.1f} {pi_err:>8.1f} {p_err:>8.1f}")


def test_g_internal_scan():
    """Can we fix proton by tuning g_internal alone?"""
    print("\n" + "=" * 70)
    print("TEST 2: Can tuning G_INTERNAL fix proton (no NL iteration)?")
    print("=" * 70)
    
    print(f"\n{'g_internal':<12} {'e err%':>8} {'pi err%':>8} {'p err%':>8}")
    print("-" * 42)
    
    for g_int in [0.001, 0.002, 0.003, 0.005, 0.01, 0.02]:
        solver = NonSeparableWavefunctionSolver(
            alpha=ALPHA, g_internal=g_int, g1=G1,
            g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
        )
        
        e = solver.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
        pion = solver.solve_meson_self_consistent(quark_wells=(1, 2))
        proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3), max_iter_nl=0)
        
        beta = 0.000511 / (e.structure_norm ** 2)
        
        e_m = beta * e.structure_norm**2 * 1000
        pi_m = beta * pion.structure_norm**2 * 1000
        p_m = beta * proton.structure_norm**2 * 1000
        
        e_err = (e_m - EXP['electron']) / EXP['electron'] * 100
        pi_err = (pi_m - EXP['pion']) / EXP['pion'] * 100
        p_err = (p_m - EXP['proton']) / EXP['proton'] * 100
        
        print(f"{g_int:<12.4f} {e_err:>8.1f} {pi_err:>8.1f} {p_err:>8.1f}")


def test_n_eff_scan():
    """Can we fix proton by tuning n_eff alone?"""
    print("\n" + "=" * 70)
    print("TEST 3: Can tuning n_eff fix proton (no NL iteration)?")
    print("=" * 70)
    
    print(f"\n{'n_eff':<10} {'proton A':>10} {'p mass':>10} {'p err%':>8}")
    print("-" * 42)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
    beta = 0.000511 / (e.structure_norm ** 2)
    
    for n_eff in np.arange(2.5, 5.5, 0.25):
        try:
            proton = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3), n_eff=n_eff, max_iter_nl=0
            )
            p_m = beta * proton.structure_norm**2 * 1000
            p_err = (p_m - EXP['proton']) / EXP['proton'] * 100
            print(f"{n_eff:<10.2f} {proton.structure_norm:>10.2f} {p_m:>10.1f} {p_err:>8.1f}%")
        except:
            print(f"{n_eff:<10.2f} {'ERROR':>10}")


def main():
    print("=" * 70)
    print("PHASE 1 ONLY: Can we fix proton WITHOUT nonlinear iteration?")
    print("=" * 70)
    print("\nQuestion: Did we over-engineer by adding NL iteration?")
    print("Test: Scan alpha, g_internal, n_eff to see if any combination")
    print("      can produce correct proton mass while keeping e, pion correct.")
    
    test_alpha_scan()
    test_g_internal_scan()
    test_n_eff_scan()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nLook at the results above:")
    print("- If any parameter scan shows proton near 0% error while e/pion stay good")
    print("  → NL iteration was NOT necessary (over-engineered)")
    print("- If no parameter combination works")
    print("  → NL iteration IS necessary for correct proton mass")


if __name__ == "__main__":
    main()

