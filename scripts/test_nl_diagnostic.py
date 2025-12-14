#!/usr/bin/env python3
"""
Diagnostic test for nonlinear iteration in baryon solver.

Goal: Understand why nonlinear iteration destroys the amplitude.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL


def run_diagnostic():
    """Run baryon solver with detailed diagnostics."""
    print("=" * 70)
    print("NONLINEAR ITERATION DIAGNOSTIC")
    print("=" * 70)
    
    # Test with small g1 to reduce nonlinear effects
    g1 = 1.0
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=g1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    # Run without NL iteration first
    print(f"\n### WITHOUT NONLINEAR ITERATION ###")
    proton_no_nl = solver.solve_baryon_self_consistent(
        quark_wells=(1, 2, 3),
        n_eff=3.0,
        max_iter_nl=0,
        verbose=True,
    )
    print(f"Final A = {proton_no_nl.structure_norm:.6f}")
    
    # Run with NL iteration
    print(f"\n### WITH NONLINEAR ITERATION (g1={g1}) ###")
    proton_nl = solver.solve_baryon_self_consistent(
        quark_wells=(1, 2, 3),
        n_eff=3.0,
        max_iter_nl=3,
        verbose=True,
    )
    print(f"Final A = {proton_nl.structure_norm:.6f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Compare amplitudes
    print(f"\nAmplitude without NL: {proton_no_nl.structure_norm:.4f}")
    print(f"Amplitude with NL:    {proton_nl.structure_norm:.4f}")
    print(f"Ratio: {proton_nl.structure_norm / proton_no_nl.structure_norm:.4f}")
    
    # The issue: NL iteration is causing amplitude to DECREASE when it should INCREASE
    if proton_nl.structure_norm < proton_no_nl.structure_norm:
        print("\n*** PROBLEM: NL iteration DECREASED amplitude (should INCREASE!) ***")
        print("\nPossible causes:")
        print("1. Nonlinear shifts are being ADDED to target energy, making E_denom LARGER")
        print("2. This SUPPRESSES induced components instead of enhancing them")
        print("3. Need to SUBTRACT shifts, or shift induced states more than target")


def test_energy_shift_direction():
    """Test what happens to energy denominators with nonlinear shifts."""
    print("\n" + "=" * 70)
    print("ENERGY SHIFT DIRECTION TEST")
    print("=" * 70)
    
    # In perturbation theory:
    # induced = -alpha * R / (E_target - E_state)
    # 
    # If E_target > E_state (positive denominator):
    #   - Adding V_nl to E_target makes denominator LARGER -> SMALLER induced
    #   - Adding V_nl to E_state makes denominator SMALLER -> LARGER induced
    #
    # The nonlinear term g1|chi|^4 creates a POSITIVE potential.
    # If we add it to both target and state equally, denominators don't change.
    # If target has more chi overlap, its shift is larger -> denominator grows -> suppression
    
    print("\nThe issue:")
    print("- E_denom = E_target_eff - E_state_eff")
    print("- V_nl_target = integral(rho_nl * |chi_primary|^2)")
    print("- V_nl_state = integral(rho_nl * |chi_induced|^2)")
    print("")
    print("If chi_primary dominates chi_total, then:")
    print("  rho_nl ~ |chi_primary|^4")
    print("  V_nl_target ~ integral(|chi_primary|^6) >> V_nl_state")
    print("")
    print("This makes E_target_eff >> E_state_eff")
    print("So E_denom grows LARGER, suppressing induced components!")
    print("")
    print("SOLUTION: We need the OPPOSITE effect!")
    print("  Option 1: Subtract V_nl from target energy")
    print("  Option 2: Only shift state energies (not target)")
    print("  Option 3: Different physical interpretation of the nonlinear term")


if __name__ == "__main__":
    run_diagnostic()
    test_energy_shift_direction()

