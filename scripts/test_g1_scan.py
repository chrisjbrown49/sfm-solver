#!/usr/bin/env python3
"""
Empirical g1 scan for baryon solver.

Goal: Find a g1 value that produces correct proton mass with nonlinear iteration.
This is not first-principles but helps validate the physics direction.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL

# Experimental values
EXP_MASSES = {
    'electron': 0.511,   # MeV
    'muon': 105.66,      # MeV
    'tau': 1776.86,      # MeV
    'pion': 139.57,      # MeV
    'proton': 938.27,    # MeV
}


def test_g1_for_proton(g1_values, max_iter_nl=5):
    """Scan g1 values and measure proton mass prediction."""
    print("=" * 70)
    print(f"G1 SCAN FOR PROTON (max_iter_nl={max_iter_nl})")
    print("=" * 70)
    print(f"\nUsing ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}")
    print(f"Target proton mass: {EXP_MASSES['proton']:.1f} MeV\n")
    
    # First get electron for beta derivation (with g1=5000 baseline)
    solver_baseline = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=5000.0,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    e = solver_baseline.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
    beta_out = 0.000511 / (e.structure_norm ** 2)
    print(f"Beta derived from electron: {beta_out:.6f} GeV\n")
    
    print(f"{'g1':<12} {'A':>10} {'Mass (MeV)':>12} {'Error%':>10} {'Converged':>10}")
    print("-" * 60)
    
    results = {}
    best_error = float('inf')
    best_g1 = None
    
    for g1 in g1_values:
        try:
            solver = NonSeparableWavefunctionSolver(
                alpha=ALPHA, g_internal=G_INTERNAL, g1=g1,
                g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
            )
            
            proton = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3),
                n_eff=3.0,
                max_iter_nl=max_iter_nl,
            )
            
            pred_mass = beta_out * proton.structure_norm**2 * 1000
            error = (pred_mass - EXP_MASSES['proton']) / EXP_MASSES['proton'] * 100
            converged = proton.converged
            
            results[g1] = {
                'A': proton.structure_norm, 
                'm': pred_mass, 
                'error': error,
                'converged': converged
            }
            
            marker = ""
            if abs(error) < abs(best_error):
                best_error = error
                best_g1 = g1
                marker = " *"
            
            print(f"{g1:<12.2f} {proton.structure_norm:>10.4f} {pred_mass:>12.1f} {error:>10.1f}% {str(converged):>10}{marker}")
            
        except Exception as e:
            print(f"{g1:<12.2f} {'ERROR':>10} {str(e)[:30]}")
    
    if best_g1 is not None:
        print(f"\n*** Best g1 = {best_g1:.2f} (error = {best_error:.1f}%) ***")
    
    return results, best_g1


def test_all_particles_with_g1(g1):
    """Test all particles with a specific g1 value."""
    print("\n" + "=" * 70)
    print(f"ALL PARTICLES WITH g1={g1}")
    print("=" * 70)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=g1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    # Leptons (no NL iteration - they work fine without it)
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
    mu = solver.solve_lepton_self_consistent(n_target=2, max_iter_nl=0)
    tau = solver.solve_lepton_self_consistent(n_target=3, max_iter_nl=0)
    
    # Hadrons (with NL iteration)
    pion = solver.solve_meson_self_consistent(quark_wells=(1, 2))
    proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3), max_iter_nl=5)
    
    beta_out = 0.000511 / (e.structure_norm ** 2)
    
    particles = [
        ('Electron', e, 0.511),
        ('Muon', mu, 105.66),
        ('Tau', tau, 1776.86),
        ('Pion', pion, 139.57),
        ('Proton', proton, 938.27),
    ]
    
    print(f"\nbeta_output = {beta_out:.6f} GeV\n")
    print(f"{'Particle':<10} {'A':>8} {'Pred MeV':>10} {'Exp MeV':>10} {'Error%':>8}")
    print("-" * 50)
    
    for name, p, exp in particles:
        pred = beta_out * p.structure_norm**2 * 1000
        err = (pred - exp) / exp * 100
        print(f"{name:<10} {p.structure_norm:>8.4f} {pred:>10.2f} {exp:>10.2f} {err:>7.1f}%")
    
    return particles


def main():
    print("=" * 70)
    print("EMPIRICAL G1 SCAN")
    print("=" * 70)
    print("\nGoal: Find g1 that produces correct proton mass with nonlinear iteration.")
    print("This validates the physics before deriving g1 from first principles.\n")
    
    # First test: Wide scan with max_iter_nl=5
    print("\n### WIDE SCAN (orders of magnitude) ###")
    g1_wide = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    results_wide, best_wide = test_g1_for_proton(g1_wide, max_iter_nl=5)
    
    # Narrow scan around the best value if found
    if best_wide and best_wide < 5000:
        print("\n### NARROW SCAN (around best value) ###")
        if best_wide < 10:
            g1_narrow = np.linspace(max(0.1, best_wide/2), best_wide*2, 10)
        else:
            g1_narrow = np.linspace(best_wide/2, best_wide*2, 10)
        results_narrow, best_narrow = test_g1_for_proton(g1_narrow, max_iter_nl=5)
        
        # Test all particles with best g1
        if best_narrow:
            test_all_particles_with_g1(best_narrow)
    
    # Also test without nonlinear iteration for comparison
    print("\n### COMPARISON: NO NONLINEAR ITERATION ###")
    g1_no_nl = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]
    results_no_nl, best_no_nl = test_g1_for_proton(g1_no_nl, max_iter_nl=0)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if best_wide and results_wide[best_wide]['A'] > 1.5:
        print(f"\nBest g1 with NL iteration: {best_wide:.2f}")
        print(f"  Proton mass: {results_wide[best_wide]['m']:.1f} MeV (error: {results_wide[best_wide]['error']:.1f}%)")
        print(f"  Amplitude: {results_wide[best_wide]['A']:.4f}")
    else:
        print("\nNo stable solution found with nonlinear iteration.")
        print("Consider: The nonlinear feedback may need a different formulation.")
    
    if best_no_nl:
        print(f"\nBest g1 without NL iteration: {best_no_nl}")
        print(f"  Proton mass: {results_no_nl[best_no_nl]['m']:.1f} MeV (error: {results_no_nl[best_no_nl]['error']:.1f}%)")


if __name__ == "__main__":
    main()

