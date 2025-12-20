#!/usr/bin/env python3
"""
Scan g1 values for SCF baryon solver.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL


def scan_g1():
    """Scan g1 values to find optimal for SCF."""
    print("=" * 60)
    print("SCF g1 SCAN")
    print("=" * 60)
    
    g1_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    print(f"\n{'g1':<10} {'A':>10} {'Mass (MeV)':>12} {'Error%':>10} {'Converged':>10}")
    print("-" * 56)
    
    # Get beta from a baseline solver
    baseline = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=1.0,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    e = baseline.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
    beta = 0.000511 / (e.structure_norm ** 2)
    
    best_err = float('inf')
    best_g1 = None
    
    for g1 in g1_values:
        try:
            solver = NonSeparableWavefunctionSolver(
                alpha=ALPHA, g_internal=G_INTERNAL, g1=g1,
                g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
            )
            
            result = solver.solve_baryon_scf(
                max_iter_scf=100,
                max_iter_outer=30,
                verbose=False,
            )
            
            A = result.structure_norm
            pred = beta * A**2 * 1000
            err = (pred - 938.27) / 938.27 * 100
            
            marker = ""
            if abs(err) < abs(best_err):
                best_err = err
                best_g1 = g1
                marker = " *"
            
            print(f"{g1:<10.3f} {A:>10.4f} {pred:>12.2f} {err:>+10.1f}% {result.converged:>10}{marker}")
            
        except Exception as ex:
            print(f"{g1:<10.3f} {'ERROR':>10} {str(ex)[:30]}")
    
    print(f"\nBest g1 = {best_g1} with error = {best_err:.1f}%")
    print(f"\nNote: A ~ 1.8 regardless of g1, because each quark is normalized.")
    print(f"The SCF doesn't include spatial-subspace coupling (alpha, R_ij)")
    print(f"which is what caused amplitude growth in the perturbative solver.")


if __name__ == "__main__":
    scan_g1()

