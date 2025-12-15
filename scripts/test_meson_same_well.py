#!/usr/bin/env python3
"""
Test hypothesis: Mesons have both quarks in the same primary well,
separated by secondary structure from V1.

Physics:
- V(σ) = V₀(1 - cos(3σ)) + V₁(1 - cos(6σ))
- Primary wells (from cos(3σ)): σ = π/3, π, 5π/3
- Secondary wells (from cos(6σ)): 6 sub-wells, period π/3
- Hypothesis: Quark and antiquark in same primary well (well 2 at σ=π),
  separated by secondary structure when V1 > 0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1

print("=" * 70)
print("TESTING MESON SAME-WELL HYPOTHESIS")
print("=" * 70)
print(f"ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}, G1={G1}")
print()

print("Hypothesis: Both quark and antiquark in same primary well (well 2)")
print("             separated by secondary structure from V1 term")
print()

# Test with different V1 values and quark_wells=(2,2)
print("=== Meson with quark_wells=(2,2) - Same Primary Well ===")
print(f'{"V1":>6} {"V0":>6} {"Pi_err":>10} {"Pi_mass":>10} {"Pi_A":>8} {"p_err":>10} {"p_mass":>10}')
print("-" * 70)

for V1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, V1=V1, n_max=5, l_max=2, N_sigma=64,
    )
    
    # Electron for beta calibration
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
    beta = 0.000511 / (e.structure_norm ** 2)
    
    # Pion with SAME well for both quarks
    pion = solver.solve_meson_self_consistent(quark_wells=(2, 2))
    
    # Proton for comparison
    proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3))
    
    pi_mass = beta * pion.structure_norm**2 * 1000
    pi_err = (pi_mass - 139.57) / 139.57 * 100
    
    p_mass = beta * proton.structure_norm**2 * 1000
    p_err = (p_mass - 938.27) / 938.27 * 100
    
    print(f'{V1:>6.2f} {1.0:>6.1f} {pi_err:>+9.1f}% {pi_mass:>10.1f} {pion.structure_norm:>8.2f} {p_err:>+9.1f}% {p_mass:>10.1f}')

print()
print("=== For comparison: Original quark_wells=(1,2) - Different Primary Wells ===")
print(f'{"V1":>6} {"V0":>6} {"Pi_err":>10} {"Pi_mass":>10} {"Pi_A":>8} {"p_err":>10} {"p_mass":>10}')
print("-" * 70)

for V1 in [0.0, 0.5, 1.0]:
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, V1=V1, n_max=5, l_max=2, N_sigma=64,
    )
    
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
    beta = 0.000511 / (e.structure_norm ** 2)
    
    # Pion with DIFFERENT wells (original)
    pion = solver.solve_meson_self_consistent(quark_wells=(1, 2))
    proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3))
    
    pi_mass = beta * pion.structure_norm**2 * 1000
    pi_err = (pi_mass - 139.57) / 139.57 * 100
    
    p_mass = beta * proton.structure_norm**2 * 1000
    p_err = (p_mass - 938.27) / 938.27 * 100
    
    print(f'{V1:>6.2f} {1.0:>6.1f} {pi_err:>+9.1f}% {pi_mass:>10.1f} {pion.structure_norm:>8.2f} {p_err:>+9.1f}% {p_mass:>10.1f}')

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("If the same-well hypothesis is correct:")
print("  - Pion should have reasonable mass with V1 > 0")
print("  - Pion amplitude should be reasonable (not collapsed)")
print("  - Proton should remain accurate")

