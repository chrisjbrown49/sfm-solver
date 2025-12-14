#!/usr/bin/env python3
"""Scan g1 values to find optimal proton mass prediction."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL

print('Scanning g1 to find optimal proton mass...')
print(f'ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}')
print()
print(f'{"g1":>8} {"e_err":>8} {"mu_err":>8} {"tau_err":>8} {"pi_err":>8} {"p_err":>8} {"p_mass":>10}')
print('-' * 70)

best_g1 = None
best_proton_err = float('inf')

for g1 in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 15000]:
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=g1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
    mu = solver.solve_lepton_self_consistent(n_target=2, max_iter_outer=30, max_iter_nl=0)
    tau = solver.solve_lepton_self_consistent(n_target=3, max_iter_outer=30, max_iter_nl=0)
    pion = solver.solve_meson_self_consistent(quark_wells=(1, 2))
    proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3))
    
    beta = 0.000511 / (e.structure_norm ** 2)
    
    def err(p, exp): 
        return (beta * p.structure_norm**2 * 1000 - exp) / exp * 100
    
    e_e = err(e, 0.511)
    mu_e = err(mu, 105.66)
    tau_e = err(tau, 1776.86)
    pi_e = err(pion, 139.57)
    p_e = err(proton, 938.27)
    p_mass = beta * proton.structure_norm**2 * 1000
    
    print(f'{g1:>8} {e_e:>+7.1f}% {mu_e:>+7.1f}% {tau_e:>+7.1f}% {pi_e:>+7.1f}% {p_e:>+7.1f}% {p_mass:>10.1f}')
    
    if abs(p_e) < abs(best_proton_err):
        best_proton_err = p_e
        best_g1 = g1

print()
print(f'Best g1 = {best_g1} with proton error = {best_proton_err:+.1f}%')

