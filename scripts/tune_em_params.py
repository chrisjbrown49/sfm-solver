#!/usr/bin/env python3
"""
Tune g2 and lambda_so to match proton-neutron mass splitting.

Target:
- Proton mass: 938.27 MeV
- Neutron mass: 939.57 MeV
- Mass difference: 1.30 MeV (neutron heavier)

Strategy:
- g2 controls the magnitude of E_EM
- lambda_so controls the spin-orbit envelope shift
- Both affect the mass difference
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1

# Experimental values
M_PROTON_EXP = 938.27  # MeV
M_NEUTRON_EXP = 939.57  # MeV
DELTA_M_EXP = M_NEUTRON_EXP - M_PROTON_EXP  # 1.30 MeV

# Fixed beta (from electron calibration)
BETA = 0.000511

def compute_baryon_masses(g2, lambda_so, verbose=False):
    """Compute proton and neutron masses for given g2 and lambda_so."""
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, beta=BETA, g_internal=G_INTERNAL, g1=G1,
        g2=g2, lambda_so=lambda_so, V0=1.0, V1=0.0, 
        n_max=5, l_max=2, N_sigma=64,
    )
    
    # Proton (uud)
    proton = solver.solve_baryon_self_consistent(
        quark_wells=(1, 2, 3),
        quark_windings=(5, 5, -3),
        verbose=False,
    )
    m_proton = BETA * proton.structure_norm**2 * 1000  # MeV
    
    # Neutron (udd)
    neutron = solver.solve_baryon_self_consistent(
        quark_wells=(1, 2, 3),
        quark_windings=(5, -3, -3),
        verbose=False,
    )
    m_neutron = BETA * neutron.structure_norm**2 * 1000  # MeV
    
    delta_m = m_neutron - m_proton
    
    if verbose:
        print(f"  Proton:  {m_proton:.2f} MeV (error {100*(m_proton-M_PROTON_EXP)/M_PROTON_EXP:+.1f}%)")
        print(f"  Neutron: {m_neutron:.2f} MeV (error {100*(m_neutron-M_NEUTRON_EXP)/M_NEUTRON_EXP:+.1f}%)")
        print(f"  Delta m: {delta_m:.2f} MeV (target: {DELTA_M_EXP:.2f} MeV)")
    
    return m_proton, m_neutron, delta_m

def objective(g2, lambda_so):
    """Compute error metric for optimization."""
    m_p, m_n, delta_m = compute_baryon_masses(g2, lambda_so)
    
    # Error in mass difference (primary target)
    err_delta = abs(delta_m - DELTA_M_EXP) / DELTA_M_EXP
    
    # Error in absolute masses (secondary)
    err_p = abs(m_p - M_PROTON_EXP) / M_PROTON_EXP
    err_n = abs(m_n - M_NEUTRON_EXP) / M_NEUTRON_EXP
    
    # Combined error: prioritize mass difference
    total_err = err_delta + 0.5 * (err_p + err_n)
    
    return total_err, err_p, err_n, err_delta, delta_m

print("=" * 70)
print("TUNING g2 AND lambda_so FOR PROTON-NEUTRON MASS SPLITTING")
print("=" * 70)
print(f"Target: m_p={M_PROTON_EXP:.2f} MeV, m_n={M_NEUTRON_EXP:.2f} MeV")
print(f"        delta_m = {DELTA_M_EXP:.2f} MeV")
print()

# Current values
print("=== Current values (g2=0.004, lambda_so=0.1) ===")
compute_baryon_masses(0.004, 0.1, verbose=True)
print()

# Scan lambda_so first (larger effect on envelope shifts)
print("=== Scanning lambda_so (g2=0.004 fixed) ===")
best_lambda = 0.1
best_err = float('inf')
for lambda_so in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    err, err_p, err_n, err_delta, delta_m = objective(0.004, lambda_so)
    print(f"  lambda_so={lambda_so:.2f}: delta_m={delta_m:.3f} MeV, err_delta={100*err_delta:.1f}%")
    if err < best_err:
        best_err = err
        best_lambda = lambda_so
print(f"  Best: lambda_so={best_lambda}")
print()

# Scan g2 with best lambda_so
print(f"=== Scanning g2 (lambda_so={best_lambda} fixed) ===")
best_g2 = 0.004
best_err = float('inf')
for g2 in [0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0]:
    err, err_p, err_n, err_delta, delta_m = objective(g2, best_lambda)
    print(f"  g2={g2:.3f}: delta_m={delta_m:.3f} MeV, err_delta={100*err_delta:.1f}%")
    if err < best_err:
        best_err = err
        best_g2 = g2
print(f"  Best: g2={best_g2}")
print()

# Fine-tune around best values
print(f"=== Fine-tuning around g2={best_g2}, lambda_so={best_lambda} ===")
best_total_err = float('inf')
best_params = (best_g2, best_lambda)

# Grid search around best values
g2_range = np.linspace(best_g2 * 0.5, best_g2 * 2.0, 5)
lambda_range = np.linspace(best_lambda * 0.5, best_lambda * 2.0, 5)

for g2 in g2_range:
    for lambda_so in lambda_range:
        err, err_p, err_n, err_delta, delta_m = objective(g2, lambda_so)
        if err < best_total_err:
            best_total_err = err
            best_params = (g2, lambda_so)

print(f"  Best: g2={best_params[0]:.4f}, lambda_so={best_params[1]:.4f}")
print()

# Final results with best parameters
print("=" * 70)
print(f"FINAL RESULTS: g2={best_params[0]:.4f}, lambda_so={best_params[1]:.4f}")
print("=" * 70)
m_p, m_n, delta_m = compute_baryon_masses(best_params[0], best_params[1], verbose=True)

err_p = 100 * abs(m_p - M_PROTON_EXP) / M_PROTON_EXP
err_n = 100 * abs(m_n - M_NEUTRON_EXP) / M_NEUTRON_EXP
err_delta = 100 * abs(delta_m - DELTA_M_EXP) / DELTA_M_EXP

print()
if err_p < 5 and err_n < 5 and err_delta < 5:
    print("[SUCCESS] All errors within 5%!")
else:
    print(f"[NOTE] Errors: proton={err_p:.1f}%, neutron={err_n:.1f}%, delta={err_delta:.1f}%")
    if err_p > 5 or err_n > 5:
        print("       Absolute mass error requires tuning alpha/g_internal, not g2/lambda_so")



