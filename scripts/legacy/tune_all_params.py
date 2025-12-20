#!/usr/bin/env python3
"""
Tune all relevant parameters to match proton and neutron masses within 5%.

From previous tuning:
- g2=0.035, lambda_so=0.2 gives correct mass difference (~1.30 MeV)
- But absolute masses are ~27% low

Now tune alpha and g_internal to fix absolute masses.
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

# Best g2 and lambda_so from previous tuning
BEST_G2 = 0.035
BEST_LAMBDA_SO = 0.2

def compute_baryon_masses(alpha, g_internal, g2, lambda_so, verbose=False):
    """Compute proton and neutron masses."""
    solver = NonSeparableWavefunctionSolver(
        alpha=alpha, beta=BETA, g_internal=g_internal, g1=G1,
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
        print(f"  Delta m: {delta_m:.2f} MeV (target: {DELTA_M_EXP:.2f} MeV, error {100*abs(delta_m-DELTA_M_EXP)/DELTA_M_EXP:.1f}%)")
    
    return m_proton, m_neutron, delta_m

print("=" * 70)
print("TUNING alpha AND g_internal FOR ABSOLUTE BARYON MASSES")
print("=" * 70)
print(f"Fixed: g2={BEST_G2}, lambda_so={BEST_LAMBDA_SO}")
print(f"Target: m_p={M_PROTON_EXP:.2f} MeV, m_n={M_NEUTRON_EXP:.2f} MeV")
print()

# Current values give masses ~27% low (690 vs 938)
# Need to increase A by factor of sqrt(938/690) = 1.166
# Since A depends on alpha and g_internal, try increasing alpha

print("=== Current values ===")
print(f"alpha={ALPHA}, g_internal={G_INTERNAL}")
compute_baryon_masses(ALPHA, G_INTERNAL, BEST_G2, BEST_LAMBDA_SO, verbose=True)
print()

# Scan alpha (larger alpha -> more coupling -> larger A)
print("=== Scanning alpha ===")
best_alpha = ALPHA
best_err = float('inf')
for alpha_factor in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    alpha = ALPHA * alpha_factor
    m_p, m_n, delta_m = compute_baryon_masses(alpha, G_INTERNAL, BEST_G2, BEST_LAMBDA_SO)
    err_p = abs(m_p - M_PROTON_EXP) / M_PROTON_EXP
    err_n = abs(m_n - M_NEUTRON_EXP) / M_NEUTRON_EXP
    err = (err_p + err_n) / 2
    print(f"  alpha={alpha:.3f}: m_p={m_p:.1f}, m_n={m_n:.1f}, avg_err={100*err:.1f}%")
    if err < best_err:
        best_err = err
        best_alpha = alpha
print(f"  Best: alpha={best_alpha}")
print()

# Scan g_internal (smaller g_internal -> larger A from self-confinement)
print("=== Scanning g_internal ===")
best_g_int = G_INTERNAL
best_err = float('inf')
for g_factor in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]:
    g_int = G_INTERNAL * g_factor
    m_p, m_n, delta_m = compute_baryon_masses(best_alpha, g_int, BEST_G2, BEST_LAMBDA_SO)
    err_p = abs(m_p - M_PROTON_EXP) / M_PROTON_EXP
    err_n = abs(m_n - M_NEUTRON_EXP) / M_NEUTRON_EXP
    err = (err_p + err_n) / 2
    print(f"  g_internal={g_int:.6f}: m_p={m_p:.1f}, m_n={m_n:.1f}, avg_err={100*err:.1f}%")
    if err < best_err:
        best_err = err
        best_g_int = g_int
print(f"  Best: g_internal={best_g_int}")
print()

# Fine-tune all parameters together
print("=== Fine-tuning all parameters ===")
best_params = (best_alpha, best_g_int, BEST_G2, BEST_LAMBDA_SO)
best_total_err = float('inf')

# Grid search
alpha_range = np.linspace(best_alpha * 0.9, best_alpha * 1.1, 5)
g_int_range = np.linspace(best_g_int * 0.8, best_g_int * 1.2, 5)
g2_range = np.linspace(BEST_G2 * 0.8, BEST_G2 * 1.2, 3)
lambda_range = np.linspace(BEST_LAMBDA_SO * 0.8, BEST_LAMBDA_SO * 1.2, 3)

for alpha in alpha_range:
    for g_int in g_int_range:
        for g2 in g2_range:
            for lambda_so in lambda_range:
                try:
                    m_p, m_n, delta_m = compute_baryon_masses(alpha, g_int, g2, lambda_so)
                    err_p = abs(m_p - M_PROTON_EXP) / M_PROTON_EXP
                    err_n = abs(m_n - M_NEUTRON_EXP) / M_NEUTRON_EXP
                    err_delta = abs(delta_m - DELTA_M_EXP) / DELTA_M_EXP
                    total_err = err_p + err_n + err_delta
                    if total_err < best_total_err:
                        best_total_err = total_err
                        best_params = (alpha, g_int, g2, lambda_so)
                except:
                    pass

print(f"  Best: alpha={best_params[0]:.4f}, g_internal={best_params[1]:.6f}")
print(f"        g2={best_params[2]:.4f}, lambda_so={best_params[3]:.4f}")
print()

# Final results
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Parameters: alpha={best_params[0]:.4f}, g_internal={best_params[1]:.6f}")
print(f"            g2={best_params[2]:.4f}, lambda_so={best_params[3]:.4f}")
print()
m_p, m_n, delta_m = compute_baryon_masses(*best_params, verbose=True)

err_p = 100 * abs(m_p - M_PROTON_EXP) / M_PROTON_EXP
err_n = 100 * abs(m_n - M_NEUTRON_EXP) / M_NEUTRON_EXP
err_delta = 100 * abs(delta_m - DELTA_M_EXP) / DELTA_M_EXP

print()
if err_p < 5 and err_n < 5 and err_delta < 5:
    print("[SUCCESS] All errors within 5%!")
    print(f"  Proton error:     {err_p:.2f}%")
    print(f"  Neutron error:    {err_n:.2f}%")
    print(f"  Mass diff error:  {err_delta:.2f}%")
else:
    print(f"[PROGRESS] Errors: proton={err_p:.1f}%, neutron={err_n:.1f}%, delta={err_delta:.1f}%")



