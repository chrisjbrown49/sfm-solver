#!/usr/bin/env python3
"""
Test electromagnetic effects on baryon mass predictions.

FIRST-PRINCIPLES TEST:
=====================
The EM self-energy emerges from the circulation integral:
    E_EM = g2 |J|^2 where J = integral of chi* d(chi)/d(sigma) dsigma

Quark winding numbers:
    - u quark: k = +5 -> Q = +2/3 e
    - d quark: k = -3 -> Q = -1/3 e

Baryons:
    - Proton (uud): k = (+5, +5, -3) -> net k = +7 -> Q = +1 e
    - Neutron (udd): k = (+5, -3, -3) -> net k = -1 -> Q = 0

Expected:
    - Proton has larger |J|^2 -> larger E_EM -> heavier
    - But experimentally neutron is ~1.3 MeV heavier (quark mass effect dominates)
    
This test verifies the EM contribution is computed correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1, G2, LAMBDA_SO

print("=" * 70)
print("BARYON ELECTROMAGNETIC SELF-ENERGY TEST")
print("=" * 70)
print(f"Parameters: ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}, G1={G1}")
print(f"g2 = {G2} (circulation coupling)")
print(f"lambda_so = {LAMBDA_SO} (spin-orbit coupling)")
print()

# Calibrate beta from electron mass (0.511 MeV)
BETA = 0.000511

solver = NonSeparableWavefunctionSolver(
    alpha=ALPHA, beta=BETA, g_internal=G_INTERNAL, g1=G1,
    g2=G2, lambda_so=LAMBDA_SO, V0=1.0, V1=0.0, n_max=5, l_max=2, N_sigma=64,
)

# Solve for electron to derive beta
print("Solving for electron (calibration)...")
e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
beta = 0.000511 / (e.structure_norm ** 2)
print(f"  beta = {beta:.6f} GeV")
print()

# Solve for proton (uud) with EM
print("=== PROTON (uud) ===")
print("Quark windings: k = (+5, +5, -3) -> net k = +7")
proton = solver.solve_baryon_self_consistent(
    quark_wells=(1, 2, 3),
    quark_windings=(5, 5, -3),  # uud
    verbose=True,
)

proton_mass = beta * proton.structure_norm**2 * 1000  # MeV
proton_err = (proton_mass - 938.27) / 938.27 * 100
print(f"\nProton: A={proton.structure_norm:.4f}, mass={proton_mass:.2f} MeV, error={proton_err:+.1f}%")
print(f"  k_eff = {proton.k_eff:.4f}")
print(f"  k_winding = {proton.k_winding}")
print(f"  E_EM = {proton.em_energy:.4e}")
print(f"  |J|^2 = {np.abs(proton.circulation)**2:.4e}")
print()

# Solve for neutron (udd) with EM
print("=== NEUTRON (udd) ===")
print("Quark windings: k = (+5, -3, -3) -> net k = -1")
neutron = solver.solve_baryon_self_consistent(
    quark_wells=(1, 2, 3),
    quark_windings=(5, -3, -3),  # udd
    verbose=True,
)

neutron_mass = beta * neutron.structure_norm**2 * 1000  # MeV
neutron_err = (neutron_mass - 939.57) / 939.57 * 100
print(f"\nNeutron: A={neutron.structure_norm:.4f}, mass={neutron_mass:.2f} MeV, error={neutron_err:+.1f}%")
print(f"  k_eff = {neutron.k_eff:.4f}")
print(f"  k_winding = {neutron.k_winding}")
print(f"  E_EM = {neutron.em_energy:.4e}")
print(f"  |J|^2 = {np.abs(neutron.circulation)**2:.4e}")
print()

# Compare
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"                    Proton          Neutron         Difference")
print(f"  Amplitude A:      {proton.structure_norm:.4f}          {neutron.structure_norm:.4f}          {neutron.structure_norm - proton.structure_norm:+.4f}")
print(f"  Mass (MeV):       {proton_mass:.2f}          {neutron_mass:.2f}          {neutron_mass - proton_mass:+.2f}")
print(f"  k_eff:            {proton.k_eff:+.4f}          {neutron.k_eff:+.4f}          {neutron.k_eff - proton.k_eff:+.4f}")
print(f"  |J|^2:            {np.abs(proton.circulation)**2:.4e}      {np.abs(neutron.circulation)**2:.4e}      {np.abs(neutron.circulation)**2 - np.abs(proton.circulation)**2:+.4e}")
print(f"  E_EM:             {proton.em_energy:.4e}      {neutron.em_energy:.4e}      {neutron.em_energy - proton.em_energy:+.4e}")
print()
print(f"Experimental mass difference: {939.57 - 938.27:.2f} MeV (neutron heavier)")
print(f"Predicted mass difference:    {neutron_mass - proton_mass:+.2f} MeV")
print()

# Check if EM contribution differentiates them
if abs(neutron.em_energy - proton.em_energy) > 1e-10:
    print("[OK] EM self-energy differentiates proton from neutron!")
    print(f"     Proton has {'larger' if proton.em_energy > neutron.em_energy else 'smaller'} E_EM")
else:
    print("[INFO] EM self-energy is similar for both baryons")

print()
print("=" * 70)
print("PHYSICS INTERPRETATION")
print("=" * 70)
print("""
The EM self-energy E_EM = g2*|J|^2 depends on the circulation integral J.
Proton (k_net=+7) and Neutron (k_net=-1) have different circulation patterns
due to their different quark charge compositions.

NOTE: The ~1.3 MeV experimental mass difference is primarily from quark masses
(d quark heavier than u quark), not EM effects. The EM contribution is ~O(1 MeV)
but works in the opposite direction (charged proton has higher E_EM).
""")

