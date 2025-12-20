#!/usr/bin/env python3
"""Test all particles - calibration and validation - with current settings."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1

print("=" * 70)
print("ALL PARTICLES TEST - Current Configuration")
print("=" * 70)
print(f"ALPHA={ALPHA}, G_INTERNAL={G_INTERNAL}, G1={G1}")
print("V1=0.0 (default), quark_wells: meson=(1,2), baryon=(1,2,3)")
print()

solver = NonSeparableWavefunctionSolver(
    alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
    g2=0.004, V0=1.0, V1=0.0, n_max=5, l_max=2, N_sigma=64,
)

# Leptons
e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
mu = solver.solve_lepton_self_consistent(n_target=2, max_iter_outer=30, max_iter_nl=0)
tau = solver.solve_lepton_self_consistent(n_target=3, max_iter_outer=30, max_iter_nl=0)

# Derive beta from electron
beta = 0.000511 / (e.structure_norm ** 2)

# Mesons - Calibration
pion_plus = solver.solve_meson_self_consistent(quark_wells=(1, 2), n_radial=1)

# Mesons - Validation (different quark content / radial excitations)
pion_minus = solver.solve_meson_self_consistent(quark_wells=(1, 2), n_radial=1)  # Same structure as pi+
pion_zero = solver.solve_meson_self_consistent(quark_wells=(1, 2), n_radial=1)   # Slightly lighter
jpsi = solver.solve_meson_self_consistent(quark_wells=(1, 2), n_radial=1)        # c-cbar ground state
psi_2S = solver.solve_meson_self_consistent(quark_wells=(1, 2), n_radial=2)      # c-cbar excited
upsilon_1S = solver.solve_meson_self_consistent(quark_wells=(1, 2), n_radial=1)  # b-bbar ground state
upsilon_2S = solver.solve_meson_self_consistent(quark_wells=(1, 2), n_radial=2)  # b-bbar excited

# Baryons
proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3))
neutron = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3))  # Same structure as proton

print(f"beta_derived = {beta:.6f} GeV")
print()
print("=== CALIBRATION PARTICLES ===")
print(f"{'Particle':<14} {'A':>8} {'dx':>10} {'ds':>8} {'Pred MeV':>10} {'Exp MeV':>10} {'Err%':>8}")
print("-" * 76)

calibration = [
    ('Electron', e, 0.511),
    ('Muon', mu, 105.66),
    ('Tau', tau, 1776.86),
    ('Pion+', pion_plus, 139.57),
    ('Proton', proton, 938.27),
]

for name, p, exp in calibration:
    pred = beta * p.structure_norm**2 * 1000
    err = (pred - exp) / exp * 100
    ds = p.delta_sigma_final if p.delta_sigma_final else 0.5
    print(f"{name:<14} {p.structure_norm:>8.4f} {p.delta_x_final:>10.6f} {ds:>8.4f} {pred:>10.2f} {exp:>10.2f} {err:>+7.1f}%")

print()
print("=== VALIDATION PARTICLES ===")
print(f"{'Particle':<14} {'A':>8} {'dx':>10} {'ds':>8} {'Pred MeV':>10} {'Exp MeV':>10} {'Err%':>8}")
print("-" * 76)

validation = [
    ('Pion-', pion_minus, 139.57),
    ('Pion0', pion_zero, 134.98),
    ('J/Psi', jpsi, 3096.90),
    ('Psi(2S)', psi_2S, 3686.10),
    ('Upsilon(1S)', upsilon_1S, 9460.30),
    ('Upsilon(2S)', upsilon_2S, 10023.26),
    ('Neutron', neutron, 939.57),
]

for name, p, exp in validation:
    pred = beta * p.structure_norm**2 * 1000
    err = (pred - exp) / exp * 100
    ds = p.delta_sigma_final if p.delta_sigma_final else 0.5
    print(f"{name:<14} {p.structure_norm:>8.4f} {p.delta_x_final:>10.6f} {ds:>8.4f} {pred:>10.2f} {exp:>10.2f} {err:>+7.1f}%")

print()
print("=" * 76)
print("Note: Current meson solver uses same structure for all mesons.")
print("      Heavy quark effects (c, b) require quark mass implementation.")
print("=" * 76)

