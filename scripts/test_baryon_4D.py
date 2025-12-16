"""
Test the new 4D three-quark baryon solver.

This solver treats each quark as a full 4D entity with wavefunction Ψᵢ(x,y,z,σ).
The baryon is the 4D superposition: Ψ_baryon = Ψ₁ + Ψ₂ + Ψ₃.
"""
import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1, G2, LAMBDA_SO, V0, V1, BETA

print("="*70)
print("4D THREE-QUARK BARYON SOLVER TEST")
print("="*70)
print()

# First, get electron to derive beta
print("Deriving beta from electron...")
solver = NonSeparableWavefunctionSolver(
    alpha=ALPHA, beta=BETA, g_internal=G_INTERNAL, g1=G1,
    g2=G2, lambda_so=LAMBDA_SO, V0=V0, V1=V1, n_max=5, l_max=2, N_sigma=64,
)

e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
beta = 0.000511 / (e.structure_norm ** 2)
print(f"  Electron: A = {e.structure_norm:.6f}")
print(f"  beta = {beta:.6f} GeV")
print()

# Recreate solver with correct beta
solver = NonSeparableWavefunctionSolver(
    alpha=ALPHA, beta=beta, g_internal=G_INTERNAL, g1=G1,
    g2=G2, lambda_so=LAMBDA_SO, V0=V0, V1=V1, n_max=5, l_max=2, N_sigma=64,
)

# Target masses
m_p_exp = 938.27  # MeV
m_n_exp = 939.57  # MeV
delta_m_exp = 1.30  # MeV

print("="*70)
print("PROTON (uud)")
print("="*70)

proton = solver.solve_baryon_4D_self_consistent(
    quark_wells=(1, 2, 3),
    color_phases=(0, 2*np.pi/3, 4*np.pi/3),
    quark_windings=(5, 5, -3),  # uud
    max_iter_outer=30,
    max_iter_scf=10,
    verbose=True,
)

A_p = proton.structure_norm
m_p = beta * A_p**2 * 1000  # MeV
error_p = 100 * (m_p - m_p_exp) / m_p_exp

print(f"\nProton Results:")
print(f"  A = {A_p:.6f}")
print(f"  mass = {m_p:.2f} MeV (target: {m_p_exp:.2f} MeV)")
print(f"  error = {error_p:+.1f}%")
print(f"  k_eff = {proton.k_eff:.4f}")
print(f"  E_EM = {proton.em_energy:.4e}")
print()

print("="*70)
print("NEUTRON (udd)")
print("="*70)

neutron = solver.solve_baryon_4D_self_consistent(
    quark_wells=(1, 2, 3),
    color_phases=(0, 2*np.pi/3, 4*np.pi/3),
    quark_windings=(5, -3, -3),  # udd
    max_iter_outer=30,
    max_iter_scf=10,
    verbose=True,
)

A_n = neutron.structure_norm
m_n = beta * A_n**2 * 1000  # MeV
error_n = 100 * (m_n - m_n_exp) / m_n_exp

print(f"\nNeutron Results:")
print(f"  A = {A_n:.6f}")
print(f"  mass = {m_n:.2f} MeV (target: {m_n_exp:.2f} MeV)")
print(f"  error = {error_n:+.1f}%")
print(f"  k_eff = {neutron.k_eff:.4f}")
print(f"  E_EM = {neutron.em_energy:.4e}")
print()

print("="*70)
print("COMPARISON")
print("="*70)
print(f"{'':20s} {'Proton':>15s} {'Neutron':>15s} {'Difference':>15s}")
print("-"*70)
print(f"{'Amplitude A:':20s} {A_p:>15.6f} {A_n:>15.6f} {A_n-A_p:>+15.6f}")
print(f"{'Mass (MeV):':20s} {m_p:>15.2f} {m_n:>15.2f} {m_n-m_p:>+15.2f}")
print(f"{'Error (%):':20s} {error_p:>+15.1f} {error_n:>+15.1f}")
print(f"{'k_eff:':20s} {proton.k_eff:>+15.4f} {neutron.k_eff:>+15.4f} {neutron.k_eff-proton.k_eff:>+15.4f}")
print(f"{'E_EM:':20s} {proton.em_energy:>15.4e} {neutron.em_energy:>15.4e} {neutron.em_energy-proton.em_energy:>+15.4e}")
print()
print(f"Experimental mass difference: {delta_m_exp:.2f} MeV (neutron heavier)")
print(f"Predicted mass difference:    {m_n-m_p:+.2f} MeV")
print()

# Check if neutron amplitude collapse is fixed
if A_n > 20:
    print("[SUCCESS] Neutron amplitude collapse FIXED!")
    print(f"  Neutron A = {A_n:.2f} (was ~4-7 with old solver)")
else:
    print("[ISSUE] Neutron amplitude still low")
    print(f"  Neutron A = {A_n:.2f}")
print()

# Overall assessment
if abs(error_p) < 10 and abs(error_n) < 10:
    print("[EXCELLENT] Both masses within 10% of experimental values!")
elif abs(error_p) < 20 and abs(error_n) < 20:
    print("[GOOD] Both masses within 20% of experimental values")
elif abs(error_p) < 50 and abs(error_n) < 50:
    print("[FAIR] Both masses within 50% of experimental values")
else:
    print("[NEEDS WORK] One or both masses still far from target")

