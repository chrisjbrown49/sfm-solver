"""
Diagnostic: Compare scan script vs test script with identical parameters.
This will help identify why results differ.
"""
import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1, G2, LAMBDA_SO, V0, V1

print("="*70)
print("DIAGNOSTIC: Scan Script vs Test Script Comparison")
print("="*70)
print(f"\nUsing constants from constants.json:")
print(f"  ALPHA      = {ALPHA}")
print(f"  G_INTERNAL = {G_INTERNAL}")
print(f"  G1         = {G1}")
print(f"  G2         = {G2}")
print(f"  LAMBDA_SO  = {LAMBDA_SO}")
print(f"  V0         = {V0}")
print(f"  V1         = {V1}")

# Initialize solver EXACTLY as in test script
print("\n" + "="*70)
print("Creating solver (test script style)")
print("="*70)
solver = NonSeparableWavefunctionSolver(
    alpha=ALPHA, 
    g_internal=G_INTERNAL, 
    g1=G1,
    g2=G2, 
    lambda_so=LAMBDA_SO, 
    V0=V0, 
    V1=V1, 
    n_max=5, 
    l_max=2, 
    N_sigma=64,
)

# Get beta from electron
print("\nDeriving beta from electron...")
e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
beta = 0.000511 / (e.structure_norm ** 2)  # GeV
beta_MeV = beta * 1000
print(f"  Electron: A = {e.structure_norm:.6f}")
print(f"  beta = {beta:.6e} GeV = {beta_MeV:.6e} MeV")

# Solve proton
print("\n" + "="*70)
print("PROTON (uud) - Test #1")
print("="*70)
proton1 = solver.solve_baryon_4D_self_consistent(
    quark_wells=(1, 2, 3),
    color_phases=(0, 2*np.pi/3, 4*np.pi/3),
    quark_windings=(5, 5, -3),
    max_iter_outer=30,
    max_iter_scf=10,
    verbose=False,
)
A_p1 = proton1.structure_norm
m_p1_test = beta * A_p1**2 * 1000  # GeV to MeV
err_p1 = 100 * (m_p1_test - 938.27) / 938.27

print(f"Results (test script style):")
print(f"  A = {A_p1:.6f}")
print(f"  mass = {m_p1_test:.2f} MeV (error = {err_p1:+.1f}%)")
print(f"  Delta_x = {proton1.delta_x_final:.6f}")
print(f"  converged = {proton1.converged}")

# Solve proton again (test repeatability)
print("\n" + "="*70)
print("PROTON (uud) - Test #2 (repeatability check)")
print("="*70)
proton2 = solver.solve_baryon_4D_self_consistent(
    quark_wells=(1, 2, 3),
    color_phases=(0, 2*np.pi/3, 4*np.pi/3),
    quark_windings=(5, 5, -3),
    max_iter_outer=30,
    max_iter_scf=10,
    verbose=False,
)
A_p2 = proton2.structure_norm
m_p2_test = beta * A_p2**2 * 1000
err_p2 = 100 * (m_p2_test - 938.27) / 938.27

print(f"Results (second run):")
print(f"  A = {A_p2:.6f}")
print(f"  mass = {m_p2_test:.2f} MeV (error = {err_p2:+.1f}%)")
print(f"  Delta_x = {proton2.delta_x_final:.6f}")
print(f"  converged = {proton2.converged}")

# Check difference
diff_A = abs(A_p2 - A_p1)
diff_m = abs(m_p2_test - m_p1_test)
print(f"\nRepeatability:")
print(f"  |A2 - A1| = {diff_A:.6f}")
print(f"  |m2 - m1| = {diff_m:.2f} MeV")
if diff_A > 0.01:
    print("  WARNING: Results are not repeatable!")
else:
    print("  OK: Results are consistent")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Expected mass: 938.27 MeV")
print(f"Test #1:       {m_p1_test:.2f} MeV (error = {err_p1:+.1f}%)")
print(f"Test #2:       {m_p2_test:.2f} MeV (error = {err_p2:+.1f}%)")

