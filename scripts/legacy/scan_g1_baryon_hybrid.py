"""
Scan g1 values for the hybrid SCF + spatial coupling baryon solver.
"""
import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G2, LAMBDA_SO, V0, V1

# First, solve for electron to get beta
print("Solving for electron to derive beta...")
solver_e = NonSeparableWavefunctionSolver(
    alpha=ALPHA,
    beta=0.000511,  # Temporary value
    g_internal=G_INTERNAL,
    g1=1.0,  # Use g1=1.0 for electron (standard)
    g2=G2,
    lambda_so=LAMBDA_SO,
    V0=V0,
    V1=V1,
    n_max=5,
    l_max=2,
    N_sigma=64,
)
electron = solver_e.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0, verbose=False)
m_e_exp = 0.511  # MeV
A_e = electron.structure_norm
beta = 0.000511 / (A_e**2)  # GeV (m_e in GeV / A^2)

print("=" * 70)
print("G1 SCAN FOR HYBRID BARYON SOLVER")
print("=" * 70)
print(f"Alpha: {ALPHA:.6f}")
print(f"G_internal: {G_INTERNAL:.8f}")
print(f"Electron: A = {A_e:.6f}")
print(f"Beta (from electron): {beta:.6f} GeV")
print()

# Target masses
m_p_exp = 938.27  # MeV
m_n_exp = 939.57  # MeV

# Test g1 values
g1_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

results = []

for g1 in g1_values:
    print(f"\n{'='*70}")
    print(f"Testing g1 = {g1}")
    print(f"{'='*70}")
    
    try:
        solver = NonSeparableWavefunctionSolver(
            alpha=ALPHA,
            beta=beta,
            g_internal=G_INTERNAL,
            g1=g1,
            g2=G2,
            lambda_so=LAMBDA_SO,
            V0=V0,
            V1=V1,
            n_max=5,
            l_max=2,
            N_sigma=64,
        )
        
        # Solve proton
        proton = solver.solve_baryon_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, 5, -3),  # uud
            max_iter_outer=30,
            verbose=False,
        )
        
        # Compute mass
        A_p = proton.structure_norm
        m_p = beta * A_p**2 * 1000  # GeV -> MeV
        error_p = 100 * (m_p - m_p_exp) / m_p_exp
        
        print(f"Proton: A = {A_p:.4f}, mass = {m_p:.2f} MeV, error = {error_p:+.1f}%")
        
        results.append({
            'g1': g1,
            'A_p': A_p,
            'm_p': m_p,
            'error_p': error_p,
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            'g1': g1,
            'A_p': np.nan,
            'm_p': np.nan,
            'error_p': np.nan,
        })

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'g1':>10s} {'A_p':>10s} {'m_p (MeV)':>12s} {'Error (%)':>12s}")
print(f"{'-'*70}")

for r in results:
    if np.isnan(r['A_p']):
        print(f"{r['g1']:>10.1f} {'FAILED':>10s} {'FAILED':>12s} {'FAILED':>12s}")
    else:
        print(f"{r['g1']:>10.1f} {r['A_p']:>10.4f} {r['m_p']:>12.2f} {r['error_p']:>+12.1f}")

# Find best
valid_results = [r for r in results if not np.isnan(r['error_p'])]
if valid_results:
    best = min(valid_results, key=lambda x: abs(x['error_p']))
    print(f"\nBest g1 = {best['g1']}: mass = {best['m_p']:.2f} MeV (error = {best['error_p']:+.1f}%)")

