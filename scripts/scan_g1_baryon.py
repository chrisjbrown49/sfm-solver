"""
Scan g1 values for baryon SCF solver to find optimal value.

Test different g1 values to see which gives correct proton mass (~938 MeV).
"""

import numpy as np
import sys
sys.path.insert(0, str(__file__).replace('scripts\\scan_g1_baryon.py', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL

# First compute beta from electron
print("Computing beta from electron...")
solver_electron = NonSeparableWavefunctionSolver(
    alpha=ALPHA, g_internal=G_INTERNAL, g1=5000.0,
)
electron = solver_electron.solve_lepton_self_consistent(n_target=1, k_winding=1, max_iter_outer=30, verbose=False)
m_electron_exp = 0.511  # MeV
A_electron = electron.structure_norm
beta = m_electron_exp / (A_electron**2 * 1000)  # GeV

print("="*70)
print("G1 PARAMETER SCAN FOR BARYON SCF SOLVER")
print("="*70)
print(f"Parameters: ALPHA={ALPHA:.6f}, G_INTERNAL={G_INTERNAL:.8f}")
print(f"Beta (from electron): {beta:.6f} GeV")
print(f"Electron: A={A_electron:.6f}\n")

# Test range of g1 values
g1_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

results = []

for g1 in g1_values:
    print(f"\n{'='*70}")
    print(f"Testing g1 = {g1}")
    print(f"{'='*70}")
    
    try:
        solver = NonSeparableWavefunctionSolver(
            alpha=ALPHA,
            g_internal=G_INTERNAL,
            g1=g1,
            g2=70.0,
            lambda_so=0.2,
            V0=1.0,
            V1=0.0,
        )
        
        # Solve for proton (uud)
        proton = solver.solve_baryon_self_consistent(
            quark_wells=(1, 2, 3),
            quark_windings=(5, 5, -3),  # uud
            max_iter_outer=30,
            verbose=False,
        )
        
        # Calculate mass
        A_proton = proton.structure_norm
        m_proton = beta * A_proton**2 * 1000  # Convert to MeV
        m_exp = 938.27  # MeV
        error = (m_proton - m_exp) / m_exp * 100
        
        print(f"  Proton: A = {A_proton:.4f}, m = {m_proton:.2f} MeV")
        print(f"  Error: {error:+.2f}%")
        
        results.append({
            'g1': g1,
            'A': A_proton,
            'm': m_proton,
            'error': error
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            'g1': g1,
            'A': 0,
            'm': 0,
            'error': 999
        })

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'g1':>10s} {'A':>10s} {'m (MeV)':>12s} {'Error (%)':>12s}")
print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*12}")

for r in results:
    if r['m'] > 0:
        print(f"{r['g1']:>10.2f} {r['A']:>10.4f} {r['m']:>12.2f} {r['error']:>+12.2f}")
    else:
        print(f"{r['g1']:>10.2f} {'FAILED':>10s} {'FAILED':>12s} {'FAILED':>12s}")

# Find best match
valid_results = [r for r in results if r['m'] > 0]
if valid_results:
    best = min(valid_results, key=lambda x: abs(x['error']))
    print(f"\nBest match: g1 = {best['g1']:.2f}")
    print(f"  Proton mass: {best['m']:.2f} MeV (error: {best['error']:+.2f}%)")

