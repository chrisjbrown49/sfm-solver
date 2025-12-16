"""
Scan g1 parameter to find optimal value for baryon masses.

g1 controls the nonlinear self-interaction: V_mean = g1 * |chi_composite|^2

Expected behavior:
- g1 > 0 (large): Strong repulsion -> low amplitude
- g1 > 0 (small): Weak repulsion -> moderate amplitude  
- g1 = 0: No interaction -> natural amplitude from coupling
- g1 < 0: Attraction -> potentially high amplitude
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver

# Test g1 values - scanning around current value of 50
g1_values = [
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,   # Current value
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    150.0,
    200.0,
]

print("="*80)
print("G1 PARAMETER SCAN FOR BARYON MASSES (AFTER BETA REMOVAL)")
print("="*80)
print("\nTarget masses:")
print("  Proton:  938.27 MeV")
print("  Neutron: 939.57 MeV")
print("\nNote: Using g2=70.0 from constants.json")
print("="*80)

# First get beta from electron (g1 shouldn't affect electron much)
print("\nDeriving beta from electron...")
solver_ref = NonSeparableWavefunctionSolver(alpha=10.5, g1=100.0, g2=70.0)
e_ref = solver_ref.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
beta = 0.511 / e_ref.structure_norm**2
print(f"  Electron: A = {e_ref.structure_norm:.6f}")
print(f"  beta = {beta:.6e} MeV")

results = []
scan_start_time = time.time()

total_tests = len(g1_values)
for idx, g1 in enumerate(g1_values, 1):
    print(f"\n{'='*80}")
    print(f"Testing g1 = {g1:+.1f} ({idx}/{total_tests})")
    print(f"{'='*80}")
    
    try:
        # Create solver with this g1 (no beta parameter - it's removed!)
        solver = NonSeparableWavefunctionSolver(
            alpha=10.5,
            g1=g1,
            g2=70.0,  # From constants.json
            lambda_so=0.2,
        )
        
        # Solve proton
        print("\nSolving Proton (uud)...")
        start_time = time.time()
        proton = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, 5, -3),
            max_iter_outer=30,
            max_iter_scf=5,
            verbose=True,
        )
        print(f"  Proton solved in {time.time() - start_time:.1f}s")
        
        # Solve neutron
        print("\nSolving Neutron (udd)...")
        start_time = time.time()
        neutron = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, -3, -3),
            max_iter_outer=30,
            max_iter_scf=5,
            verbose=True,
        )
        print(f"  Neutron solved in {time.time() - start_time:.1f}s")
        
        # Extract results
        A_p = proton.structure_norm
        A_n = neutron.structure_norm
        m_p = beta * A_p**2  # MeV (beta already in MeV units)
        m_n = beta * A_n**2  # MeV
        Dx_p = proton.delta_x_final
        Dx_n = neutron.delta_x_final
        
        err_p = 100 * (m_p - 938.27) / 938.27
        err_n = 100 * (m_n - 939.57) / 939.57
        
        print(f"\nResults:")
        print(f"  Proton:  A={A_p:.4f}, m={m_p:.2f} MeV (err={err_p:+.1f}%), Dx={Dx_p:.4f}")
        print(f"  Neutron: A={A_n:.4f}, m={m_n:.2f} MeV (err={err_n:+.1f}%), Dx={Dx_n:.4f}")
        
        # Estimate time remaining
        elapsed = time.time() - scan_start_time
        avg_time_per_test = elapsed / idx
        remaining_tests = total_tests - idx
        est_remaining = avg_time_per_test * remaining_tests
        print(f"\nProgress: {idx}/{total_tests} tests complete. Estimated time remaining: {est_remaining/60:.1f} min")
        
        results.append({
            'g1': g1,
            'A_p': A_p,
            'A_n': A_n,
            'm_p': m_p,
            'm_n': m_n,
            'err_p': err_p,
            'err_n': err_n,
            'Dx_p': Dx_p,
            'Dx_n': Dx_n,
            'converged': proton.converged and neutron.converged,
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            'g1': g1,
            'A_p': np.nan,
            'A_n': np.nan,
            'm_p': np.nan,
            'm_n': np.nan,
            'err_p': np.nan,
            'err_n': np.nan,
            'Dx_p': np.nan,
            'Dx_n': np.nan,
            'converged': False,
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'g1':>10s} {'A_p':>8s} {'A_n':>8s} {'m_p(MeV)':>10s} {'m_n(MeV)':>10s} {'err_p(%)':>10s} {'err_n(%)':>10s} {'Dx_p':>8s}")
print("-"*80)

for r in results:
    if np.isnan(r['A_p']):
        print(f"{r['g1']:>+10.1f}  {'FAILED':^60s}")
    else:
        print(f"{r['g1']:>+10.1f} {r['A_p']:>8.4f} {r['A_n']:>8.4f} {r['m_p']:>10.2f} "
              f"{r['m_n']:>10.2f} {r['err_p']:>+10.1f} {r['err_n']:>+10.1f} {r['Dx_p']:>8.4f}")

# Find best
valid_results = [r for r in results if not np.isnan(r['err_p'])]
if valid_results:
    # Best by proton error
    best_p = min(valid_results, key=lambda x: abs(x['err_p']))
    best_n = min(valid_results, key=lambda x: abs(x['err_n']))
    best_avg = min(valid_results, key=lambda x: (abs(x['err_p']) + abs(x['err_n']))/2)
    
    print("\n" + "="*80)
    print("OPTIMAL VALUES")
    print("="*80)
    print(f"\nBest for proton:")
    print(f"  g1 = {best_p['g1']:+.1f}")
    print(f"  Proton:  A={best_p['A_p']:.4f}, m={best_p['m_p']:.2f} MeV (err={best_p['err_p']:+.1f}%)")
    print(f"  Neutron: A={best_p['A_n']:.4f}, m={best_p['m_n']:.2f} MeV (err={best_p['err_n']:+.1f}%)")
    
    print(f"\nBest for neutron:")
    print(f"  g1 = {best_n['g1']:+.1f}")
    print(f"  Proton:  A={best_n['A_p']:.4f}, m={best_n['m_p']:.2f} MeV (err={best_n['err_p']:+.1f}%)")
    print(f"  Neutron: A={best_n['A_n']:.4f}, m={best_n['m_n']:.2f} MeV (err={best_n['err_n']:+.1f}%)")
    
    print(f"\nBest overall (avg error):")
    print(f"  g1 = {best_avg['g1']:+.1f}")
    print(f"  Proton:  A={best_avg['A_p']:.4f}, m={best_avg['m_p']:.2f} MeV (err={best_avg['err_p']:+.1f}%)")
    print(f"  Neutron: A={best_avg['A_n']:.4f}, m={best_avg['m_n']:.2f} MeV (err={best_avg['err_n']:+.1f}%)")
    
    print("\n" + "="*80)
