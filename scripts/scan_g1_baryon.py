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
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G2, LAMBDA_SO, V0, V1

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
print("G1 PARAMETER SCAN FOR BARYON MASSES (FIXED VERSION)")
print("="*80)
print("\nTarget masses:")
print("  Proton:  938.27 MeV")
print("  Neutron: 939.57 MeV")
print(f"\nUsing g2={G2} from constants.json")
print("\nFIXES APPLIED:")
print("  1. Beta derived with SAME g1 being tested (not hardcoded 100)")
print("  2. max_iter_scf increased from 5 to 10")
print("  3. Convergence checks added")
print("="*80)

results = []
scan_start_time = time.time()

total_tests = len(g1_values)
for idx, g1 in enumerate(g1_values, 1):
    print(f"\n{'='*80}")
    print(f"Testing g1 = {g1:+.1f} ({idx}/{total_tests})")
    print(f"{'='*80}")
    
    try:
        # FIX #1: Create solver with this g1 for BOTH electron and baryons
        solver = NonSeparableWavefunctionSolver(
            alpha=ALPHA,
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
        
        # Derive beta with the SAME g1
        print(f"\nDeriving beta with g1={g1}...")
        e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
        beta = 0.511 / e.structure_norm**2
        print(f"  Electron: A = {e.structure_norm:.6f}, beta = {beta:.6e} MeV")
        
        # FIX #2: Increased max_iter_scf from 5 to 10
        # Solve proton
        print("\nSolving Proton (uud)...")
        start_time = time.time()
        proton = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, 5, -3),
            max_iter_outer=30,
            max_iter_scf=10,  # Increased from 5
            verbose=False,  # Less verbose to reduce output
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
            max_iter_scf=10,  # Increased from 5
            verbose=False,  # Less verbose to reduce output
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
        
        # FIX #3: Add convergence checks and warnings
        converged_p = proton.converged
        converged_n = neutron.converged
        
        print(f"\nResults:")
        print(f"  Proton:  A={A_p:.4f}, m={m_p:.2f} MeV (err={err_p:+.1f}%), Dx={Dx_p:.4f}, converged={converged_p}")
        print(f"  Neutron: A={A_n:.4f}, m={m_n:.2f} MeV (err={err_n:+.1f}%), Dx={Dx_n:.4f}, converged={converged_n}")
        
        # Check for pathological solutions
        warnings = []
        if not converged_p or not converged_n:
            warnings.append("NON-CONVERGENT")
        if Dx_p < 0.001 or Dx_n < 0.001:
            warnings.append("PATHOLOGICAL Dx")
        if A_p > 100 or A_n > 100:
            warnings.append("EXTREME AMPLITUDE")
        if proton.em_energy > 1e6 or neutron.em_energy > 1e6:
            warnings.append("EM EXPLOSION")
        
        if warnings:
            print(f"  ⚠️  WARNING: {', '.join(warnings)}")
        
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
            'converged': converged_p and converged_n,
            'warnings': len(warnings) > 0,
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
print(f"\n{'g1':>10s} {'A_p':>8s} {'A_n':>8s} {'m_p(MeV)':>10s} {'m_n(MeV)':>10s} {'err_p(%)':>10s} {'err_n(%)':>10s} {'Conv':>6s} {'Status':>10s}")
print("-"*80)

for r in results:
    if np.isnan(r['A_p']):
        print(f"{r['g1']:>+10.1f}  {'FAILED':^70s}")
    else:
        conv_str = 'YES' if r['converged'] else 'NO'
        warn_str = '⚠️ ISSUE' if r['warnings'] else 'OK'
        print(f"{r['g1']:>+10.1f} {r['A_p']:>8.4f} {r['A_n']:>8.4f} {r['m_p']:>10.2f} "
              f"{r['m_n']:>10.2f} {r['err_p']:>+10.1f} {r['err_n']:>+10.1f} {conv_str:>6s} {warn_str:>10s}")

# Find best (only consider converged results without warnings)
valid_results = [r for r in results if not np.isnan(r['err_p']) and r['converged'] and not r['warnings']]
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
else:
    print("\n" + "="*80)
    print("WARNING: No fully converged results without issues found!")
    print("All tested g1 values produced either non-convergent or pathological solutions.")
    print("="*80)
