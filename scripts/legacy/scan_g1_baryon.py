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
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver

# Test g1 values
g1_values = [
    -1000.0,  # Strong attraction
    -100.0,   # Moderate attraction
    -10.0,    # Weak attraction
    -1.0,     # Very weak attraction
    0.0,      # No mean field
    0.1,      # Tiny repulsion
    1.0,      # Weak repulsion
    10.0,     # Moderate repulsion
    100.0,    # Strong repulsion
    1000.0,   # Very strong repulsion
    5000.0,   # Current value
]

print("="*80)
print("G1 PARAMETER SCAN FOR BARYON MASSES")
print("="*80)
print("\nTarget masses:")
print("  Proton:  938.27 MeV")
print("  Neutron: 939.57 MeV")
print("\n" + "="*80)

# First get beta from electron with default g1
print("\nDeriving beta from electron (g1=5000)...")
solver_ref = NonSeparableWavefunctionSolver(alpha=10.5, beta=None, g1=5000.0)
e_ref = solver_ref.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
beta = 0.511 / e_ref.structure_norm**2
print(f"  Electron: A = {e_ref.structure_norm:.6f}")
print(f"  beta = {beta:.6e} GeV")

results = []

for g1 in g1_values:
    print(f"\n{'='*80}")
    print(f"Testing g1 = {g1:+.1f}")
    print(f"{'='*80}")
    
    try:
        # Create solver with this g1
        solver = NonSeparableWavefunctionSolver(
            alpha=10.5,
            beta=beta,
            g1=g1,
            g2=0.035,
            lambda_so=0.2,
        )
        
        # Solve proton
        print("\nProton (uud)...")
        proton = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, 5, -3),
            max_iter_outer=30,
            max_iter_scf=5,
            verbose=False,
        )
        
        # Solve neutron
        print("Neutron (udd)...")
        neutron = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, -3, -3),
            max_iter_outer=30,
            max_iter_scf=5,
            verbose=False,
        )
        
        # Extract results
        A_p = proton.structure_norm
        A_n = neutron.structure_norm
        m_p = beta * A_p**2 * 1000  # MeV
        m_n = beta * A_n**2 * 1000  # MeV
        Dx_p = proton.delta_x_final
        Dx_n = neutron.delta_x_final
        
        err_p = 100 * (m_p - 938.27) / 938.27
        err_n = 100 * (m_n - 939.57) / 939.57
        
        print(f"\nResults:")
        print(f"  Proton:  A={A_p:.4f}, m={m_p:.2f} MeV (err={err_p:+.1f}%), Dx={Dx_p:.4f}")
        print(f"  Neutron: A={A_n:.4f}, m={m_n:.2f} MeV (err={err_n:+.1f}%), Dx={Dx_n:.4f}")
        
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
