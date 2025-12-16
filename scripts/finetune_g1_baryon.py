"""
Fine-tune g1 parameter around optimal range (50-2000) for baryon masses.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver

# Fine-tune g1 values around 100-1000
g1_values = [
    50.0,
    75.0,
    100.0,
    125.0,
    150.0,
    175.0,
    200.0,
    250.0,
    300.0,
    400.0,
    500.0,
    750.0,
    1000.0,
    1250.0,
    1500.0,
    2000.0,
]

print("="*80)
print("G1 FINE-TUNING FOR BARYON MASSES")
print("="*80)
print("\nTarget masses:")
print("  Proton:  938.27 MeV")
print("  Neutron: 939.57 MeV")
print("\n" + "="*80)

# Get beta from electron
print("\nDeriving beta from electron...")
solver_ref = NonSeparableWavefunctionSolver(alpha=10.5, beta=None, g1=100.0)
e_ref = solver_ref.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
beta = 0.511 / e_ref.structure_norm**2
print(f"  Electron: A = {e_ref.structure_norm:.6f}")
print(f"  beta = {beta:.6e} GeV")

results = []

for g1 in g1_values:
    print(f"\n{'='*80}")
    print(f"Testing g1 = {g1:.1f}")
    print(f"{'='*80}")
    
    try:
        solver = NonSeparableWavefunctionSolver(
            alpha=10.5,
            beta=beta,
            g1=g1,
            g2=0.035,
            lambda_so=0.2,
        )
        
        # Solve proton
        proton = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, 5, -3),
            max_iter_outer=30,
            max_iter_scf=5,
            verbose=False,
        )
        
        # Solve neutron
        neutron = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, -3, -3),
            max_iter_outer=30,
            max_iter_scf=5,
            verbose=False,
        )
        
        A_p = proton.structure_norm
        A_n = neutron.structure_norm
        m_p = beta * A_p**2 * 1000  # MeV
        m_n = beta * A_n**2 * 1000  # MeV
        Dx_p = proton.delta_x_final
        Dx_n = neutron.delta_x_final
        
        err_p = 100 * (m_p - 938.27) / 938.27
        err_n = 100 * (m_n - 939.57) / 939.57
        err_avg = (abs(err_p) + abs(err_n)) / 2
        
        print(f"  Proton:  A={A_p:.4f}, m={m_p:.2f} MeV (err={err_p:+.2f}%), Dx={Dx_p:.4f}")
        print(f"  Neutron: A={A_n:.4f}, m={m_n:.2f} MeV (err={err_n:+.2f}%), Dx={Dx_n:.4f}")
        print(f"  Average error: {err_avg:.2f}%")
        
        results.append({
            'g1': g1,
            'A_p': A_p,
            'A_n': A_n,
            'm_p': m_p,
            'm_n': m_n,
            'err_p': err_p,
            'err_n': err_n,
            'err_avg': err_avg,
            'Dx_p': Dx_p,
            'Dx_n': Dx_n,
            'delta_mn': m_n - m_p,
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'g1':>8s} {'A_p':>8s} {'A_n':>8s} {'m_p(MeV)':>10s} {'m_n(MeV)':>10s} "
      f"{'err_p(%)':>10s} {'err_n(%)':>10s} {'avg_err':>10s} {'Δm(MeV)':>10s}")
print("-"*98)

for r in results:
    print(f"{r['g1']:>8.1f} {r['A_p']:>8.4f} {r['A_n']:>8.4f} {r['m_p']:>10.2f} "
          f"{r['m_n']:>10.2f} {r['err_p']:>+10.2f} {r['err_n']:>+10.2f} "
          f"{r['err_avg']:>10.2f} {r['delta_mn']:>+10.2f}")

# Find optimal values
if results:
    best_avg = min(results, key=lambda x: x['err_avg'])
    best_p = min(results, key=lambda x: abs(x['err_p']))
    best_n = min(results, key=lambda x: abs(x['err_n']))
    
    # Best by mass splitting (target is 1.3 MeV)
    best_split = min(results, key=lambda x: abs(x['delta_mn'] - 1.3))
    
    print("\n" + "="*80)
    print("OPTIMAL VALUES")
    print("="*80)
    
    print(f"\nBest overall (minimum average error):")
    print(f"  g1 = {best_avg['g1']:.1f}")
    print(f"  Proton:  A={best_avg['A_p']:.4f}, m={best_avg['m_p']:.2f} MeV (err={best_avg['err_p']:+.2f}%)")
    print(f"  Neutron: A={best_avg['A_n']:.4f}, m={best_avg['m_n']:.2f} MeV (err={best_avg['err_n']:+.2f}%)")
    print(f"  Average error: {best_avg['err_avg']:.2f}%")
    print(f"  Mass splitting: {best_avg['delta_mn']:+.2f} MeV (target: +1.30 MeV)")
    
    print(f"\nBest for proton:")
    print(f"  g1 = {best_p['g1']:.1f}")
    print(f"  Proton:  m={best_p['m_p']:.2f} MeV (err={best_p['err_p']:+.2f}%)")
    print(f"  Neutron: m={best_p['m_n']:.2f} MeV (err={best_p['err_n']:+.2f}%)")
    
    print(f"\nBest for neutron:")
    print(f"  g1 = {best_n['g1']:.1f}")
    print(f"  Proton:  m={best_n['m_p']:.2f} MeV (err={best_n['err_p']:+.2f}%)")
    print(f"  Neutron: m={best_n['m_n']:.2f} MeV (err={best_n['err_n']:+.2f}%)")
    
    print(f"\nBest mass splitting:")
    print(f"  g1 = {best_split['g1']:.1f}")
    print(f"  Proton:  m={best_split['m_p']:.2f} MeV")
    print(f"  Neutron: m={best_split['m_n']:.2f} MeV")
    print(f"  Mass splitting: {best_split['delta_mn']:+.2f} MeV (target: +1.30 MeV)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"\nOptimal g1 for constants.json: {best_avg['g1']:.1f}")
    print(f"This gives the best overall match to both proton and neutron masses.")

