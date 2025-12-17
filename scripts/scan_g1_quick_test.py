"""
Quick test of g1 scan with fixes applied - only test 3 values
"""
import numpy as np
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G2, LAMBDA_SO, V0, V1

# Test stability around g1=50
g1_values = [49.0, 50.0, 51.0]

print("="*80)
print("G1 PARAMETER QUICK TEST (FIXED VERSION)")
print("="*80)
print("\nTarget masses:")
print("  Proton:  938.27 MeV")
print("  Neutron: 939.57 MeV")
print(f"\nUsing g2={G2} from constants.json")
print("\nFIXES APPLIED:")
print("  1. Beta derived with SAME g1 being tested")
print("  2. max_iter_scf = 10")
print("  3. Convergence checks added")
print("="*80)

results = []

for idx, g1 in enumerate(g1_values, 1):
    print(f"\n{'='*80}")
    print(f"Testing g1 = {g1:+.1f} ({idx}/{len(g1_values)})")
    print(f"{'='*80}")
    
    try:
        # Create solver with this g1 for BOTH electron and baryons
        print(f"\nCreating solver with g1={g1}...")
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
        print(f"Deriving beta with g1={g1}...")
        e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
        beta = 0.511 / e.structure_norm**2
        print(f"  Electron: A = {e.structure_norm:.6f}, beta = {beta:.6e} MeV")
        
        # Solve proton
        print("\nSolving Proton (uud)...")
        start_time = time.time()
        proton = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, 5, -3),
            max_iter_outer=30,
            max_iter_scf=10,
            verbose=False,
        )
        print(f"  Proton solved in {time.time() - start_time:.1f}s")
        
        # Solve neutron
        print("Solving Neutron (udd)...")
        start_time = time.time()
        neutron = solver.solve_baryon_4D_self_consistent(
            quark_wells=(1, 2, 3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            quark_windings=(5, -3, -3),
            max_iter_outer=30,
            max_iter_scf=10,
            verbose=False,
        )
        print(f"  Neutron solved in {time.time() - start_time:.1f}s")
        
        # Extract results
        A_p = proton.structure_norm
        A_n = neutron.structure_norm
        m_p = beta * A_p**2
        m_n = beta * A_n**2
        Dx_p = proton.delta_x_final
        Dx_n = neutron.delta_x_final
        
        err_p = 100 * (m_p - 938.27) / 938.27
        err_n = 100 * (m_n - 939.57) / 939.57
        
        # Convergence checks
        converged_p = proton.converged
        converged_n = neutron.converged
        
        print(f"\nResults:")
        print(f"  Proton:  A={A_p:.4f}, m={m_p:.2f} MeV (err={err_p:+.1f}%), Dx={Dx_p:.6f}, conv={converged_p}")
        print(f"  Neutron: A={A_n:.4f}, m={m_n:.2f} MeV (err={err_n:+.1f}%), Dx={Dx_n:.6f}, conv={converged_n}")
        print(f"  EM terms: proton={proton.em_energy:.2e}, neutron={neutron.em_energy:.2e}")
        
        # Check for issues
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
            print(f"  WARNING: {', '.join(warnings)}")
        else:
            print(f"  Status: OK")
        
        results.append({
            'g1': g1,
            'A_p': A_p,
            'A_n': A_n,
            'm_p': m_p,
            'm_n': m_n,
            'err_p': err_p,
            'err_n': err_n,
            'converged': converged_p and converged_n,
            'warnings': warnings,
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            'g1': g1,
            'error': str(e),
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for r in results:
    if 'error' in r:
        print(f"\ng1 = {r['g1']:+.1f}: FAILED - {r['error']}")
    else:
        status = "OK" if (r['converged'] and not r['warnings']) else "ISSUES"
        print(f"\ng1 = {r['g1']:+.1f}: {status}")
        print(f"  Proton:  A={r['A_p']:.4f}, m={r['m_p']:.2f} MeV (err={r['err_p']:+.1f}%)")
        print(f"  Neutron: A={r['A_n']:.4f}, m={r['m_n']:.2f} MeV (err={r['err_n']:+.1f}%)")
        if r['warnings']:
            print(f"  Issues: {', '.join(r['warnings'])}")

print("\n" + "="*80)

