"""
Focused scan of G_5D values around 1 GeV^-2 to find optimal operating regime.
"""

import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver

# Test G_5D values around 0.3 (where we saw promising results)
g5d_values = [0.1, 0.3, 0.5, 0.7, 1.0]

print("="*80)
print("FOCUSED G_5D SCAN AROUND 1 GeV^-2")
print("="*80)
print()

results = []

for g5d in g5d_values:
    print(f"\nTesting G_5D = {g5d:.1e} GeV^-2...")
    print("-" * 80)
    
    try:
        solver = UnifiedSFMSolver(G_5D=g5d, verbose=False)
        
        # Test electron
        result_e = solver.solve_lepton(
            winding_k=1,
            generation_n=1,
            max_iter=200,
            tol=1e-6,
            max_iter_outer=20,
            tol_outer=1e-3
        )
        
        # Test muon
        result_mu = solver.solve_lepton(
            winding_k=1,
            generation_n=2,
            max_iter=200,
            tol=1e-6,
            max_iter_outer=20,
            tol_outer=1e-3
        )
        
        # Test tau
        result_tau = solver.solve_lepton(
            winding_k=1,
            generation_n=3,
            max_iter=200,
            tol=1e-6,
            max_iter_outer=20,
            tol_outer=1e-3
        )
        
        # Calculate masses using legacy calibration approach
        m_e = result_e.A ** 2
        m_mu = result_mu.A ** 2
        m_tau = result_tau.A ** 2
        
        # Check if converged
        all_converged = (result_e.outer_converged and 
                        result_mu.outer_converged and 
                        result_tau.outer_converged)
        
        # Check bounds
        e_at_bound = (abs(result_e.A - 0.001) < 1e-6 or 
                     abs(result_e.Delta_x - 0.01) < 1e-6 or
                     abs(result_e.Delta_x - 100) < 1e-3)
        mu_at_bound = (abs(result_mu.A - 0.001) < 1e-6 or 
                      abs(result_mu.Delta_x - 0.01) < 1e-6 or
                      abs(result_mu.Delta_x - 100) < 1e-3)
        tau_at_bound = (abs(result_tau.A - 0.001) < 1e-6 or 
                       abs(result_tau.Delta_x - 0.01) < 1e-6 or
                       abs(result_tau.Delta_x - 100) < 1e-3)
        
        any_at_bound = e_at_bound or mu_at_bound or tau_at_bound
        
        print(f"  Electron:  A={result_e.A:.4f}, Dx={result_e.Delta_x:.3f} fm, E={result_e.E_total:.2e} GeV")
        print(f"  Muon:      A={result_mu.A:.4f}, Dx={result_mu.Delta_x:.3f} fm, E={result_mu.E_total:.2e} GeV")
        print(f"  Tau:       A={result_tau.A:.4f}, Dx={result_tau.Delta_x:.3f} fm, E={result_tau.E_total:.2e} GeV")
        print(f"  Mass ratio (mu/e):  {m_mu/m_e:.2f} (exp: 206.77)")
        print(f"  Mass ratio (tau/e): {m_tau/m_e:.2f} (exp: 3477)")
        print(f"  Converged: {all_converged}, At bounds: {any_at_bound}")
        
        results.append({
            'G_5D': g5d,
            'A_e': result_e.A,
            'A_mu': result_mu.A,
            'A_tau': result_tau.A,
            'E_e': result_e.E_total,
            'E_mu': result_mu.E_total,
            'E_tau': result_tau.E_total,
            'Dx_e': result_e.Delta_x,
            'Dx_mu': result_mu.Delta_x,
            'Dx_tau': result_tau.Delta_x,
            'ratio_mu_e': m_mu/m_e,
            'ratio_tau_e': m_tau/m_e,
            'converged': all_converged,
            'at_bounds': any_at_bound
        })
        
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)}")
        results.append({
            'G_5D': g5d,
            'converged': False,
            'error': str(e)
        })

print("\n" + "="*80)
print("SCAN SUMMARY")
print("="*80)
print()
print(f"{'G_5D':>12} {'A_e':>8} {'A_mu':>8} {'A_tau':>9} {'mu/e':>8} {'tau/e':>8} {'Conv':>6} {'Bounds':>7}")
print("-"*80)

for r in results:
    if 'error' in r:
        print(f"{r['G_5D']:>12.1e} FAILED: {r['error'][:50]}")
    else:
        print(f"{r['G_5D']:>12.1e} {r['A_e']:>8.4f} {r['A_mu']:>8.4f} {r['A_tau']:>9.4f} "
              f"{r['ratio_mu_e']:>8.2f} {r['ratio_tau_e']:>8.1f} "
              f"{str(r['converged']):>6} {str(r['at_bounds']):>7}")

print()
print("Best candidate (converged, not at bounds, reasonable ratios):")
for r in results:
    if 'error' not in r and r['converged'] and not r['at_bounds']:
        if 100 < r['ratio_mu_e'] < 400 and 1000 < r['ratio_tau_e'] < 5000:
            print(f"  G_5D = {r['G_5D']:.2e} GeV^-2")
            print(f"    mu/e = {r['ratio_mu_e']:.2f}, tau/e = {r['ratio_tau_e']:.1f}")
            break
else:
    print("  No ideal candidate found. Closest:")
    # Find best by convergence + not at bounds
    valid = [r for r in results if 'error' not in r and not r['at_bounds']]
    if valid:
        best = max(valid, key=lambda x: x['converged'])
        print(f"  G_5D = {best['G_5D']:.2e} GeV^-2")
        print(f"    mu/e = {best['ratio_mu_e']:.2f}, tau/e = {best['ratio_tau_e']:.1f}")
        print(f"    Converged: {best['converged']}, At bounds: {best['at_bounds']}")

