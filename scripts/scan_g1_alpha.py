"""
2D scan of g1 and alpha parameters at g_internal = 2e8.

Previous scans identified:
  - g_internal = 2e8 optimal for tau (A_tau/A_e = 62.14, only 5.3% error!)
  - At g1=50: tau perfect but muon too low (A_mu/A_e = 3.43)
  - g1 alone can't balance both generations (competition effect)

This script performs a 3x3 grid scan of both g1 and alpha to find
the optimal combination that balances all three lepton generations.

Strategy:
  - Fix g_internal = 2e8
  - Scan g1 = [50, 100, 150] (3 values)
  - Scan alpha = [20, 30, 50] (3 values)
  - Look for combination that achieves both targets simultaneously
"""

import numpy as np
from tabulate import tabulate
import json
import os

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.calculate_beta import calibrate_beta_from_electron
from sfm_solver.core.particle_configurations import ELECTRON, MUON, TAU


def test_parameter_combination(g1_value, alpha_value, g_internal_fixed=1e8):
    """Test a single (g1, alpha) combination with fixed g_internal and return results."""
    
    try:
        # Create solver with modified g1, alpha and fixed g_internal
        solver = UnifiedSFMSolver(
            g_internal=g_internal_fixed,
            g1=g1_value,
            alpha=alpha_value,
            n_max=5,
            l_max=2,
            N_sigma=64,
            verbose=False
        )
        
        # Calibrate beta from electron
        beta = calibrate_beta_from_electron(solver, electron_mass_exp=0.000510999)
        
        # Solve all three leptons (with outer loop iterations)
        result_e = solver.solve_lepton(winding_k=1, generation_n=1, max_iter=200, max_iter_outer=30)
        result_mu = solver.solve_lepton(winding_k=1, generation_n=2, max_iter=200, max_iter_outer=30)
        result_tau = solver.solve_lepton(winding_k=1, generation_n=3, max_iter=200, max_iter_outer=30)
        
        # Calculate masses
        mass_e = beta * result_e.A**2 * 1000  # MeV
        mass_mu = beta * result_mu.A**2 * 1000  # MeV
        mass_tau = beta * result_tau.A**2 * 1000  # MeV
        
        # Calculate ratios
        A_ratio_mu_e = result_mu.A / result_e.A
        A_ratio_tau_e = result_tau.A / result_e.A
        A_ratio_tau_mu = result_tau.A / result_mu.A
        mass_ratio_mu_e = mass_mu / mass_e
        mass_ratio_tau_e = mass_tau / mass_e
        
        # Calculate errors from targets
        mu_error = abs(A_ratio_mu_e - 14.4) / 14.4 * 100
        tau_error = abs(A_ratio_tau_e - 59.0) / 59.0 * 100
        combined_error = (mu_error + tau_error) / 2
        
        # Check confinement
        confined = (result_e.Delta_x < 9999.0 and 
                   result_mu.Delta_x < 9999.0 and 
                   result_tau.Delta_x < 9999.0)
        
        # Check convergence (outer loop is key)
        converged = (result_e.outer_converged and 
                    result_mu.outer_converged and 
                    result_tau.outer_converged)
        
        return {
            'g1': g1_value,
            'alpha': alpha_value,
            'g_internal': g_internal_fixed,
            'beta': beta,
            'A_e': result_e.A,
            'A_mu': result_mu.A,
            'A_tau': result_tau.A,
            'A_ratio_mu_e': A_ratio_mu_e,
            'A_ratio_tau_e': A_ratio_tau_e,
            'A_ratio_tau_mu': A_ratio_tau_mu,
            'mass_e': mass_e,
            'mass_mu': mass_mu,
            'mass_tau': mass_tau,
            'mass_ratio_mu_e': mass_ratio_mu_e,
            'mass_ratio_tau_e': mass_ratio_tau_e,
            'mu_error_pct': mu_error,
            'tau_error_pct': tau_error,
            'combined_error_pct': combined_error,
            'Delta_x_e': result_e.Delta_x,
            'Delta_x_mu': result_mu.Delta_x,
            'Delta_x_tau': result_tau.Delta_x,
            'confined': confined,
            'converged': converged,
            'error': None
        }
        
    except Exception as e:
        return {
            'g1': g1_value,
            'alpha': alpha_value,
            'g_internal': g_internal_fixed,
            'beta': None,
            'A_e': None,
            'A_mu': None,
            'A_tau': None,
            'A_ratio_mu_e': None,
            'A_ratio_tau_e': None,
            'A_ratio_tau_mu': None,
            'mass_e': None,
            'mass_mu': None,
            'mass_tau': None,
            'mass_ratio_mu_e': None,
            'mass_ratio_tau_e': None,
            'mu_error_pct': None,
            'tau_error_pct': None,
            'combined_error_pct': None,
            'Delta_x_e': None,
            'Delta_x_mu': None,
            'Delta_x_tau': None,
            'confined': False,
            'converged': False,
            'error': str(e)
        }


def main():
    print("="*100)
    print("2D SCAN: g1 AND alpha AT g_internal = 1e8")
    print("="*100)
    print("\nFixed parameter:")
    print("  g_internal = 1.00e+08  (only value with convergence + physical scales)")
    print("\nPrevious results:")
    print("  - Coarse scan showed g_internal = 1e8 is the only regime with:")
    print("    * Outer loop convergence")
    print("    * Physical spatial scales (Delta_x ~ 4.5 fm)")
    print("  - At (g1=50, alpha=30): All generations identical (A_mu/A_e=1.0, A_tau/A_e=1.0)")
    print("  - Need to find g1 and alpha that create generation hierarchy")
    print("\nGoal: Find (g1, alpha) combination that differentiates generations")
    print("\nTarget metrics:")
    print("  - A_mu/A_e: ~14.4 (from sqrt(206.8))")
    print("  - A_tau/A_e: ~59.0 (from sqrt(3477))")
    print("  - Mass ratios: mu/e = 206.8, tau/e = 3477")
    
    # 3x3 grid scan - exploring wide parameter space
    g1_values = [10, 50, 200]  # Low to high
    alpha_values = [10, 30, 100]  # Low to high
    
    g_internal_fixed = 1e8
    
    total_tests = len(g1_values) * len(alpha_values)
    print(f"\nScanning {len(g1_values)} g1 values x {len(alpha_values)} alpha values = {total_tests} combinations...")
    print("="*100)
    
    results = []
    test_num = 0
    for g1 in g1_values:
        for alpha in alpha_values:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] Testing g1={g1:.0f}, alpha={alpha:.0f}...", end=" ")
            result = test_parameter_combination(g1, alpha, g_internal_fixed)
            results.append(result)
            
            if result['error']:
                print(f"[ERROR: {result['error']}]")
            elif not result['confined']:
                print("[NOT CONFINED]")
            else:
                print(f"mu/e={result['A_ratio_mu_e']:.2f}, tau/e={result['A_ratio_tau_e']:.2f}, err={result['combined_error_pct']:.1f}%")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE (3x3 GRID)")
    print("="*100)
    
    table_data = []
    for r in results:
        if r['error']:
            table_data.append([
                f"{r['g1']:.0f}",
                f"{r['alpha']:.0f}",
                "ERROR",
                "---",
                "---",
                "---"
            ])
        else:
            mu_str = f"{r['A_ratio_mu_e']:.2f}" if r['A_ratio_mu_e'] else "---"
            tau_str = f"{r['A_ratio_tau_e']:.2f}" if r['A_ratio_tau_e'] else "---"
            combined_err_str = f"{r['combined_error_pct']:.1f}%" if r['combined_error_pct'] else "---"
            confined_str = "[OK]" if r['confined'] else "[X]"
            
            table_data.append([
                f"{r['g1']:.0f}",
                f"{r['alpha']:.0f}",
                mu_str,
                tau_str,
                combined_err_str,
                confined_str
            ])
    
    headers = ['g1', 'alpha', 'A_mu/A_e', 'A_tau/A_e', 'Combined Error', 'Confined']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\nTarget values:")
    print("  A_mu/A_e = 14.4, A_tau/A_e = 59.0")
    
    # Find best candidates
    print("\n" + "="*100)
    print("DETAILED ANALYSIS")
    print("="*100)
    
    valid_results = [r for r in results if r['combined_error_pct'] is not None and not r['error']]
    
    if valid_results:
        # Sort by combined error
        valid_results.sort(key=lambda r: r['combined_error_pct'])
        
        print(f"\nTop 3 candidates (sorted by combined error):")
        
        for idx, r in enumerate(valid_results[:3]):
            print(f"\n{'-'*100}")
            print(f"#{idx+1}: g1={r['g1']:.0f}, alpha={r['alpha']:.0f}  (Combined error: {r['combined_error_pct']:.1f}%)")
            print(f"{'-'*100}")
            print(f"  Parameters: g_internal=1e8, g1={r['g1']:.0f}, alpha={r['alpha']:.0f}")
            print(f"  Beta: {r['beta']:.6f} GeV")
            print(f"\n  Amplitudes:")
            print(f"    A_e   = {r['A_e']:.6f}")
            print(f"    A_mu  = {r['A_mu']:.6f}  (ratio: {r['A_ratio_mu_e']:.2f}, target: 14.4, error: {r['mu_error_pct']:.1f}%)")
            print(f"    A_tau = {r['A_tau']:.6f}  (ratio: {r['A_ratio_tau_e']:.2f}, target: 59.0, error: {r['tau_error_pct']:.1f}%)")
            print(f"    A_tau/A_mu = {r['A_ratio_tau_mu']:.2f}  (target: 4.1 from 59/14.4)")
            print(f"\n  Masses:")
            print(f"    m_e   = {r['mass_e']:.3f} MeV  (exp: 0.511 MeV)")
            print(f"    m_mu  = {r['mass_mu']:.3f} MeV  (exp: 105.7 MeV)")
            print(f"    m_tau = {r['mass_tau']:.3f} MeV  (exp: 1776.9 MeV)")
            print(f"\n  Mass ratios:")
            print(f"    m_mu/m_e  = {r['mass_ratio_mu_e']:.2f}  (exp: 206.8)")
            print(f"    m_tau/m_e = {r['mass_ratio_tau_e']:.2f}  (exp: 3477)")
            print(f"\n  Spatial extents:")
            print(f"    Delta_x_e   = {r['Delta_x_e']:.4f} fm")
            print(f"    Delta_x_mu  = {r['Delta_x_mu']:.4f} fm")
            print(f"    Delta_x_tau = {r['Delta_x_tau']:.4f} fm")
            print(f"\n  Status:")
            print(f"    Confined: {'Yes' if r['confined'] else 'No'}")
            print(f"    Converged: {'Yes' if r['converged'] else 'No'}")
        
        # Find best by individual metrics
        best = valid_results[0]
        
        print("\n" + "="*100)
        print("RECOMMENDATIONS")
        print("="*100)
        
        if best['combined_error_pct'] < 30.0:
            print(f"\n  EXCELLENT CANDIDATE FOUND!")
            print(f"  (g1={best['g1']:.0f}, alpha={best['alpha']:.0f}) gives combined error of only {best['combined_error_pct']:.1f}%")
            print(f"\n  Optimal parameters:")
            print(f"    g_internal = 1e8")
            print(f"    g1 = {best['g1']:.0f}")
            print(f"    alpha = {best['alpha']:.0f}")
            print(f"\n  Next steps:")
            print(f"    - Update constants.json with these values")
            print(f"    - Run full lepton test to verify")
            print(f"    - Use parameter optimizer for final fine-tuning")
        elif best['combined_error_pct'] < 50.0:
            print(f"\n  PROMISING CANDIDATE FOUND!")
            print(f"  (g1={best['g1']:.0f}, alpha={best['alpha']:.0f}) gives combined error of {best['combined_error_pct']:.1f}%")
            print(f"\n  Next steps:")
            print(f"    - Refine search around this region")
            print(f"    - Try intermediate values: g1={best['g1']-25:.0f}-{best['g1']+25:.0f}, alpha={best['alpha']-10:.0f}-{best['alpha']+10:.0f}")
            print(f"    - Use parameter optimizer for final tuning")
        else:
            print(f"\n  Best result has {best['combined_error_pct']:.1f}% combined error")
            print(f"  (g1={best['g1']:.0f}, alpha={best['alpha']:.0f})")
            print(f"\n  Analysis:")
            print(f"    Muon error: {best['mu_error_pct']:.1f}% (A_mu/A_e = {best['A_ratio_mu_e']:.2f}, target: 14.4)")
            print(f"    Tau error:  {best['tau_error_pct']:.1f}% (A_tau/A_e = {best['A_ratio_tau_e']:.2f}, target: 59.0)")
            print(f"\n  Further tuning needed:")
            
            if best['mu_error_pct'] > best['tau_error_pct']:
                print(f"    - Muon error is larger")
                if best['A_ratio_mu_e'] < 14.4:
                    print(f"    - Try higher g1 or different alpha")
                else:
                    print(f"    - Try lower g1 or different alpha")
            else:
                print(f"    - Tau error is larger")
                print(f"    - May need to adjust g_internal or g2 parameters")
    
    else:
        print("\nNo valid results - all runs encountered errors")


if __name__ == '__main__':
    main()

