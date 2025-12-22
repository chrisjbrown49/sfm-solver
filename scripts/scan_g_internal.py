"""
Refined scan of g_internal parameter from 1e7 to 1e9.

This script scans g_internal values in the "crossover" region identified
by the coarse logarithmic scan, where interesting physics transitions occur.

Key metrics to watch:
- Beta value (should be ~0.001-0.1 GeV)
- Amplitude ratios (A_mu/A_e should be ~14, A_tau/A_e should be ~59)
- Mass ratios (should match experimental values)
- Convergence (should converge within bounds)
- Delta_x values (should not hit bounds)
- Look for region where all three generations properly separate
"""

import numpy as np
from tabulate import tabulate
import json
import os

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.calculate_beta import calibrate_beta_from_electron
from sfm_solver.core.particle_configurations import ELECTRON, MUON, TAU


def test_g_internal_value(g_internal_value):
    """Test a single g_internal value and return results."""
    
    # Load base constants
    constants_path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'sfm_solver', 'core', 'constants.json'
    )
    with open(constants_path, 'r') as f:
        constants = json.load(f)
    
    # Override g_internal
    constants['g_internal'] = g_internal_value
    
    try:
        # Create solver with modified g_internal
        solver = UnifiedSFMSolver(
            g_internal=g_internal_value,
            n_max=5,
            l_max=2,
            N_sigma=64,
            verbose=False
        )
        
        # Calibrate beta from electron
        beta = calibrate_beta_from_electron(solver, electron_mass_exp=0.000510999)
        
        # Solve all three leptons (max_iter is for shape solver, outer loop has separate limit)
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
        mass_ratio_mu_e = mass_mu / mass_e
        mass_ratio_tau_e = mass_tau / mass_e
        
        # Calculate errors from targets
        mu_error = abs(A_ratio_mu_e - 14.4) / 14.4 * 100
        tau_error = abs(A_ratio_tau_e - 59.0) / 59.0 * 100
        combined_error = (mu_error + tau_error) / 2
        
        # Check confinement (Delta_x should be < 9999 fm)
        confined = (result_e.Delta_x < 9999.0 and 
                   result_mu.Delta_x < 9999.0 and 
                   result_tau.Delta_x < 9999.0)
        
        # Check convergence (outer loop is the key metric)
        converged = (result_e.outer_converged and 
                    result_mu.outer_converged and 
                    result_tau.outer_converged)
        
        return {
            'g_internal': g_internal_value,
            'beta': beta,
            'A_e': result_e.A,
            'A_mu': result_mu.A,
            'A_tau': result_tau.A,
            'A_ratio_mu_e': A_ratio_mu_e,
            'A_ratio_tau_e': A_ratio_tau_e,
            'mu_error_pct': mu_error,
            'tau_error_pct': tau_error,
            'combined_error_pct': combined_error,
            'mass_e': mass_e,
            'mass_mu': mass_mu,
            'mass_tau': mass_tau,
            'mass_ratio_mu_e': mass_ratio_mu_e,
            'mass_ratio_tau_e': mass_ratio_tau_e,
            'Delta_x_e': result_e.Delta_x,
            'Delta_x_mu': result_mu.Delta_x,
            'Delta_x_tau': result_tau.Delta_x,
            'confined': confined,
            'converged': converged,
            'error': None
        }
        
    except Exception as e:
        return {
            'g_internal': g_internal_value,
            'beta': None,
            'A_e': None,
            'A_mu': None,
            'A_tau': None,
            'A_ratio_mu_e': None,
            'A_ratio_tau_e': None,
            'mu_error_pct': None,
            'tau_error_pct': None,
            'combined_error_pct': None,
            'mass_e': None,
            'mass_mu': None,
            'mass_tau': None,
            'mass_ratio_mu_e': None,
            'mass_ratio_tau_e': None,
            'Delta_x_e': None,
            'Delta_x_mu': None,
            'Delta_x_tau': None,
            'confined': False,
            'converged': False,
            'error': str(e)
        }


def main():
    print("="*100)
    print("COARSE LOGARITHMIC SCAN OF g_internal PARAMETER (1e-4 to 1e8)")
    print("="*100)
    print("\nScanning to find physically reasonable regime with outer loop convergence")
    print("\nTarget metrics:")
    print("  - Beta: ~0.001-1.0 GeV")
    print("  - A_mu/A_e: ~14.4 (from sqrt(206.8))")
    print("  - A_tau/A_e: ~59.0 (from sqrt(3477))")
    print("  - Mass ratios: mu/e = 206.8, tau/e = 3477")
    print("  - Convergence: All leptons should converge")
    print("  - Confinement: Delta_x < 9999 fm (unconstrained range)")
    print("\nLooking for:")
    print("  - Regime where outer loop converges")
    print("  - Physical spatial scales (Delta_x ~ 0.1-100 fm)")
    print("  - Generation-dependent amplitudes")
    
    # Logarithmic scan: one value per decade from 1e-4 to 1e8
    g_internal_values = [10**i for i in range(-4, 9)]  # 1e-4, 1e-3, ..., 1e7, 1e8
    
    print(f"\nTesting {len(g_internal_values)} values (one per decade)...")
    print("="*100)
    
    results = []
    for i, g_int in enumerate(g_internal_values):
        print(f"\n[{i+1}/{len(g_internal_values)}] Testing g_internal = {g_int:.2e}...", end=" ")
        result = test_g_internal_value(g_int)
        results.append(result)
        
        if result['error']:
            print(f"[ERROR: {result['error']}]")
        elif not result['confined']:
            print("[NOT CONFINED]")
        else:
            print(f"mu/e={result['A_ratio_mu_e']:.2f}, tau/e={result['A_ratio_tau_e']:.2f}, err={result['combined_error_pct']:.1f}%")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    table_data = []
    for r in results:
        if r['error']:
            table_data.append([
                f"{r['g_internal']:.1e}",
                "ERROR",
                "---",
                "---",
                "---",
                "---"
            ])
        else:
            confined_str = "[OK]" if r['confined'] else "[X]"
            beta_str = f"{r['beta']:.4f}" if r['beta'] else "---"
            A_mu_e_str = f"{r['A_ratio_mu_e']:.2f}" if r['A_ratio_mu_e'] else "---"
            A_tau_e_str = f"{r['A_ratio_tau_e']:.2f}" if r['A_ratio_tau_e'] else "---"
            combined_err_str = f"{r['combined_error_pct']:.1f}%" if r['combined_error_pct'] else "---"
            
            table_data.append([
                f"{r['g_internal']:.1e}",
                beta_str,
                A_mu_e_str,
                A_tau_e_str,
                combined_err_str,
                confined_str
            ])
    
    headers = ['g_internal', 'Beta (GeV)', 'A_mu/A_e', 'A_tau/A_e', 'Combined Error', 'Confined']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Print detailed results for promising candidates
    print("\n" + "="*100)
    print("DETAILED ANALYSIS")
    print("="*100)
    
    # Find candidates with reasonable beta and good confinement
    # In this high g_internal regime, beta can be higher (up to 1.0 GeV)
    candidates = [r for r in results 
                  if r['beta'] is not None 
                  and 0.001 < r['beta'] < 1.0
                  and r['confined']
                  and not r['error']]
    
    if candidates:
        print(f"\nFound {len(candidates)} candidates with reasonable parameters:")
        
        for r in candidates:
            print(f"\n{'='*100}")
            print(f"g_internal = {r['g_internal']:.2e}")
            print(f"{'='*100}")
            print(f"  Beta: {r['beta']:.6f} GeV")
            print(f"\n  Amplitudes:")
            print(f"    A_e   = {r['A_e']:.6f}")
            print(f"    A_mu  = {r['A_mu']:.6f}  (ratio: {r['A_ratio_mu_e']:.2f}, target: 14.4)")
            print(f"    A_tau = {r['A_tau']:.6f}  (ratio: {r['A_ratio_tau_e']:.2f}, target: 59.0)")
            print(f"\n  Masses:")
            print(f"    m_e   = {r['mass_e']:.3f} MeV  (exp: 0.511 MeV)")
            print(f"    m_mu  = {r['mass_mu']:.3f} MeV  (exp: 105.7 MeV)")
            print(f"    m_tau = {r['mass_tau']:.3f} MeV  (exp: 1776.9 MeV)")
            print(f"\n  Mass ratios:")
            print(f"    m_mu/m_e  = {r['mass_ratio_mu_e']:.2f}  (exp: 206.8)")
            print(f"    m_tau/m_e = {r['mass_ratio_tau_e']:.2f}  (exp: 3477)")
            print(f"\n  Spatial extents:")
            print(f"    Delta_x_e   = {r['Delta_x_e']:.2f} fm")
            print(f"    Delta_x_mu  = {r['Delta_x_mu']:.2f} fm")
            print(f"    Delta_x_tau = {r['Delta_x_tau']:.2f} fm")
            print(f"\n  Status: {'Converged' if r['converged'] else 'Not converged'}")
    else:
        print("\nNo candidates found with reasonable parameters in this range.")
        print("Consider:")
        print("  - Adjusting other parameters (g1, alpha)")
        print("  - Increasing energy minimizer iterations")
        print("  - Adjusting bounds for Delta_x and Delta_sigma")
    
    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    
    # Find best by amplitude ratio
    valid_results = [r for r in results if r['A_ratio_mu_e'] is not None and not r['error']]
    if valid_results:
        # Find result with muon ratio closest to target (14.4)
        best_mu = min(valid_results, key=lambda r: abs(r['A_ratio_mu_e'] - 14.4))
        
        # Find result with tau ratio closest to target (59.0)  
        best_tau = min(valid_results, key=lambda r: abs(r['A_ratio_tau_e'] - 59.0))
        
        # Find result with best overall balance (minimize sum of relative errors)
        def score_result(r):
            mu_error = abs(r['A_ratio_mu_e'] - 14.4) / 14.4
            tau_error = abs(r['A_ratio_tau_e'] - 59.0) / 59.0
            return mu_error + tau_error
        
        best_overall = min(valid_results, key=score_result)
        
        print(f"\nBest muon ratio:")
        print(f"  g_internal = {best_mu['g_internal']:.2e}")
        print(f"  A_mu/A_e = {best_mu['A_ratio_mu_e']:.2f} (target: 14.4, error: {abs(best_mu['A_ratio_mu_e']-14.4)/14.4*100:.1f}%)")
        print(f"  A_tau/A_e = {best_mu['A_ratio_tau_e']:.2f} (target: 59.0, error: {abs(best_mu['A_ratio_tau_e']-59.0)/59.0*100:.1f}%)")
        
        print(f"\nBest tau ratio:")
        print(f"  g_internal = {best_tau['g_internal']:.2e}")
        print(f"  A_mu/A_e = {best_tau['A_ratio_mu_e']:.2f} (target: 14.4, error: {abs(best_tau['A_ratio_mu_e']-14.4)/14.4*100:.1f}%)")
        print(f"  A_tau/A_e = {best_tau['A_ratio_tau_e']:.2f} (target: 59.0, error: {abs(best_tau['A_ratio_tau_e']-59.0)/59.0*100:.1f}%)")
        
        print(f"\nBest overall balance:")
        print(f"  g_internal = {best_overall['g_internal']:.2e}")
        print(f"  A_mu/A_e = {best_overall['A_ratio_mu_e']:.2f} (target: 14.4, error: {abs(best_overall['A_ratio_mu_e']-14.4)/14.4*100:.1f}%)")
        print(f"  A_tau/A_e = {best_overall['A_ratio_tau_e']:.2f} (target: 59.0, error: {abs(best_overall['A_ratio_tau_e']-59.0)/59.0*100:.1f}%)")
        print(f"  Combined error score: {score_result(best_overall)*100:.1f}%")
        
        if best_overall['A_ratio_mu_e'] > 5.0 and best_overall['A_ratio_mu_e'] < 25.0:
            print("\n  PROMISING CANDIDATE FOUND!")
            print("  Muon ratio in reasonable range. Next steps:")
            print("    - Fine-tune g1 and alpha around this g_internal value")
            print("    - Increase energy minimizer iterations for better convergence")
        else:
            print("\n  Need further adjustment:")
            if best_overall['A_ratio_mu_e'] < 5.0:
                print("    - Try higher g_internal (stronger effects)")
                print("    - Or increase g1 (stronger generation coupling)")
            else:
                print("    - Try intermediate values between scanned points")
                print("    - Adjust g1 and alpha to balance generation effects")


if __name__ == '__main__':
    main()

