"""
Scan g1 and alpha parameters to understand their effect on mass hierarchy.

Uses a fixed g_internal value from the intermediate regime where proper
confinement occurs, then varies:
- g1: Nonlinear subspace coupling
- alpha: Spatial-subspace coupling

This should reveal how these parameters affect the lepton mass ratios.
"""

import numpy as np
import json
import time
from pathlib import Path
from tabulate import tabulate

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.particle_configurations import CALIBRATION_LEPTONS

# Fixed g_internal from intermediate regime (showed good confinement)
G_INTERNAL_FIXED = 2.68e5  # From previous scan: Dx_e=130 fm, decent confinement

# Scan ranges (3 values each = 9 combinations)
g1_values = [10.0, 50.0, 200.0]  # Nonlinear subspace coupling
alpha_values = [5.0, 10.0, 30.0]  # Spatial-subspace coupling

print("="*80)
print("SCANNING g1 AND alpha PARAMETER SPACE")
print("="*80)
print(f"\nFixed: g_internal = {G_INTERNAL_FIXED:.2e}")
print(f"\nScanning:")
print(f"  g1 (nonlinear subspace): {g1_values}")
print(f"  alpha (spatial-subspace coupling): {alpha_values}")
print(f"\nTotal combinations: {len(g1_values) * len(alpha_values)}")

# Expected ratios
expected_A_ratio_mu_e = np.sqrt(206.8)  # ~14.4
expected_A_ratio_tau_e = np.sqrt(3477)  # ~59.0

print(f"\nTarget amplitude ratios:")
print(f"  A_mu/A_e = {expected_A_ratio_mu_e:.2f}")
print(f"  A_tau/A_e = {expected_A_ratio_tau_e:.2f}")

results = []
test_num = 0

for g1 in g1_values:
    for alpha in alpha_values:
        test_num += 1
        print(f"\n{'='*80}")
        print(f"TEST {test_num}/{len(g1_values)*len(alpha_values)}: g1={g1:.1f}, alpha={alpha:.1f}")
        print(f"{'='*80}")
        
        try:
            # Create solver with explicit parameters
            solver = UnifiedSFMSolver(
                g_internal=G_INTERNAL_FIXED,
                g1=g1,
                alpha=alpha,
                auto_calibrate_beta=True,
                n_max=5,
                l_max=2,
                N_sigma=64,
                verbose=False
            )
            
            # Test all three leptons
            lepton_results = {}
            
            for lepton in CALIBRATION_LEPTONS:
                try:
                    start_time = time.time()
                    result = solver.solve_lepton(
                        winding_k=lepton.winding,
                        generation_n=lepton.generation,
                        max_iter=200,
                        tol=1e-6
                    )
                    elapsed = time.time() - start_time
                    
                    lepton_results[lepton.name] = {
                        'success': True,
                        'A': result.A,
                        'Delta_x': result.Delta_x,
                        'Delta_sigma': result.Delta_sigma,
                        'mass_GeV': result.mass,
                        'energy_converged': result.energy_converged,
                        'iterations': result.energy_iterations,
                        'time': elapsed,
                        'E_total': result.E_total
                    }
                    
                    print(f"  {lepton.name:10s}: A={result.A:.4f}, Dx={result.Delta_x:.2f} fm, m={result.mass*1000:.2f} MeV")
                    
                except Exception as e:
                    lepton_results[lepton.name] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  {lepton.name:10s}: FAILED - {e}")
            
            # Analyze results
            if all(r.get('success', False) for r in lepton_results.values()):
                e_data = lepton_results['electron']
                mu_data = lepton_results['muon']
                tau_data = lepton_results['tau']
                
                # Calculate ratios
                A_ratio_mu_e = mu_data['A'] / e_data['A']
                A_ratio_tau_e = tau_data['A'] / e_data['A']
                A_ratio_tau_mu = tau_data['A'] / mu_data['A']
                
                m_ratio_mu_e = mu_data['mass_GeV'] / e_data['mass_GeV']
                m_ratio_tau_e = tau_data['mass_GeV'] / e_data['mass_GeV']
                
                # Check if hitting bounds
                hitting_upper_bound = (
                    e_data['Delta_x'] >= 999 or 
                    mu_data['Delta_x'] >= 999 or 
                    tau_data['Delta_x'] >= 999
                )
                hitting_lower_bound = (
                    e_data['Delta_x'] <= 0.002 or 
                    mu_data['Delta_x'] <= 0.002 or 
                    tau_data['Delta_x'] <= 0.002
                )
                
                # Check convergence
                all_converged = (
                    e_data['energy_converged'] and 
                    mu_data['energy_converged'] and 
                    tau_data['energy_converged']
                )
                
                # Quality scores for both ratios
                error_mu_e = abs(A_ratio_mu_e - expected_A_ratio_mu_e) / expected_A_ratio_mu_e
                error_tau_e = abs(A_ratio_tau_e - expected_A_ratio_tau_e) / expected_A_ratio_tau_e
                
                # Combined quality (closer to 1 is better)
                quality_score = 1.0 / (1.0 + error_mu_e + error_tau_e)
                
                # Check hierarchy direction (should be e < mu < tau)
                correct_order = (e_data['A'] < mu_data['A'] < tau_data['A'])
                
                results.append({
                    'g1': g1,
                    'alpha': alpha,
                    'A_electron': e_data['A'],
                    'A_muon': mu_data['A'],
                    'A_tau': tau_data['A'],
                    'A_ratio_mu_e': A_ratio_mu_e,
                    'A_ratio_tau_e': A_ratio_tau_e,
                    'A_ratio_tau_mu': A_ratio_tau_mu,
                    'Delta_x_electron': e_data['Delta_x'],
                    'Delta_x_muon': mu_data['Delta_x'],
                    'Delta_x_tau': tau_data['Delta_x'],
                    'Delta_sigma_electron': e_data['Delta_sigma'],
                    'Delta_sigma_muon': mu_data['Delta_sigma'],
                    'Delta_sigma_tau': tau_data['Delta_sigma'],
                    'mass_electron_MeV': e_data['mass_GeV'] * 1000,
                    'mass_muon_MeV': mu_data['mass_GeV'] * 1000,
                    'mass_tau_MeV': tau_data['mass_GeV'] * 1000,
                    'm_ratio_mu_e': m_ratio_mu_e,
                    'm_ratio_tau_e': m_ratio_tau_e,
                    'all_converged': all_converged,
                    'correct_order': correct_order,
                    'hitting_upper_bound': hitting_upper_bound,
                    'hitting_lower_bound': hitting_lower_bound,
                    'quality_score': quality_score,
                    'error_mu_e': error_mu_e,
                    'error_tau_e': error_tau_e,
                    'status': 'OK'
                })
                
                print(f"\n  Amplitude ratios: mu/e={A_ratio_mu_e:.2f} (exp:{expected_A_ratio_mu_e:.2f}), "
                      f"tau/e={A_ratio_tau_e:.2f} (exp:{expected_A_ratio_tau_e:.2f})")
                print(f"  Hierarchy: {'e<mu<tau OK' if correct_order else 'WRONG'}")
                print(f"  Quality score: {quality_score:.4f}")
                
            else:
                results.append({
                    'g1': g1,
                    'alpha': alpha,
                    'status': 'FAILED',
                    'quality_score': 0.0
                })
                
        except Exception as e:
            print(f"  SOLVER CREATION FAILED: {e}")
            results.append({
                'g1': g1,
                'alpha': alpha,
                'status': f'ERROR: {e}',
                'quality_score': 0.0
            })

print("\n" + "="*80)
print("SCAN RESULTS SUMMARY")
print("="*80)

# Create summary table
successful_results = [r for r in results if r.get('status') == 'OK']

if successful_results:
    table_data = []
    for r in successful_results:
        table_data.append([
            f"{r['g1']:.0f}",
            f"{r['alpha']:.0f}",
            f"{r['A_ratio_mu_e']:.2f}",
            f"{r['A_ratio_tau_e']:.2f}",
            "OK" if r['correct_order'] else "NO",
            f"{r['Delta_x_electron']:.1f}",
            f"{r['Delta_x_muon']:.1f}",
            f"{r['Delta_x_tau']:.1f}",
            f"{r['quality_score']:.4f}"
        ])
    
    headers = ['g1', 'alpha', 'A_mu/A_e', 'A_tau/A_e', 'Order', 'Dx_e', 'Dx_mu', 'Dx_tau', 'Quality']
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print(f"\nTarget ratios: A_mu/A_e = {expected_A_ratio_mu_e:.2f}, A_tau/A_e = {expected_A_ratio_tau_e:.2f}")
    
    # Find best result
    best_result = max(successful_results, key=lambda x: x['quality_score'])
    print(f"\n{'='*80}")
    print("BEST RESULT")
    print(f"{'='*80}")
    print(f"g1 = {best_result['g1']:.1f}, alpha = {best_result['alpha']:.1f}")
    print(f"  A_mu/A_e = {best_result['A_ratio_mu_e']:.2f} (expected: {expected_A_ratio_mu_e:.2f}, error: {best_result['error_mu_e']:.1%})")
    print(f"  A_tau/A_e = {best_result['A_ratio_tau_e']:.2f} (expected: {expected_A_ratio_tau_e:.2f}, error: {best_result['error_tau_e']:.1%})")
    print(f"  Hierarchy order: {'e<mu<tau OK' if best_result['correct_order'] else 'WRONG'}")
    print(f"  Quality score: {best_result['quality_score']:.4f}")
    
    # Trend analysis
    print(f"\n{'='*80}")
    print("TREND ANALYSIS")
    print(f"{'='*80}")
    
    # Group by g1
    print("\n--- Effect of g1 (at each alpha) ---")
    for alpha_val in alpha_values:
        alpha_results = [r for r in successful_results if r['alpha'] == alpha_val]
        if alpha_results:
            print(f"\nalpha = {alpha_val}:")
            for r in sorted(alpha_results, key=lambda x: x['g1']):
                print(f"  g1={r['g1']:6.0f}: A_mu/A_e={r['A_ratio_mu_e']:5.2f}, A_tau/A_e={r['A_ratio_tau_e']:5.2f}, "
                      f"Quality={r['quality_score']:.4f}")
    
    # Group by alpha
    print("\n--- Effect of alpha (at each g1) ---")
    for g1_val in g1_values:
        g1_results = [r for r in successful_results if r['g1'] == g1_val]
        if g1_results:
            print(f"\ng1 = {g1_val}:")
            for r in sorted(g1_results, key=lambda x: x['alpha']):
                print(f"  alpha={r['alpha']:6.0f}: A_mu/A_e={r['A_ratio_mu_e']:5.2f}, A_tau/A_e={r['A_ratio_tau_e']:5.2f}, "
                      f"Quality={r['quality_score']:.4f}")
    
    # Check for monotonic trends
    print("\n--- Monotonicity Check ---")
    
    # Does quality improve with g1?
    for alpha_val in alpha_values:
        alpha_results = [r for r in successful_results if r['alpha'] == alpha_val]
        if len(alpha_results) >= 2:
            alpha_results_sorted = sorted(alpha_results, key=lambda x: x['g1'])
            qualities = [r['quality_score'] for r in alpha_results_sorted]
            g1s = [r['g1'] for r in alpha_results_sorted]
            trend = "increasing" if all(qualities[i] <= qualities[i+1] for i in range(len(qualities)-1)) else \
                    "decreasing" if all(qualities[i] >= qualities[i+1] for i in range(len(qualities)-1)) else "non-monotonic"
            print(f"  alpha={alpha_val}: Quality vs g1 is {trend}")
    
    # Does quality improve with alpha?
    for g1_val in g1_values:
        g1_results = [r for r in successful_results if r['g1'] == g1_val]
        if len(g1_results) >= 2:
            g1_results_sorted = sorted(g1_results, key=lambda x: x['alpha'])
            qualities = [r['quality_score'] for r in g1_results_sorted]
            alphas = [r['alpha'] for r in g1_results_sorted]
            trend = "increasing" if all(qualities[i] <= qualities[i+1] for i in range(len(qualities)-1)) else \
                    "decreasing" if all(qualities[i] >= qualities[i+1] for i in range(len(qualities)-1)) else "non-monotonic"
            print(f"  g1={g1_val}: Quality vs alpha is {trend}")

else:
    print("\nNo successful results in scan!")

failed_count = len([r for r in results if r.get('status') != 'OK'])
if failed_count > 0:
    print(f"\nFailed tests: {failed_count}/{len(results)}")

print("\n" + "="*80)
print("Scan complete!")
print("="*80)

