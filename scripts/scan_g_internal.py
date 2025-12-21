"""
Scan g_internal parameter to find valid regime for lepton solver.

Tests multiple values of g_internal between 0.01 and 10,000 and analyzes:
- Convergence behavior
- Mass hierarchy (amplitude ratios)
- Spatial scales (Delta_x)
- Energy balance
"""

import numpy as np
import json
import time
from pathlib import Path
from tabulate import tabulate

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.particle_configurations import CALIBRATION_LEPTONS

# Define scan range (logarithmic spacing)
# Extended range to find where confinement actually occurs
g_internal_values = np.logspace(np.log10(10000), np.log10(1e8), num=15)

print("="*80)
print("SCANNING g_internal PARAMETER SPACE (EXTENDED RANGE)")
print("="*80)
print(f"\nTesting {len(g_internal_values)} values from 10,000 to 100,000,000")
print(f"Values: {[f'{g:.2e}' for g in g_internal_values[:5]]} ... {[f'{g:.2e}' for g in g_internal_values[-5:]]}")

# Path to constants.json
constants_path = Path(__file__).parent.parent / "src" / "sfm_solver" / "core" / "constants.json"

# Load original constants
with open(constants_path, 'r') as f:
    original_constants = json.load(f)

print(f"\nOriginal constants:")
for key, val in original_constants.items():
    if key not in ['hbar', 'c']:
        print(f"  {key}: {val}")

results = []

for i, g_internal in enumerate(g_internal_values):
    print(f"\n{'='*80}")
    print(f"TEST {i+1}/{len(g_internal_values)}: g_internal = {g_internal:.6e}")
    print(f"{'='*80}")
    
    # Note: We pass g_internal explicitly to the solver rather than modifying constants.json
    # because constants are cached at module import time
    
    try:
        # Create fresh solver with explicitly passed g_internal
        # NOTE: Must pass explicitly because constants are cached at module import
        solver = UnifiedSFMSolver(
            g_internal=float(g_internal),  # Explicitly pass the value!
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
                
                print(f"  {lepton.name:10s}: A={result.A:.6f}, Dx={result.Delta_x:.3f} fm, m={result.mass*1000:.3f} MeV")
                
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
            
            m_ratio_mu_e = mu_data['mass_GeV'] / e_data['mass_GeV']
            m_ratio_tau_e = tau_data['mass_GeV'] / e_data['mass_GeV']
            
            # Expected ratios
            expected_A_ratio_mu_e = np.sqrt(206.8)  # ~14.4
            expected_A_ratio_tau_e = np.sqrt(3477)  # ~59.0
            
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
            
            # Quality score (how close to expected ratios)
            A_ratio_error = abs(A_ratio_mu_e - expected_A_ratio_mu_e) / expected_A_ratio_mu_e
            quality_score = 1.0 / (1.0 + A_ratio_error)
            
            results.append({
                'g_internal': g_internal,
                'A_electron': e_data['A'],
                'A_muon': mu_data['A'],
                'A_tau': tau_data['A'],
                'A_ratio_mu_e': A_ratio_mu_e,
                'A_ratio_tau_e': A_ratio_tau_e,
                'expected_A_ratio_mu_e': expected_A_ratio_mu_e,
                'expected_A_ratio_tau_e': expected_A_ratio_tau_e,
                'Delta_x_electron': e_data['Delta_x'],
                'Delta_x_muon': mu_data['Delta_x'],
                'Delta_x_tau': tau_data['Delta_x'],
                'mass_electron_MeV': e_data['mass_GeV'] * 1000,
                'mass_muon_MeV': mu_data['mass_GeV'] * 1000,
                'mass_tau_MeV': tau_data['mass_GeV'] * 1000,
                'm_ratio_mu_e': m_ratio_mu_e,
                'm_ratio_tau_e': m_ratio_tau_e,
                'all_converged': all_converged,
                'hitting_upper_bound': hitting_upper_bound,
                'hitting_lower_bound': hitting_lower_bound,
                'quality_score': quality_score,
                'status': 'OK'
            })
            
            print(f"\n  Amplitude ratios: mu/e={A_ratio_mu_e:.2f} (exp:{expected_A_ratio_mu_e:.2f}), tau/e={A_ratio_tau_e:.2f} (exp:{expected_A_ratio_tau_e:.2f})")
            print(f"  Converged: {all_converged}, Bounds: {'Upper' if hitting_upper_bound else 'Lower' if hitting_lower_bound else 'OK'}")
            print(f"  Quality score: {quality_score:.4f}")
        else:
            results.append({
                'g_internal': g_internal,
                'status': 'FAILED',
                'quality_score': 0.0
            })
            
    except Exception as e:
        print(f"  SOLVER CREATION FAILED: {e}")
        results.append({
            'g_internal': g_internal,
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
            f"{r['g_internal']:.2e}",
            f"{r['A_ratio_mu_e']:.2f}",
            f"{r['A_ratio_tau_e']:.2f}",
            f"{r['Delta_x_electron']:.2f}",
            f"{r['Delta_x_muon']:.2f}",
            f"{r['Delta_x_tau']:.2f}",
            "Yes" if r['all_converged'] else "No",
            "YES" if r['hitting_upper_bound'] else "yes" if r['hitting_lower_bound'] else "",
            f"{r['quality_score']:.4f}"
        ])
    
    headers = ['g_internal', 'A_mu/A_e', 'A_tau/A_e', 'Dx_e (fm)', 'Dx_mu (fm)', 'Dx_tau (fm)', 'Conv', 'Bounds?', 'Quality']
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print(f"\nExpected ratios: A_mu/A_e = {np.sqrt(206.8):.2f}, A_tau/A_e = {np.sqrt(3477):.2f}")
    
    # Find best result
    best_result = max(successful_results, key=lambda x: x['quality_score'])
    print(f"\nBest g_internal: {best_result['g_internal']:.6e}")
    print(f"  A_mu/A_e = {best_result['A_ratio_mu_e']:.2f} (expected: {best_result['expected_A_ratio_mu_e']:.2f})")
    print(f"  A_tau/A_e = {best_result['A_ratio_tau_e']:.2f} (expected: {best_result['expected_A_ratio_tau_e']:.2f})")
    print(f"  Delta_x range: {best_result['Delta_x_electron']:.2f} - {best_result['Delta_x_tau']:.2f} fm")
    print(f"  Quality score: {best_result['quality_score']:.4f}")
    print(f"  All converged: {best_result['all_converged']}")
    
    # Identify regimes
    print("\n" + "="*80)
    print("REGIME ANALYSIS")
    print("="*80)
    
    delocalized = [r for r in successful_results if r['hitting_upper_bound']]
    over_confined = [r for r in successful_results if r['hitting_lower_bound']]
    good_regime = [r for r in successful_results if not r['hitting_upper_bound'] and not r['hitting_lower_bound']]
    
    print(f"\nDelocalized regime (Dx at upper bound): {len(delocalized)} values")
    if delocalized:
        print(f"  g_internal < {max(r['g_internal'] for r in delocalized):.2e}")
    
    print(f"\nOver-confined regime (Dx at lower bound): {len(over_confined)} values")
    if over_confined:
        print(f"  g_internal > {min(r['g_internal'] for r in over_confined):.2e}")
    
    print(f"\nIntermediate regime: {len(good_regime)} values")
    if good_regime:
        g_min = min(r['g_internal'] for r in good_regime)
        g_max = max(r['g_internal'] for r in good_regime)
        print(f"  {g_min:.2e} < g_internal < {g_max:.2e}")
        
        # Check if any produce good hierarchy
        good_hierarchy = [r for r in good_regime if 10 < r['A_ratio_mu_e'] < 20 and 50 < r['A_ratio_tau_e'] < 70]
        if good_hierarchy:
            print(f"\n  Found {len(good_hierarchy)} values with GOOD HIERARCHY!")
            for r in good_hierarchy:
                print(f"    g_internal = {r['g_internal']:.6e}: A_mu/A_e={r['A_ratio_mu_e']:.2f}, A_tau/A_e={r['A_ratio_tau_e']:.2f}")
        else:
            print(f"\n  WARNING: No values produce correct mass hierarchy!")
            print(f"  Best amplitude ratios in intermediate regime:")
            for r in sorted(good_regime, key=lambda x: x['quality_score'], reverse=True)[:3]:
                print(f"    g_internal = {r['g_internal']:.6e}: A_mu/A_e={r['A_ratio_mu_e']:.2f}, A_tau/A_e={r['A_ratio_tau_e']:.2f}")
else:
    print("\nNo successful results in scan!")

failed_count = len([r for r in results if r.get('status') != 'OK'])
if failed_count > 0:
    print(f"\nFailed tests: {failed_count}/{len(results)}")

print("\n" + "="*80)
print("Scan complete!")
print("="*80)

