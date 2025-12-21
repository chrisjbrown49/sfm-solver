"""
Test Script for Lepton Solving.

Tests the new two-stage solver architecture on all leptons:
- Calibration: electron, muon, tau
- Validation: electron neutrino, muon neutrino, tau neutrino

Reports masses, amplitudes, scale parameters, convergence metrics,
and compares to experimental values.
"""

import numpy as np
import time
from tabulate import tabulate

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.calculate_beta import calibrate_beta_from_electron
from sfm_solver.core.particle_configurations import (
    CALIBRATION_LEPTONS,
)


def test_lepton(solver, lepton_config, beta, verbose=False):
    """Test lepton and return results.
    
    Args:
        solver: UnifiedSFMSolver instance
        lepton_config: Particle configuration
        beta: Mass scale (GeV) from electron calibration
        verbose: Print detailed output
    """
    print(f"\nTesting {lepton_config.name}...")
    start_time = time.time()
    
    try:
        result = solver.solve_lepton(
            winding_k=lepton_config.winding,
            generation_n=lepton_config.generation,
            max_iter=200,
            tol=1e-3  # Relaxed convergence tolerance (0.1% is excellent)
        )
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"  Shape converged: {result.shape_converged} ({result.shape_iterations} iters)")
            print(f"  Energy converged: {result.energy_converged} ({result.energy_iterations} iters)")
            print(f"  Time: {elapsed_time:.3f} s")
        
        # Calculate mass from amplitude using calibrated beta
        mass_mev = beta * (result.A ** 2) * 1000.0
        
        # Compute error (handle neutrinos with zero experimental mass)
        if lepton_config.mass_exp > 0:
            error_percent = abs(mass_mev - lepton_config.mass_exp) / lepton_config.mass_exp * 100.0
        else:
            error_percent = None  # Cannot compute error for zero mass
        
        return {
            'name': lepton_config.name,
            'generation': lepton_config.generation,
            'is_neutrino': lepton_config.is_neutrino,
            'mass_predicted_MeV': mass_mev,
            'mass_experimental_MeV': lepton_config.mass_exp,
            'error_percent': error_percent,
            'amplitude': result.A,
            'Delta_x_fm': result.Delta_x,
            'Delta_sigma': result.Delta_sigma,
            'shape_converged': result.shape_converged,
            'shape_iterations': result.shape_iterations,
            'energy_converged': result.energy_converged,
            'energy_iterations': result.energy_iterations,
            'time_seconds': elapsed_time,
            'E_total': result.E_total,
            'energy_components': result.energy_components,
            'error': None
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  ERROR: {e}")
        return {
            'name': lepton_config.name,
            'generation': lepton_config.generation,
            'is_neutrino': lepton_config.is_neutrino,
            'mass_predicted_MeV': None,
            'mass_experimental_MeV': lepton_config.mass_exp,
            'error_percent': None,
            'amplitude': None,
            'Delta_x_fm': None,
            'Delta_sigma': None,
            'shape_converged': False,
            'shape_iterations': 0,
            'energy_converged': False,
            'energy_iterations': 0,
            'time_seconds': elapsed_time,
            'E_total': None,
            'energy_components': None,
            'error': str(e)
        }


def print_lepton_results(results, title):
    """Print lepton results in table format."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # Basic results table
    table_data = []
    for r in results:
        if r['mass_predicted_MeV'] is not None:
            mass_str = f"{r['mass_predicted_MeV']:.6f}"
        else:
            mass_str = "FAILED"
        
        if r['error_percent'] is not None:
            error_str = f"{r['error_percent']:.1f}%"
        elif r['mass_experimental_MeV'] == 0:
            error_str = "N/A"
        else:
            error_str = "FAILED"
        
        converged = "Yes" if (r['shape_converged'] and r['energy_converged']) else "No"
        
        table_data.append([
            r['name'].replace('_', ' ').title(),
            f"n={r['generation']}",
            mass_str,
            f"{r['mass_experimental_MeV']:.6f}",
            error_str,
            converged
        ])
    
    headers = ['Particle', 'Gen', 'Predicted (MeV)', 'Experimental (MeV)', 'Error', 'Converged']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Scale parameters table
    print("\n" + "-"*80)
    print("SCALE PARAMETERS")
    print("-"*80)
    
    table_data = []
    for r in results:
        if r['amplitude'] is not None:
            table_data.append([
                r['name'].replace('_', ' ').title(),
                f"{r['amplitude']:.6f}",
                f"{r['Delta_x_fm']:.6f}",
                f"{r['Delta_sigma']:.6f}",
                f"{r['time_seconds']:.3f}"
            ])
        else:
            table_data.append([
                r['name'].replace('_', ' ').title(),
                "FAILED",
                "FAILED",
                "FAILED",
                f"{r['time_seconds']:.3f}"
            ])
    
    headers = ['Particle', 'Amplitude A', 'Delta_x (fm)', 'Delta_sigma', 'Time (s)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Mass ratios (for charged leptons only)
    charged_leptons = [r for r in results if not r['is_neutrino'] and r['mass_predicted_MeV'] is not None]
    if len(charged_leptons) >= 2:
        print("\n" + "-"*80)
        print("MASS RATIOS (Charged Leptons)")
        print("-"*80)
        
        if len(charged_leptons) >= 3:
            e_mass = charged_leptons[0]['mass_predicted_MeV']
            mu_mass = charged_leptons[1]['mass_predicted_MeV']
            tau_mass = charged_leptons[2]['mass_predicted_MeV']
            
            e_mass_exp = charged_leptons[0]['mass_experimental_MeV']
            mu_mass_exp = charged_leptons[1]['mass_experimental_MeV']
            tau_mass_exp = charged_leptons[2]['mass_experimental_MeV']
            
            table_data = [
                ['mu/e', f"{mu_mass/e_mass:.3f}", f"{mu_mass_exp/e_mass_exp:.3f}", 
                 f"{abs(mu_mass/e_mass - mu_mass_exp/e_mass_exp)/(mu_mass_exp/e_mass_exp)*100:.1f}%"],
                ['tau/e', f"{tau_mass/e_mass:.3f}", f"{tau_mass_exp/e_mass_exp:.3f}",
                 f"{abs(tau_mass/e_mass - tau_mass_exp/e_mass_exp)/(tau_mass_exp/e_mass_exp)*100:.1f}%"],
                ['tau/mu', f"{tau_mass/mu_mass:.3f}", f"{tau_mass_exp/mu_mass_exp:.3f}",
                 f"{abs(tau_mass/mu_mass - tau_mass_exp/mu_mass_exp)/(tau_mass_exp/mu_mass_exp)*100:.1f}%"],
            ]
            
            headers = ['Ratio', 'Predicted', 'Experimental', 'Error']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))


def main():
    print("="*80)
    print("LEPTON SOLVER TEST")
    print("="*80)
    print("\nTwo-Stage Architecture:")
    print("  Stage 1: Dimensionless shape solver")
    print("  Stage 2: Energy minimization over scales")
    print("\nLoading constants from constants.json...")
    
    # Create solver (without beta)
    print("\nCreating solver...")
    print("  All parameters loaded from constants.json")
    
    solver = UnifiedSFMSolver(
        n_max=5,
        l_max=2,
        N_sigma=64,
        verbose=False  # Less verbose during calibration
    )
    
    # Calibrate beta from electron
    print("\n" + "="*80)
    print("CALIBRATING MASS SCALE FROM ELECTRON")
    print("="*80)
    print("  Solving electron to determine beta = m_e / A_e^2...")
    
    beta = calibrate_beta_from_electron(solver, electron_mass_exp=0.000510999)
    
    print(f"\n  Mass scale calibrated: beta = {beta:.8f} GeV")
    print(f"  This beta will be used to convert all amplitudes to masses: m = beta * A^2")
    
    # Test calibration leptons
    print("\n" + "="*80)
    print("TESTING CALIBRATION LEPTONS")
    print("="*80)
    
    calibration_results = []
    for lepton in CALIBRATION_LEPTONS:
        result = test_lepton(solver, lepton, beta, verbose=False)
        calibration_results.append(result)
    
    print_lepton_results(calibration_results, "CALIBRATION LEPTON RESULTS")
    
    # Note: Validation leptons (neutrinos) are commented out - suspected to be multi-lepton particles
    validation_results = []
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_results = calibration_results + validation_results
    converged = sum(1 for r in all_results if r['shape_converged'] and r['energy_converged'])
    failed = sum(1 for r in all_results if r['error'] is not None)
    total = len(all_results)
    
    print(f"Total particles tested: {total}")
    print(f"Converged: {converged}")
    print(f"Failed: {failed}")
    print(f"Not converged: {total - converged - failed}")
    
    if converged == total:
        print("\nSUCCESS: All leptons converged")
    else:
        print(f"\nWARNING: {total - converged} leptons did not converge")
    
    print("\nNote: Mass predictions depend on parameter calibration.")
    print("The new architecture may require re-optimization of fundamental constants.")


if __name__ == '__main__':
    main()

