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
from sfm_solver.core.particle_configurations import (
    CALIBRATION_LEPTONS,
)


def test_lepton(solver, lepton_config, G_5D, verbose=False):
    """Test lepton and return results.
    
    Args:
        solver: UnifiedSFMSolver instance
        lepton_config: Particle configuration
        G_5D: 5D gravitational constant (GeV^-2)
        verbose: Print detailed output
    """
    print(f"\nTesting {lepton_config.name}...")
    start_time = time.time()
    
    try:
        result = solver.solve_lepton(
            winding_k=lepton_config.winding,
            generation_n=lepton_config.generation,
            max_iter=200,
            tol=1e-6,  # Shape solver tolerance (Stage 1)
            max_iter_outer=100,  # Outer loop iterations
            tol_outer=1e-4  # Outer loop tolerance
        )
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"  Shape converged: {result.shape_converged} ({result.shape_iterations} iters)")
            print(f"  Energy converged: {result.energy_converged} ({result.energy_iterations} iters)")
            print(f"  Outer loop: {result.outer_iterations} iterations, converged={result.outer_converged}")
            print(f"  Time: {elapsed_time:.3f} s")
            
            # Show scale evolution for first few iterations
            if result.scale_history and len(result.scale_history['A']) > 1:
                print(f"\n  Scale evolution (first 5 iterations):")
                for i in range(min(5, len(result.scale_history['A']))):
                    A = result.scale_history['A'][i]
                    Dx = result.scale_history['Delta_x'][i]
                    Ds = result.scale_history['Delta_sigma'][i]
                    print(f"    Iter {i}: A={A:.6f}, Dx={Dx:.6f} fm, Ds={Ds:.6f}")
        
        # Calculate mass from amplitude using "The Beautiful Equation": m = G_5D × c × A²
        # In natural units (c = 1): m = G_5D × A² (in GeV)
        mass_gev = G_5D * (result.A ** 2)
        mass_mev = mass_gev * 1000.0
        
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
            'outer_converged': result.outer_converged,
            'outer_iterations': result.outer_iterations,
            'scale_history': result.scale_history,
            'shape_solver_history': result.shape_solver_history,
            'energy_minimizer_history': result.energy_minimizer_history,
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


def print_convergence_report(results):
    """Print detailed convergence report for each particle."""
    print("\n" + "="*80)
    print("DETAILED CONVERGENCE REPORT")
    print("="*80)
    
    for r in results:
        if r.get('error') is not None:
            continue  # Skip failed particles
            
        print(f"\n{'-'*80}")
        print(f"{r['name'].replace('_', ' ').title()} (n={r['generation']})")
        print(f"{'-'*80}")
        
        # Convergence status with detailed tracking
        print(f"  Shape Solver (Stage 1) - LAST ITERATION:")
        print(f"    Converged: {r['shape_converged']} after {r['shape_iterations']} iterations")
        
        # Shape solver statistics across ALL outer loop iterations
        if r.get('shape_solver_history'):
            ssh = r['shape_solver_history']
            total_shape_calls = len(ssh['iterations'])
            total_shape_iters = sum(ssh['iterations'])
            avg_shape_iters = total_shape_iters / total_shape_calls if total_shape_calls > 0 else 0
            shape_failures = sum(1 for c in ssh['converged'] if not c)
            min_shape_iters = min(ssh['iterations']) if ssh['iterations'] else 0
            max_shape_iters = max(ssh['iterations']) if ssh['iterations'] else 0
            
            print(f"  Shape Solver - ALL {total_shape_calls} OUTER LOOP ITERATIONS:")
            print(f"    Total solver calls:    {total_shape_calls}")
            print(f"    Total iterations:      {total_shape_iters}")
            print(f"    Avg iterations/call:   {avg_shape_iters:.1f}")
            print(f"    Min iterations:        {min_shape_iters}")
            print(f"    Max iterations:        {max_shape_iters}")
            print(f"    Failed to converge:    {shape_failures}")
        
        print(f"\n  Energy Minimizer (Stage 2 inner loop) - LAST ITERATION:")
        print(f"    Converged: {r['energy_converged']} after {r['energy_iterations']} iterations")
        
        # Energy minimizer statistics across ALL outer loop iterations
        if r.get('energy_minimizer_history'):
            emh = r['energy_minimizer_history']
            total_energy_calls = len(emh['iterations'])
            total_energy_iters = sum(emh['iterations'])
            avg_energy_iters = total_energy_iters / total_energy_calls if total_energy_calls > 0 else 0
            energy_failures = sum(1 for c in emh['converged'] if not c)
            min_energy_iters = min(emh['iterations']) if emh['iterations'] else 0
            max_energy_iters = max(emh['iterations']) if emh['iterations'] else 0
            
            print(f"  Energy Minimizer - ALL {total_energy_calls} OUTER LOOP ITERATIONS:")
            print(f"    Total minimizer calls: {total_energy_calls}")
            print(f"    Total iterations:      {total_energy_iters}")
            print(f"    Avg iterations/call:   {avg_energy_iters:.1f}")
            print(f"    Min iterations:        {min_energy_iters}")
            print(f"    Max iterations:        {max_energy_iters}")
            print(f"    Failed to converge:    {energy_failures}")
        
        print(f"\n  Outer Loop (Stage 1 <-> Stage 2 iteration):")
        print(f"    Converged: {r['outer_converged']} after {r['outer_iterations']} iterations")
        print(f"    Tolerance: 1e-4 (0.01% relative change)")
        
        # Final scale parameters
        print(f"  Final Scale Parameters:")
        print(f"    A (amplitude):   {r['amplitude']:.6f}")
        print(f"    Delta_x:         {r['Delta_x_fm']:.6f} fm")
        print(f"    Delta_sigma:     {r['Delta_sigma']:.6f}")
        
        # Mass prediction
        print(f"  Mass Prediction:")
        print(f"    Predicted:       {r['mass_predicted_MeV']:.6f} MeV")
        print(f"    Experimental:    {r['mass_experimental_MeV']:.6f} MeV")
        if r['error_percent'] is not None:
            print(f"    Error:           {r['error_percent']:.2f}%")
        
        # Show convergence evolution (last 10 iterations)
        if r.get('scale_history') and len(r['scale_history']['A']) > 1:
            print(f"\n  Scale Evolution (last 10 iterations):")
            n_show = min(10, len(r['scale_history']['A']))
            start_idx = max(0, len(r['scale_history']['A']) - n_show)
            
            print(f"    {'Iter':<6} {'A':>12} {'Delta_x (fm)':>14} {'Delta_sigma':>14}")
            print(f"    {'-'*6} {'-'*12} {'-'*14} {'-'*14}")
            for i in range(start_idx, len(r['scale_history']['A'])):
                iter_num = r['scale_history']['iteration'][i]
                A = r['scale_history']['A'][i]
                Dx = r['scale_history']['Delta_x'][i]
                Ds = r['scale_history']['Delta_sigma'][i]
                print(f"    {iter_num:<6} {A:12.6f} {Dx:14.6f} {Ds:14.6f}")
            
            # Show convergence rate if available
            if len(r['scale_history']['A']) >= 2:
                A_init = r['scale_history']['A'][0]
                A_final = r['scale_history']['A'][-1]
                A_change = abs(A_final - A_init)
                A_rel_change = A_change / max(abs(A_init), 0.01) * 100
                print(f"\n    Total change in A: {A_change:.6f} ({A_rel_change:.2f}%)")
        
        print(f"  Computation Time: {r['time_seconds']:.2f} seconds")


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
    
    # Create solver
    print("\nCreating solver...")
    print("  All parameters loaded from constants.json")
    
    solver = UnifiedSFMSolver(
        n_max=5,
        l_max=2,
        N_sigma=64,
        verbose=False
    )
    
    # Get G_5D from solver for mass calculation
    G_5D = solver.G_5D
    print("\n" + "="*80)
    print("MASS CALCULATION USING THE BEAUTIFUL EQUATION")
    print("="*80)
    print(f"  G_5D = {G_5D:.6f} GeV^-2")
    print("  Mass formula: m = G_5D × c × A^2")
    print("  (In natural units: m = G_5D × A^2)")
    
    # Test leptons
    print("\n" + "="*80)
    print("TESTING LEPTONS")
    print("="*80)
    
    calibration_results = []
    for lepton in CALIBRATION_LEPTONS:
        result = test_lepton(solver, lepton, G_5D, verbose=False)
        calibration_results.append(result)
    
    # Print detailed convergence report first
    print_convergence_report(calibration_results)
    
    # Then print summary tables
    print_lepton_results(calibration_results, "LEPTON RESULTS")
    
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
    
    print("\nNote: Masses calculated using The Beautiful Equation: m = G_5D × c × A^2")
    print(f"      Current G_5D = {G_5D:.6f} GeV^-2")
    print("      Mass predictions depend on fundamental constants.")
    print("      The fully coupled architecture requires proper parameter calibration.")


if __name__ == '__main__':
    main()

