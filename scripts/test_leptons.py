"""
Test Script for Lepton Solving.

Tests the new two-stage solver architecture on all leptons:
- Calibration: electron, muon, tau
- Validation: electron neutrino, muon neutrino, tau neutrino

Reports masses, amplitudes, scale parameters, convergence metrics,
bounds status, energy components, and compares to experimental values.

All output is saved to a timestamped log file in the logs folder.
"""

import numpy as np
import time
import sys
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.particle_configurations import (
    CALIBRATION_LEPTONS,
)

# Bounds for scale parameters (must match energy_minimizer.py)
MIN_DELTA_X = 0.001
MAX_DELTA_X = 100000.0  # Widened to allow escape from local minima
# Delta_sigma is no longer optimized - it's fixed from Stage 1

class LogWriter:
    """Writes output to both console and log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def check_bounds_status(Delta_x):
    """Check if Delta_x hit bounds. Delta_sigma is FIXED from Stage 1."""
    dx_status = "OK"
    
    # Check Delta_x (use relative tolerance of 1% for large bounds)
    if abs(Delta_x - MIN_DELTA_X) / max(MIN_DELTA_X, 0.001) < 0.01:
        dx_status = "MIN"
    elif abs(Delta_x - MAX_DELTA_X) / MAX_DELTA_X < 0.01:
        dx_status = "MAX"
    
    any_at_bound = (dx_status != "OK")
    
    return {
        'dx_status': dx_status,
        'any_at_bound': any_at_bound
    }


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
        
        # Check bounds status
        bounds = check_bounds_status(result.Delta_x)
        
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
            'bounds_status': bounds,
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
        print(f"    A (amplitude):   {r['amplitude']:.6f} [optimized]")
        print(f"    Delta_x:         {r['Delta_x_fm']:.6f} fm [optimized]")
        print(f"    Delta_sigma:     {r['Delta_sigma']:.6f} [fixed from Stage 1]")
        
        # Bounds status
        if r.get('bounds_status'):
            bounds = r['bounds_status']
            dx_indicator = " (AT BOUND)" if bounds['dx_status'] != "OK" else ""
            print(f"  Bounds Status:")
            print(f"    Delta_x:         {bounds['dx_status']}{dx_indicator}")
            print(f"    Delta_sigma:     N/A (fixed from Stage 1)")
            if bounds['any_at_bound']:
                print(f"    WARNING: Delta_x at boundary!")
        
        # Mass prediction
        print(f"  Mass Prediction:")
        print(f"    Predicted:       {r['mass_predicted_MeV']:.6f} MeV")
        print(f"    Experimental:    {r['mass_experimental_MeV']:.6f} MeV")
        if r['error_percent'] is not None:
            print(f"    Error:           {r['error_percent']:.2f}%")
        
        # Energy components breakdown
        if r.get('energy_components'):
            comp = r['energy_components']
            E_total = r['E_total']
            print(f"  Energy Components (MeV):")
            print(f"    {'Component':<20} {'Value (MeV)':>15} {'% of |Total|':>12}")
            print(f"    {'-'*20} {'-'*15} {'-'*12}")
            
            abs_total = abs(E_total * 1000) if E_total != 0 else 1.0  # Convert GeV to MeV
            
            # Spatial
            E_spatial = comp.get('E_spatial', 0) * 1000
            print(f"    {'E_spatial':<20} {E_spatial:>15.3f} {100*abs(E_spatial)/abs_total:>11.1f}%")
            
            # Subspace (total and components)
            E_sigma = comp.get('E_sigma', 0) * 1000
            E_sigma_kin = comp.get('E_kinetic_sigma', 0) * 1000
            E_sigma_pot = comp.get('E_potential_sigma', 0) * 1000
            E_sigma_nl = comp.get('E_nonlinear_sigma', 0) * 1000
            print(f"    {'E_sigma (total)':<20} {E_sigma:>15.3f} {100*abs(E_sigma)/abs_total:>11.1f}%")
            print(f"      {'E_kinetic_sigma':<18} {E_sigma_kin:>15.3f} {100*abs(E_sigma_kin)/abs_total:>11.1f}%")
            print(f"      {'E_potential_sigma':<18} {E_sigma_pot:>15.3f} {100*abs(E_sigma_pot)/abs_total:>11.1f}%")
            print(f"      {'E_nonlinear_sigma':<18} {E_sigma_nl:>15.3f} {100*abs(E_sigma_nl)/abs_total:>11.1f}%")
            
            # Coupling
            E_coupling = comp.get('E_coupling', 0) * 1000
            print(f"    {'E_coupling':<20} {E_coupling:>15.3f} {100*abs(E_coupling)/abs_total:>11.1f}%")
            
            # EM
            E_em = comp.get('E_em', 0) * 1000
            print(f"    {'E_em':<20} {E_em:>15.3f} {100*abs(E_em)/abs_total:>11.1f}%")
            
            # Spin-orbit
            E_spin_orbit = comp.get('E_spin_orbit', 0) * 1000
            print(f"    {'E_spin_orbit':<20} {E_spin_orbit:>15.3f} {100*abs(E_spin_orbit)/abs_total:>11.1f}%")
            
            # Curvature
            E_curvature = comp.get('E_curvature', 0) * 1000
            print(f"    {'E_curvature':<20} {E_curvature:>15.3f} {100*abs(E_curvature)/abs_total:>11.1f}%")
            
            # Total
            E_total_mev = E_total * 1000
            print(f"    {'-'*20} {'-'*15} {'-'*12}")
            print(f"    {'E_total':<20} {E_total_mev:>15.3f} {'100.0%':>12}")
            
            # Verify subspace components sum to total
            E_sigma_sum = E_sigma_kin + E_sigma_pot + E_sigma_nl
            if abs(E_sigma - E_sigma_sum) > 0.001:
                print(f"    WARNING: E_sigma components don't sum correctly!")
                print(f"             E_sigma = {E_sigma:.3f}, sum of components = {E_sigma_sum:.3f}")
        
        # Show convergence evolution (last 10 iterations)
        if r.get('scale_history') and len(r['scale_history']['A']) > 1:
            print(f"\n  Scale Evolution (last 10 iterations):")
            n_show = min(10, len(r['scale_history']['A']))
            start_idx = max(0, len(r['scale_history']['A']) - n_show)
            
            # Delta_sigma is FIXED from Stage 1, not in scale_history
            print(f"    {'Iter':<6} {'A':>12} {'Delta_x (fm)':>14}")
            print(f"    {'-'*6} {'-'*12} {'-'*14}")
            for i in range(start_idx, len(r['scale_history']['A'])):
                iter_num = r['scale_history']['iteration'][i]
                A = r['scale_history']['A'][i]
                Dx = r['scale_history']['Delta_x'][i]
                print(f"    {iter_num:<6} {A:12.6f} {Dx:14.6f}")
            print(f"    (Delta_sigma = {r['Delta_sigma']:.6f}, FIXED from Stage 1)")
            
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
    # Set up log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"leptons_log_{timestamp}.txt"
    
    # Redirect output to both console and log file
    log_writer = LogWriter(log_path)
    sys.stdout = log_writer
    
    try:
        print("="*80)
        print("LEPTON SOLVER TEST")
        print("="*80)
        print(f"\nLog file: {log_path}")
        print(f"Timestamp: {timestamp}")
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
    
        # Get fundamental constants from solver
        G_5D = solver.G_5D
        alpha = solver.alpha
        V0 = solver.V0
        g1 = solver.g1
        g2 = solver.g2
        lambda_so = solver.lambda_so
        
        print("\n" + "="*80)
        print("FUNDAMENTAL CONSTANTS")
        print("="*80)
        print(f"  G_5D      = {G_5D:.6f} GeV^-2   (5D gravitational constant)")
        print(f"  alpha     = {alpha:.1f}          (Spacetime-subspace coupling)")
        print(f"  V0        = {V0:.6f} GeV        (Subspace potential depth)")
        print(f"  g1        = {g1:.1f}            (Nonlinear self-interaction)")
        print(f"  g2        = {g2:.6f}           (EM circulation coupling)")
        print(f"  lambda_so = {lambda_so:.6f}           (Spin-orbit coupling)")
        print(f"  Generation scaling: f(n) = n^2  (QUADRATIC scaling with POSITIVE coupling cost)")
        print(f"  Bounds: Delta_x in [{MIN_DELTA_X}, {MAX_DELTA_X}] fm")
        print(f"          Delta_sigma FIXED from Stage 1 natural width")
        
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
        at_bounds = sum(1 for r in all_results if r.get('bounds_status', {}).get('any_at_bound', False))
        total = len(all_results)
        
        print(f"Total particles tested: {total}")
        print(f"Converged: {converged}")
        print(f"Failed: {failed}")
        print(f"Not converged: {total - converged - failed}")
        print(f"At bounds: {at_bounds}")
        
        if converged == total and at_bounds == 0:
            print("\nSUCCESS: All leptons converged with natural (non-boundary) solutions")
        elif converged == total:
            print(f"\nPARTIAL SUCCESS: All leptons converged, but {at_bounds} hit boundary constraints")
        else:
            print(f"\nWARNING: {total - converged} leptons did not converge")
        
        print("\nNote: Masses calculated using The Beautiful Equation: m = G_5D × c × A^2")
        print(f"      Current G_5D = {G_5D:.6f} GeV^-2")
        print("      Mass predictions depend on fundamental constants.")
        print("      The fully coupled architecture requires proper parameter calibration.")
        print(f"\nLog saved to: {log_path}")
        
    finally:
        # Restore stdout and close log file
        sys.stdout = log_writer.terminal
        log_writer.close()
        print(f"\nLog file saved: {log_path}")


if __name__ == '__main__':
    main()

