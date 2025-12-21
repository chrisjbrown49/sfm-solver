"""
Performance Test Script for Unified Solver.

Tests the new two-stage solver architecture on calibration particles:
- Leptons: electron, muon, tau
- Baryons: proton, neutron

Reports masses, amplitudes, scale parameters, convergence metrics,
and compares to experimental values.
"""

import numpy as np
import time
from tabulate import tabulate

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.calculate_beta import calibrate_beta_from_electron
from sfm_solver.core.particle_configurations import ELECTRON, MUON, TAU, PROTON, NEUTRON


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
    
    result = solver.solve_lepton(
        winding_k=lepton_config.winding,
        generation_n=lepton_config.generation,
        max_iter=200,
        tol=1e-6
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"  Shape converged: {result.shape_converged} ({result.shape_iterations} iters)")
        print(f"  Energy converged: {result.energy_converged} ({result.energy_iterations} iters)")
        print(f"  Time: {elapsed_time:.3f} s")
    
    # Calculate mass from amplitude using calibrated beta
    mass_mev = beta * (result.A ** 2) * 1000.0
    
    # Compute error
    if lepton_config.mass_exp > 0:
        error_percent = abs(mass_mev - lepton_config.mass_exp) / lepton_config.mass_exp * 100.0
    else:
        error_percent = 0.0
    
    return {
        'name': lepton_config.name,
        'generation': lepton_config.generation,
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
        'E_total': result.E_total
    }


def test_baryon(solver, baryon_config, beta, verbose=False):
    """Test baryon and return results.
    
    Args:
        solver: UnifiedSFMSolver instance
        baryon_config: Particle configuration
        beta: Mass scale (GeV) from electron calibration
        verbose: Print detailed output
    """
    print(f"\nTesting {baryon_config.name}...")
    start_time = time.time()
    
    # Use default color phases for color neutrality: (0, 2π/3, 4π/3)
    result = solver.solve_baryon(
        quark_windings=baryon_config.windings,
        color_phases=(0, 2*np.pi/3, 4*np.pi/3),
        n_target=1,
        max_scf_iter=300,
        scf_tol=1e-4,
        scf_mixing=0.1
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"  Shape converged: {result.shape_converged} ({result.shape_iterations} iters)")
        print(f"  Energy converged: {result.energy_converged} ({result.energy_iterations} iters)")
        print(f"  Time: {elapsed_time:.3f} s")
    
    # Calculate mass from amplitude using calibrated beta
    mass_mev = beta * (result.A ** 2) * 1000.0
    
    # Compute error
    error_percent = abs(mass_mev - baryon_config.mass_exp) / baryon_config.mass_exp * 100.0
    
    return {
        'name': baryon_config.name,
        'quark_content': baryon_config.quarks,
        'mass_predicted_MeV': mass_mev,
        'mass_experimental_MeV': baryon_config.mass_exp,
        'error_percent': error_percent,
        'amplitude': result.A,
        'Delta_x_fm': result.Delta_x,
        'Delta_sigma': result.Delta_sigma,
        'shape_converged': result.shape_converged,
        'shape_iterations': result.shape_iterations,
        'energy_converged': result.energy_converged,
        'energy_iterations': result.energy_iterations,
        'time_seconds': elapsed_time,
        'E_total': result.E_total
    }


def print_lepton_results(results):
    """Print lepton results in table format."""
    print("\n" + "="*80)
    print("LEPTON RESULTS")
    print("="*80)
    
    # Basic results table
    table_data = []
    for r in results:
        table_data.append([
            r['name'].capitalize(),
            f"n={r['generation']}",
            f"{r['mass_predicted_MeV']:.6f}",
            f"{r['mass_experimental_MeV']:.6f}",
            f"{r['error_percent']:.1f}%",
            "Yes" if r['shape_converged'] and r['energy_converged'] else "No"
        ])
    
    headers = ['Particle', 'Gen', 'Predicted (MeV)', 'Experimental (MeV)', 'Error', 'Converged']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Scale parameters table
    print("\n" + "-"*80)
    print("SCALE PARAMETERS")
    print("-"*80)
    
    table_data = []
    for r in results:
        table_data.append([
            r['name'].capitalize(),
            f"{r['amplitude']:.6f}",
            f"{r['Delta_x_fm']:.6f}",
            f"{r['Delta_sigma']:.6f}",
            f"{r['time_seconds']:.3f}"
        ])
    
    headers = ['Particle', 'Amplitude A', 'Delta_x (fm)', 'Delta_sigma', 'Time (s)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Check mass ratios
    if len(results) == 3:
        m_e = results[0]['mass_predicted_MeV']
        m_mu = results[1]['mass_predicted_MeV']
        m_tau = results[2]['mass_predicted_MeV']
        
        m_e_exp = results[0]['mass_experimental_MeV']
        m_mu_exp = results[1]['mass_experimental_MeV']
        m_tau_exp = results[2]['mass_experimental_MeV']
        
        print("\n" + "-"*80)
        print("MASS RATIOS")
        print("-"*80)
        
        ratio_data = [
            ['mu/e', f"{m_mu/m_e:.3f}", f"{m_mu_exp/m_e_exp:.3f}", 
             f"{abs(m_mu/m_e - m_mu_exp/m_e_exp)/(m_mu_exp/m_e_exp)*100:.1f}%"],
            ['tau/e', f"{m_tau/m_e:.3f}", f"{m_tau_exp/m_e_exp:.3f}",
             f"{abs(m_tau/m_e - m_tau_exp/m_e_exp)/(m_tau_exp/m_e_exp)*100:.1f}%"],
            ['tau/mu', f"{m_tau/m_mu:.3f}", f"{m_tau_exp/m_mu_exp:.3f}",
             f"{abs(m_tau/m_mu - m_tau_exp/m_mu_exp)/(m_tau_exp/m_mu_exp)*100:.1f}%"]
        ]
        
        headers = ['Ratio', 'Predicted', 'Experimental', 'Error']
        print(tabulate(ratio_data, headers=headers, tablefmt='grid'))


def print_baryon_results(results):
    """Print baryon results in table format."""
    print("\n" + "="*80)
    print("BARYON RESULTS")
    print("="*80)
    
    # Basic results table
    table_data = []
    for r in results:
        table_data.append([
            r['name'].capitalize(),
            r['quark_content'],
            f"{r['mass_predicted_MeV']:.3f}",
            f"{r['mass_experimental_MeV']:.3f}",
            f"{r['error_percent']:.1f}%",
            "Yes" if r['shape_converged'] and r['energy_converged'] else "No"
        ])
    
    headers = ['Particle', 'Quarks', 'Predicted (MeV)', 'Experimental (MeV)', 'Error', 'Converged']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Scale parameters table
    print("\n" + "-"*80)
    print("SCALE PARAMETERS")
    print("-"*80)
    
    table_data = []
    for r in results:
        table_data.append([
            r['name'].capitalize(),
            f"{r['amplitude']:.6f}",
            f"{r['Delta_x_fm']:.6f}",
            f"{r['Delta_sigma']:.6f}",
            f"{r['time_seconds']:.3f}"
        ])
    
    headers = ['Particle', 'Amplitude A', 'Delta_x (fm)', 'Delta_sigma', 'Time (s)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Mass splitting
    if len(results) == 2:
        print("\n" + "-"*80)
        print("NEUTRON-PROTON MASS SPLITTING")
        print("-"*80)
        
        m_p = results[0]['mass_predicted_MeV']
        m_n = results[1]['mass_predicted_MeV']
        
        m_p_exp = results[0]['mass_experimental_MeV']
        m_n_exp = results[1]['mass_experimental_MeV']
        
        splitting_pred = m_n - m_p
        splitting_exp = m_n_exp - m_p_exp
        
        print(f"  Predicted: {splitting_pred:.6f} MeV")
        print(f"  Experimental: {splitting_exp:.6f} MeV")
        print(f"  Error: {abs(splitting_pred - splitting_exp):.6f} MeV")


def main():
    """Main performance test."""
    print("="*80)
    print("UNIFIED SOLVER PERFORMANCE TEST")
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
    
    # CRITICAL: Calibrate beta from electron FIRST
    print("\n" + "="*80)
    print("CALIBRATING MASS SCALE FROM ELECTRON")
    print("="*80)
    print("  Solving electron to determine beta = m_e / A_e^2...")
    
    beta = calibrate_beta_from_electron(solver, electron_mass_exp=0.000510999)
    
    print(f"\n  Mass scale calibrated: beta = {beta:.8f} GeV")
    print(f"  Electron mass: {beta * 1.0**2 * 1000:.6f} MeV (by construction)")
    print(f"  This beta will be used to convert all particle amplitudes to masses")
    
    # Test leptons
    print("\n" + "="*80)
    print("TESTING LEPTONS")
    print("="*80)
    
    lepton_results = []
    for lepton_config in [ELECTRON, MUON, TAU]:
        try:
            result = test_lepton(solver, lepton_config, beta, verbose=True)
            lepton_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    if lepton_results:
        print_lepton_results(lepton_results)
    
    # Test baryons
    print("\n" + "="*80)
    print("TESTING BARYONS")
    print("="*80)
    
    baryon_results = []
    for baryon_config in [PROTON, NEUTRON]:
        try:
            result = test_baryon(solver, baryon_config, beta, verbose=True)
            baryon_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    if baryon_results:
        print_baryon_results(baryon_results)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_converged = all(
        r['shape_converged'] and r['energy_converged'] 
        for r in lepton_results + baryon_results
    )
    
    if all_converged:
        print("SUCCESS: All particles converged")
    else:
        print("WARNING: Some particles did not converge")
    
    print("\nNote: Mass predictions depend on parameter calibration.")
    print("The new architecture may require re-optimization of fundamental constants.")
    print("\nNext step: Run parameter_optimizer.py to find optimal parameters")
    print("for the new two-stage architecture.")


if __name__ == '__main__':
    main()

