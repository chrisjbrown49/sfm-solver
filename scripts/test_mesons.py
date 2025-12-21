"""
Test Script for Meson Solving.

Tests the new two-stage solver architecture on all mesons:
- Calibration: pi+, J/psi
- Validation: pi-, pi0, psi(2S), Upsilon(1S), Upsilon(2S)

Reports masses, amplitudes, scale parameters, convergence metrics,
and compares to experimental values.
"""

import numpy as np
import time
from tabulate import tabulate

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.particle_configurations import (
    CALIBRATION_MESONS,
    VALIDATION_MESONS,
)


def test_meson(solver, meson_config, verbose=False):
    """Test meson and return results."""
    print(f"\nTesting {meson_config.name}...")
    start_time = time.time()
    
    try:
        result = solver.solve_meson(
            quark_winding=meson_config.quark1_winding,
            antiquark_winding=meson_config.quark2_winding,
            quark_generation=meson_config.quark1_generation,
            antiquark_generation=meson_config.quark2_generation,
            n_target=1,
            max_scf_iter=500,
            scf_tol=1e-4,
            scf_mixing=0.1
        )
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"  Shape converged: {result.shape_converged} ({result.shape_iterations} iters)")
            print(f"  Energy converged: {result.energy_converged} ({result.energy_iterations} iters)")
            print(f"  Time: {elapsed_time:.3f} s")
        
        # Convert mass to MeV
        mass_mev = result.mass * 1000.0
        
        # Compute error
        error_percent = abs(mass_mev - meson_config.mass_exp) / meson_config.mass_exp * 100.0
        
        return {
            'name': meson_config.name,
            'quarks': meson_config.quarks,
            'spin': meson_config.spin,
            'q1_winding': meson_config.quark1_winding,
            'q2_winding': meson_config.quark2_winding,
            'q1_generation': meson_config.quark1_generation,
            'q2_generation': meson_config.quark2_generation,
            'mass_predicted_MeV': mass_mev,
            'mass_experimental_MeV': meson_config.mass_exp,
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
            'name': meson_config.name,
            'quarks': meson_config.quarks,
            'spin': meson_config.spin,
            'q1_winding': meson_config.quark1_winding,
            'q2_winding': meson_config.quark2_winding,
            'q1_generation': meson_config.quark1_generation,
            'q2_generation': meson_config.quark2_generation,
            'mass_predicted_MeV': None,
            'mass_experimental_MeV': meson_config.mass_exp,
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


def print_meson_results(results, title):
    """Print meson results in table format."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # Basic results table
    table_data = []
    for r in results:
        if r['mass_predicted_MeV'] is not None:
            mass_str = f"{r['mass_predicted_MeV']:.3f}"
        else:
            mass_str = "FAILED"
        
        if r['error_percent'] is not None:
            error_str = f"{r['error_percent']:.1f}%"
        else:
            error_str = "FAILED"
        
        converged = "Yes" if (r['shape_converged'] and r['energy_converged']) else "No"
        
        spin_str = "S=0" if r['spin'] == 0 else "S=1"
        
        table_data.append([
            r['name'],
            r['quarks'],
            spin_str,
            mass_str,
            f"{r['mass_experimental_MeV']:.3f}",
            error_str,
            converged
        ])
    
    headers = ['Particle', 'Quarks', 'Spin', 'Predicted (MeV)', 'Experimental (MeV)', 'Error', 'Converged']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Scale parameters table
    print("\n" + "-"*80)
    print("SCALE PARAMETERS")
    print("-"*80)
    
    table_data = []
    for r in results:
        if r['amplitude'] is not None:
            table_data.append([
                r['name'],
                f"({r['q1_winding']},{r['q2_winding']})",
                f"({r['q1_generation']},{r['q2_generation']})",
                f"{r['amplitude']:.6f}",
                f"{r['Delta_x_fm']:.6f}",
                f"{r['Delta_sigma']:.6f}",
                f"{r['shape_iterations']}/{r['energy_iterations']}",
                f"{r['time_seconds']:.3f}"
            ])
        else:
            table_data.append([
                r['name'],
                f"({r['q1_winding']},{r['q2_winding']})",
                f"({r['q1_generation']},{r['q2_generation']})",
                "FAILED",
                "FAILED",
                "FAILED",
                "FAILED",
                f"{r['time_seconds']:.3f}"
            ])
    
    headers = ['Particle', '(k_q,k_qbar)', '(n_q,n_qbar)', 'Amplitude A', 'Delta_x (fm)', 'Delta_sigma', 'Iters (S/E)', 'Time (s)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Energy breakdown for converged particles
    converged_results = [r for r in results if r['energy_components'] is not None]
    if converged_results:
        print("\n" + "-"*80)
        print("ENERGY BREAKDOWN (GeV)")
        print("-"*80)
        
        table_data = []
        for r in converged_results:
            ec = r['energy_components']
            table_data.append([
                r['name'],
                f"{ec['E_sigma']:.3e}",
                f"{ec['E_spatial']:.3e}",
                f"{ec['E_coupling']:.3e}",
                f"{ec['E_curvature']:.3e}",
                f"{ec['E_em']:.3e}",
                f"{r['E_total']:.3e}"
            ])
        
        headers = ['Particle', 'E_sigma', 'E_spatial', 'E_coupling', 'E_curv', 'E_EM', 'E_total']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))


def main():
    print("="*80)
    print("MESON SOLVER TEST")
    print("="*80)
    print("\nTwo-Stage Architecture:")
    print("  Stage 1: Dimensionless shape solver (SCF)")
    print("  Stage 2: Energy minimization over scales")
    print("\nLoading constants from constants.json...")
    
    # Create solver with auto-calibration
    print("\nCreating solver...")
    print("  Note: beta will be auto-calibrated from electron mass")
    print("  All other parameters loaded from constants.json")
    
    solver = UnifiedSFMSolver(
        auto_calibrate_beta=True,  # Calibrate from electron
        n_max=5,
        l_max=2,
        N_sigma=64,
        verbose=True
    )
    
    # Test calibration mesons
    print("\n" + "="*80)
    print("TESTING CALIBRATION MESONS")
    print("="*80)
    
    calibration_results = []
    for meson in CALIBRATION_MESONS:
        result = test_meson(solver, meson, verbose=False)
        calibration_results.append(result)
    
    print_meson_results(calibration_results, "CALIBRATION MESON RESULTS")
    
    # Test validation mesons
    print("\n" + "="*80)
    print("TESTING VALIDATION MESONS")
    print("="*80)
    
    validation_results = []
    for meson in VALIDATION_MESONS:
        result = test_meson(solver, meson, verbose=False)
        validation_results.append(result)
    
    print_meson_results(validation_results, "VALIDATION MESON RESULTS")
    
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
    
    # Check SCF convergence specifically
    scf_not_converged = sum(1 for r in all_results if not r['shape_converged'] and r['error'] is None)
    if scf_not_converged > 0:
        print(f"\nWARNING: {scf_not_converged} mesons failed SCF convergence")
        print("Consider:")
        print("  - Increasing max_scf_iter")
        print("  - Adjusting scf_mixing parameter")
        print("  - Checking for numerical stability issues")
    
    if converged == total:
        print("\nSUCCESS: All mesons converged")
    else:
        print(f"\nWARNING: {total - converged} mesons did not converge")
    
    print("\nNote: Mass predictions depend on parameter calibration.")
    print("The new architecture may require re-optimization of fundamental constants.")


if __name__ == '__main__':
    main()

