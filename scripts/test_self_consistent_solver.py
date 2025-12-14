#!/usr/bin/env python3
"""
Test script for the self-consistent delta_x solver.

This tests the implementation of Steps 1-3 from the missing_physics_fix.md plan:
- Step 1: Self-consistent delta_x iteration
- Step 2: Nonlinear feedback iteration  
- Step 3: Resonance enhancement

Expected improvements:
- Step 1: 10-100x improvement in mass ratios
- Step 2: 2-5x additional improvement
- Step 3: Fine-tuning to <10% of experimental

Targets:
- m_mu/m_e ~ 207 (experimental)
- m_tau/m_e ~ 3477 (experimental)
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.universal_energy_minimizer import UniversalEnergyMinimizer


def test_self_consistent_solver(verbose: bool = True):
    """Test the self-consistent delta_x solver on all three leptons."""
    
    print("=" * 70)
    print("TESTING SELF-CONSISTENT DELTA_X SOLVER")
    print("=" * 70)
    
    # Optimal parameters from missing_physics_report.md
    alpha = 10.5
    beta = 100.0  # Note: gives correct RATIOS but wrong absolute masses
    kappa = 0.00003
    g1 = 5000.0
    g2 = 0.004
    V0 = 1.0
    
    print(f"\nParameters:")
    print(f"  alpha = {alpha}, beta = {beta}, kappa = {kappa}")
    print(f"  g1 = {g1}, g2 = {g2}, V0 = {V0}")
    
    # Create solver
    solver = NonSeparableWavefunctionSolver(
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        g1=g1,
        g2=g2,
        V0=V0,
        n_max=5,
        l_max=2,
        N_sigma=64,
    )
    
    print(f"\nBasis: {solver.basis.N_spatial} spatial states x {solver.basis.N_sigma} sigma points")
    
    # Test different resonance widths
    resonance_widths = [0.0, 0.5, 1.0, 2.0]
    
    for gamma in resonance_widths:
        print(f"\n{'='*70}")
        print(f"Testing with Resonance Width Gamma = {gamma}")
        print(f"{'='*70}")
        
        results = {}
        
        for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
            print(f"\n--- {name} (n={n}) ---")
            
            result = solver.solve_lepton_self_consistent(
                n_target=n,
                k_winding=1,
                max_iter_outer=30,
                max_iter_nl=15,
                tol_outer=1e-5,
                tol_nl=1e-4,
                resonance_width=gamma,
                verbose=verbose,
            )
            
            results[name] = result
            
            print(f"\nResult for {name}:")
            print(f"  Converged: {result.converged} (iterations: {result.iterations})")
            print(f"  Final dx: {result.delta_x_final:.6f}" if result.delta_x_final else "  Final dx: N/A")
            print(f"  k_eff: {result.k_eff:.4f}")
            print(f"  Structure norm: {result.structure_norm:.6f}")
            print(f"  l-composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(result.l_composition.items()))}")
        
        # Compute mass ratios
        A_e = results['Electron'].structure_norm
        A_mu = results['Muon'].structure_norm
        A_tau = results['Tau'].structure_norm
        
        # Mass proportional to A^2, so ratio = (A_i/A_e)^2
        ratio_mu_e = (A_mu / A_e)**2
        ratio_tau_e = (A_tau / A_e)**2
        
        print(f"\n{'='*50}")
        print(f"MASS RATIOS (Gamma = {gamma})")
        print(f"{'='*50}")
        print(f"m_mu/m_e = {ratio_mu_e:.1f} (target: 207)")
        print(f"m_tau/m_e = {ratio_tau_e:.1f} (target: 3477)")
        
        # Check success criteria
        if 150 < ratio_mu_e < 300 and 2000 < ratio_tau_e < 5000:
            print("\n[SUCCESS] Mass ratios within expected range!")
        elif 20 < ratio_mu_e < 500:
            print("\n[PARTIAL] Significant improvement, but not yet at target")
        elif 5 < ratio_mu_e < 20:
            print("\n[PARTIAL] Some improvement, continuing to tune...")
        else:
            print("\n[NEEDS WORK] Mass ratios not yet improved enough")
        
        # Check delta_x scaling
        if results['Electron'].delta_x_final and results['Tau'].delta_x_final:
            Dx_e = results['Electron'].delta_x_final
            Dx_tau = results['Tau'].delta_x_final
            print(f"\ndelta_x scaling check:")
            print(f"  dx(e) = {Dx_e:.4f}")
            print(f"  dx(tau) = {Dx_tau:.4f}")
            print(f"  dx(e)/dx(tau) = {Dx_e/Dx_tau:.2f} (should be >> 1 if tau is more compact)")
    
    return results


def compare_original_vs_self_consistent():
    """Compare original solver with self-consistent solver."""
    
    print("\n" + "="*70)
    print("COMPARISON: Original vs Self-Consistent Solver")
    print("="*70)
    
    # Parameters
    alpha = 20.0
    beta = 100.0
    kappa = 0.0001
    g1 = 5000.0
    g2 = 0.004
    V0 = 1.0
    
    solver = NonSeparableWavefunctionSolver(
        alpha=alpha, beta=beta, kappa=kappa,
        g1=g1, g2=g2, V0=V0,
        n_max=5, l_max=2, N_sigma=64,
    )
    
    print("\n--- Original Solver (solve_lepton) ---")
    original_results = {}
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        result = solver.solve_lepton(n_target=n, k_winding=1, verbose=False)
        original_results[name] = result
        print(f"  {name}: norm = {result.structure_norm:.6f}, k_eff = {result.k_eff:.4f}")
    
    A_e_orig = original_results['Electron'].structure_norm
    A_mu_orig = original_results['Muon'].structure_norm
    A_tau_orig = original_results['Tau'].structure_norm
    
    ratio_mu_e_orig = (A_mu_orig / A_e_orig)**2
    ratio_tau_e_orig = (A_tau_orig / A_e_orig)**2
    
    print(f"\n  Mass ratios (original):")
    print(f"    m_mu/m_e = {ratio_mu_e_orig:.2f}")
    print(f"    m_tau/m_e = {ratio_tau_e_orig:.2f}")
    
    print("\n--- Self-Consistent Solver (solve_lepton_self_consistent) ---")
    print("    (Phase 2: Clean Step 1 - no phenomenological enhancements)")
    sc_results = {}
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        result = solver.solve_lepton_self_consistent(
            n_target=n, k_winding=1,
            max_iter_outer=30, max_iter_nl=0,  # Step 1 only - no nonlinear iteration
            resonance_width=0.0,  # No resonance enhancement
            verbose=False,
        )
        sc_results[name] = result
        print(f"  {name}: norm = {result.structure_norm:.6f}, k_eff = {result.k_eff:.4f}, dx = {result.delta_x_final:.4f}")
    
    A_e_sc = sc_results['Electron'].structure_norm
    A_mu_sc = sc_results['Muon'].structure_norm
    A_tau_sc = sc_results['Tau'].structure_norm
    
    ratio_mu_e_sc = (A_mu_sc / A_e_sc)**2
    ratio_tau_e_sc = (A_tau_sc / A_e_sc)**2
    
    print(f"\n  Mass ratios (self-consistent):")
    print(f"    m_mu/m_e = {ratio_mu_e_sc:.2f}")
    print(f"    m_tau/m_e = {ratio_tau_e_sc:.2f}")
    
    # Improvement factor
    improvement_mu = ratio_mu_e_sc / ratio_mu_e_orig if ratio_mu_e_orig > 0 else 0
    improvement_tau = ratio_tau_e_sc / ratio_tau_e_orig if ratio_tau_e_orig > 0 else 0
    
    print(f"\n  Improvement factors:")
    print(f"    mu/e ratio: {improvement_mu:.1f}x better")
    print(f"    tau/e ratio: {improvement_tau:.1f}x better")
    
    print("\n  Target ratios:")
    print(f"    m_mu/m_e = 207 (experimental)")
    print(f"    m_tau/m_e = 3477 (experimental)")
    
    # Check how close we are to target
    mu_error = abs(ratio_mu_e_sc - 207) / 207 * 100
    tau_error = abs(ratio_tau_e_sc - 3477) / 3477 * 100
    
    print(f"\n  Errors from target:")
    print(f"    mu/e: {mu_error:.1f}%")
    print(f"    tau/e: {tau_error:.1f}%")


def test_convergence_history():
    """Test and plot convergence history."""
    
    print("\n" + "="*70)
    print("CONVERGENCE HISTORY ANALYSIS")
    print("="*70)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=20.0, beta=100.0, kappa=0.0001,
        g1=5000.0, g2=0.004, V0=1.0,
        n_max=5, l_max=2, N_sigma=64,
    )
    
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        print(f"\n--- {name} Convergence History ---")
        
        result = solver.solve_lepton_self_consistent(
            n_target=n, k_winding=1,
            max_iter_outer=30, max_iter_nl=15,
            resonance_width=1.0,
            verbose=False,
        )
        
        history = result.convergence_history
        if history:
            print(f"  Iterations: {len(history['Delta_x'])}")
            print(f"  dx: {history['Delta_x'][0]:.4f} -> {history['Delta_x'][-1]:.4f}")
            print(f"  A:  {history['A'][0]:.4f} -> {history['A'][-1]:.4f}")
            if history['spatial_coupling_max']:
                print(f"  Max coupling: {history['spatial_coupling_max'][0]:.4f} -> {history['spatial_coupling_max'][-1]:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Test self-consistent solver")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    parser.add_argument('--compare', action='store_true', help="Compare original vs self-consistent")
    parser.add_argument('--history', action='store_true', help="Show convergence history")
    parser.add_argument('--all', '-a', action='store_true', help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all or not (args.compare or args.history):
        test_self_consistent_solver(verbose=args.verbose)
    
    if args.all or args.compare:
        compare_original_vs_self_consistent()
    
    if args.all or args.history:
        test_convergence_history()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
