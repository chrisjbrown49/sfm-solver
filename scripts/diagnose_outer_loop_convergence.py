"""
Diagnostic script to examine outer loop convergence behavior.

This script tracks the full evolution of scale parameters (A, Delta_x, Delta_sigma)
over outer loop iterations to identify convergence issues.
"""
import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver

def analyze_convergence_history(result, particle_name):
    """Analyze and display convergence history."""
    print(f"\n{'='*80}")
    print(f"CONVERGENCE ANALYSIS: {particle_name}")
    print(f"{'='*80}")
    
    if not result.scale_history or 'A' not in result.scale_history:
        print("No scale history available!")
        return
    
    A_hist = result.scale_history['A']
    Dx_hist = result.scale_history['Delta_x']
    Ds_hist = result.scale_history['Delta_sigma']
    
    n_iter = len(A_hist)
    
    print(f"\nOuter loop iterations: {result.outer_iterations}")
    print(f"Outer loop converged: {result.outer_converged}")
    print(f"Shape converged: {result.shape_converged}")
    print(f"Energy converged: {result.energy_converged}")
    
    print(f"\nFinal values:")
    print(f"  A          = {result.A:.6f}")
    print(f"  Delta_x    = {result.Delta_x:.6f} fm")
    print(f"  Delta_sigma= {result.Delta_sigma:.6f}")
    
    # Show first 10 iterations
    print(f"\n{'Iter':<6} {'A':<12} {'dA/A (%)':<12} {'Delta_x (fm)':<14} {'dDx/Dx (%)':<14} {'Delta_sigma':<12} {'dDs/Ds (%)':<12}")
    print("-" * 90)
    
    for i in range(min(10, n_iter)):
        A = A_hist[i]
        Dx = Dx_hist[i]
        Ds = Ds_hist[i]
        
        if i > 0:
            dA_pct = abs(A - A_hist[i-1]) / max(A_hist[i-1], 0.01) * 100
            dDx_pct = abs(Dx - Dx_hist[i-1]) / max(Dx_hist[i-1], 0.001) * 100
            dDs_pct = abs(Ds - Ds_hist[i-1]) / max(Ds_hist[i-1], 0.01) * 100
            print(f"{i:<6} {A:<12.6f} {dA_pct:<12.4f} {Dx:<14.6f} {dDx_pct:<14.4f} {Ds:<12.6f} {dDs_pct:<12.4f}")
        else:
            print(f"{i:<6} {A:<12.6f} {'---':<12} {Dx:<14.6f} {'---':<14} {Ds:<12.6f} {'---':<12}")
    
    if n_iter > 20:
        print("  ...")
        # Show last 10 iterations
        print("\nLast 10 iterations:")
        print(f"{'Iter':<6} {'A':<12} {'dA/A (%)':<12} {'Delta_x (fm)':<14} {'dDx/Dx (%)':<14} {'Delta_sigma':<12} {'dDs/Ds (%)':<12}")
        print("-" * 90)
        
        for i in range(max(n_iter - 10, 10), n_iter):
            A = A_hist[i]
            Dx = Dx_hist[i]
            Ds = Ds_hist[i]
            
            dA_pct = abs(A - A_hist[i-1]) / max(A_hist[i-1], 0.01) * 100
            dDx_pct = abs(Dx - Dx_hist[i-1]) / max(Dx_hist[i-1], 0.001) * 100
            dDs_pct = abs(Ds - Ds_hist[i-1]) / max(Ds_hist[i-1], 0.01) * 100
            print(f"{i:<6} {A:<12.6f} {dA_pct:<12.4f} {Dx:<14.6f} {dDx_pct:<14.4f} {Ds:<12.6f} {dDs_pct:<12.4f}")
    
    # Detect oscillations
    if n_iter > 5:
        print(f"\n{'='*80}")
        print("OSCILLATION DETECTION:")
        print(f"{'='*80}")
        
        # Check for sign changes in differences (oscillation)
        dA = np.diff(A_hist)
        dDx = np.diff(Dx_hist)
        dDs = np.diff(Ds_hist)
        
        # Count sign changes
        A_sign_changes = np.sum(np.diff(np.sign(dA)) != 0)
        Dx_sign_changes = np.sum(np.diff(np.sign(dDx)) != 0)
        Ds_sign_changes = np.sum(np.diff(np.sign(dDs)) != 0)
        
        print(f"Sign changes in differences (indicates oscillation):")
        print(f"  A:          {A_sign_changes} out of {len(dA)-1} steps")
        print(f"  Delta_x:    {Dx_sign_changes} out of {len(dDx)-1} steps")
        print(f"  Delta_sigma:{Ds_sign_changes} out of {len(dDs)-1} steps")
        
        if A_sign_changes > len(dA) // 3:
            print(f"\n  WARNING: A is oscillating heavily ({A_sign_changes} sign changes)")
        if Dx_sign_changes > len(dDx) // 3:
            print(f"  WARNING: Delta_x is oscillating heavily ({Dx_sign_changes} sign changes)")
        if Ds_sign_changes > len(dDs) // 3:
            print(f"  WARNING: Delta_sigma is oscillating heavily ({Ds_sign_changes} sign changes)")
        
        # Check for monotonic drift
        if A_sign_changes < 3:
            if dA[-1] > 0:
                print(f"\n  NOTE: A is monotonically INCREASING (potential runaway)")
            else:
                print(f"\n  NOTE: A is monotonically DECREASING (potential runaway)")
        
        # Check convergence trend in last 20% of iterations
        if n_iter > 10:
            last_20pct = max(10, int(0.2 * n_iter))
            recent_dA = [abs(A_hist[i] - A_hist[i-1]) / max(A_hist[i-1], 0.01) for i in range(n_iter - last_20pct, n_iter)]
            avg_recent_change = np.mean(recent_dA) * 100
            
            print(f"\n  Average relative change in last {last_20pct} iterations: {avg_recent_change:.4f}%")
            if avg_recent_change > 1.0:
                print(f"    STATUS: NOT CONVERGING (changes still > 1%)")
            elif avg_recent_change > 0.1:
                print(f"    STATUS: SLOW CONVERGENCE (changes 0.1-1%)")
            else:
                print(f"    STATUS: CONVERGED (changes < 0.1%)")


def main():
    print("="*80)
    print("OUTER LOOP CONVERGENCE DIAGNOSTIC")
    print("="*80)
    print("\nParameters from constants.json:")
    print("  alpha = 9.45, g_internal = 1e7, g1 = 5000")
    
    # Create solver
    solver = UnifiedSFMSolver(
        n_max=5,
        l_max=2,
        N_sigma=64,
        verbose=False
    )
    
    # Test with different max_iter_outer values
    test_cases = [
        ("50 iterations", 50),
        ("100 iterations", 100),
    ]
    
    for test_name, max_iter in test_cases:
        print(f"\n\n{'#'*80}")
        print(f"# TEST: {test_name}")
        print(f"{'#'*80}")
        
        # Solve muon (generation n=2)
        print(f"\nSolving muon with max_iter_outer={max_iter}...")
        result = solver.solve_lepton(
            winding_k=1,
            generation_n=2,
            max_iter=200,
            max_iter_outer=max_iter,
            tol_outer=1e-3
        )
        
        # Analyze convergence
        analyze_convergence_history(result, f"Muon (max_iter={max_iter})")
        
        # Calculate mass
        m_e_exp = 0.511  # MeV
        # Need electron amplitude to calculate beta
        result_e = solver.solve_lepton(
            winding_k=1,
            generation_n=1,
            max_iter=200,
            max_iter_outer=max_iter,
            tol_outer=1e-3
        )
        beta = m_e_exp / (result_e.A ** 2)
        mass_mu = beta * result.A ** 2
        
        print(f"\n{'='*80}")
        print(f"MASS PREDICTION:")
        print(f"{'='*80}")
        print(f"  Muon mass: {mass_mu:.3f} MeV (experimental: 105.658 MeV)")
        print(f"  Error: {abs(mass_mu - 105.658) / 105.658 * 100:.1f}%")


if __name__ == '__main__':
    main()

