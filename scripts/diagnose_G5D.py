"""
Diagnose what's happening with the G_5D implementation.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.unified_solver import UnifiedSFMSolver

print("=" * 80)
print("DIAGNOSING G_5D IMPLEMENTATION")
print("=" * 80)
print()

# Test a few representative values
test_values = [1e-10, 1e-5, 1e0, 1e5, 1e7, 1e10]

for G5D in test_values:
    print(f"Testing G_5D = {G5D:.1e} GeV^-2...")
    print("-" * 80)
    
    try:
        solver = UnifiedSFMSolver(G_5D=G5D, verbose=True)
        print("  Solver created successfully")
        print()
        
        print("  Solving electron...")
        result = solver.solve_lepton(
            winding_k=1,
            generation_n=1,
            max_iter=200,
            tol=1e-6,
            max_iter_outer=5,  # Just a few iterations for testing
            tol_outer=1e-3
        )
        
        print()
        print(f"  Result: A={result.A:.6f}, Dx={result.Delta_x:.3f} fm, converged={result.outer_converged}")
        print(f"  Energy components:")
        print(f"    E_spatial:   {result.E_spatial:.6e} GeV")
        print(f"    E_curvature: {result.E_curvature:.6e} GeV")
        print(f"    E_total:     {result.E_total:.6e} GeV")
        print()
        
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        print()
        print("  Traceback:")
        traceback.print_exc()
        print()
    
    print()
    print()

print("=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)


