"""
Check energy components for all three lepton generations.
"""

import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver

print("="*80)
print("ENERGY COMPONENTS BY GENERATION")
print("="*80)

solver = UnifiedSFMSolver(verbose=False)

for generation_n, name in [(1, "Electron"), (2, "Muon"), (3, "Tau")]:
    print(f"\n{name} (n={generation_n}):")
    print("-" * 40)
    
    result = solver.solve_lepton(winding_k=1, generation_n=generation_n)
    
    E_total = result.energy_components['E_total']
    E_spatial = result.energy_components['E_spatial']
    E_curvature = result.energy_components['E_curvature']
    E_coupling = result.energy_components['E_coupling']
    E_sigma = result.energy_components['E_sigma']
    
    print(f"  A = {result.A:.6f}")
    print(f"  Delta_x = {result.Delta_x:.3f} fm")
    print(f"  E_total     = {E_total:.6e}")
    print(f"  E_spatial   = {E_spatial:.6e}")
    print(f"  E_curvature = {E_curvature:.6e}")
    print(f"  E_coupling  = {E_coupling:.6e}  <-- Key for hierarchy!")
    print(f"  E_sigma     = {E_sigma:.6e}")
    print(f"  |E_coupling/E_total| = {abs(E_coupling/E_total):.6e}")
    
print(f"\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("If E_coupling is negligible (~1e-6 or smaller), then all generations")
print("will have the same energy landscape and identical amplitudes.")
print("We need E_coupling to be O(0.1) or larger to see generation hierarchy.")

