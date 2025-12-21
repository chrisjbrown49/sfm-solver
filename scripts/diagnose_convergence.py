"""
Diagnostic script to check energy minimizer convergence behavior.

Runs electron with verbose output to see actual convergence.
"""

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.particle_configurations import ELECTRON

# Create solver with verbose mode
solver = UnifiedSFMSolver(
    verbose=True  # Enable ALL verbose output
)

print("="*80)
print("CONVERGENCE DIAGNOSTIC: ELECTRON")
print("="*80)
print("\nSolving electron with full verbose output...")
print("This will show every iteration of the energy minimizer.\n")

result = solver.solve_lepton(
    winding_k=1,
    generation_n=1,
    max_iter=200,
    tol=1e-6
)

print("\n" + "="*80)
print("CONVERGENCE SUMMARY")
print("="*80)
print(f"Shape converged: {result.shape_converged} ({result.shape_iterations} iterations)")
print(f"Energy converged: {result.energy_converged} ({result.energy_iterations} iterations)")
print(f"\nFinal parameters:")
print(f"  A = {result.A:.8f}")
print(f"  Delta_x = {result.Delta_x:.8f} fm")
print(f"  Delta_sigma = {result.Delta_sigma:.8f}")
print(f"  E_total = {result.E_total:.8e}")

if not result.energy_converged:
    print("\n" + "!"*80)
    print("WARNING: Energy minimizer did NOT converge!")
    print("!"*80)
    print("\nCheck the iteration output above to see why.")
    print("Common issues:")
    print("  - Tolerance too strict (1e-6 may be unrealistic)")
    print("  - Oscillating parameters (adaptive mixing not working)")
    print("  - Maximum iterations too low (200 may not be enough)")


