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
    tol=1e-6,
    max_iter_outer=30,  # Outer loop iterations
    tol_outer=1e-4  # Outer loop tolerance (legacy value)
)

print("\n" + "="*80)
print("CONVERGENCE SUMMARY")
print("="*80)
print(f"Shape converged: {result.shape_converged} ({result.shape_iterations} iterations)")
print(f"Energy converged: {result.energy_converged} ({result.energy_iterations} iterations)")
print(f"Outer loop: {result.outer_iterations} iterations, converged: {result.outer_converged}")
print(f"\nFinal parameters:")
print(f"  A = {result.A:.8f}")
print(f"  Delta_x = {result.Delta_x:.8f} fm")
print(f"  Delta_sigma = {result.Delta_sigma:.8f}")
print(f"  E_total = {result.E_total:.8e}")

if result.scale_history and len(result.scale_history['A']) > 1:
    print(f"\nScale evolution:")
    for i in range(min(10, len(result.scale_history['A']))):
        A = result.scale_history['A'][i]
        Dx = result.scale_history['Delta_x'][i]
        Ds = result.scale_history['Delta_sigma'][i]
        print(f"  Iter {i}: A={A:.6f}, Dx={Dx:.6f} fm, Ds={Ds:.6f}")

if not result.outer_converged:
    print("\n" + "!"*80)
    print("WARNING: Outer loop did NOT converge!")
    print("!"*80)
    print("\nCheck the scale evolution above to see why.")
    print("Common issues:")
    print("  - Tolerance too strict (1e-4 may need relaxing)")
    print("  - Oscillating scales (adaptive mixing not sufficient)")
    print("  - Maximum iterations too low (30 may not be enough)")
elif not result.energy_converged:
    print("\n" + "!"*80)
    print("WARNING: Energy minimizer did NOT converge (but outer loop did)!")
    print("!"*80)
    print("The outer loop converged, meaning scales are stable,")
    print("but Stage 2 energy minimization didn't converge within its iteration limit.")


