"""
Check if circulation is preserved after scaling in the energy minimizer.
"""

import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver

print("="*80)
print("CHECKING CIRCULATION AFTER SCALING")
print("="*80)

solver = UnifiedSFMSolver(verbose=False)

# Solve electron
result = solver.solve_lepton(winding_k=1, generation_n=1)

# Get the 4D structure (before scaling)
shape_result = solver.shape_solver.solve_lepton_shape(winding_k=1, generation_n=1)
structure_4d = solver.spatial_coupling.build_4d_structure(
    subspace_shape=shape_result.composite_shape,
    n_target=1,
    l_target=0,
    m_target=0
)

N = 64
dsigma = 2 * np.pi / N

# Build derivative operator
D1 = np.zeros((N, N), dtype=complex)
for i in range(N):
    D1[i, (i+1) % N] = 1.0
    D1[i, (i-1) % N] = -1.0
D1 = D1 / (2 * dsigma)

# Check circulation before scaling
chi_total_unscaled = sum(structure_4d.values())
norm_unscaled = np.sqrt(np.sum(np.abs(chi_total_unscaled)**2) * dsigma)
dchi_unscaled = D1 @ chi_total_unscaled
circ_unscaled = np.sum(np.conj(chi_total_unscaled) * dchi_unscaled) * dsigma

print(f"\nBefore scaling:")
print(f"  Norm: {norm_unscaled:.6f}")
print(f"  Im(circulation): {np.imag(circ_unscaled):.6f}")

# Now scale using energy minimizer's scaling
A = result.A
Delta_sigma = result.Delta_sigma
scaling_factor = A / np.sqrt(Delta_sigma)

print(f"\nScaling parameters:")
print(f"  A = {A:.6f}")
print(f"  Delta_sigma = {Delta_sigma:.6f}")
print(f"  scaling_factor = A/sqrt(Delta_sigma) = {scaling_factor:.6f}")

chi_scaled = {}
for key, chi in structure_4d.items():
    chi_scaled[key] = scaling_factor * chi

# Check circulation after scaling
chi_total_scaled = sum(chi_scaled.values())
norm_scaled = np.sqrt(np.sum(np.abs(chi_total_scaled)**2) * dsigma)
dchi_scaled = D1 @ chi_total_scaled
circ_scaled = np.sum(np.conj(chi_total_scaled) * dchi_scaled) * dsigma

print(f"\nAfter scaling:")
print(f"  Norm: {norm_scaled:.6f}")
print(f"  Im(circulation): {np.imag(circ_scaled):.6f}")
print(f"  Expected: norm ~ A = {A:.6f}")

# The circulation should scale as: circ_scaled = scaling_factor × circ_unscaled
circ_expected = scaling_factor * np.imag(circ_unscaled)
circ_actual = np.imag(circ_scaled)

print(f"\nCirculation scaling:")
print(f"  Expected: {circ_expected:.6f} (= scaling_factor × circ_unscaled)")
print(f"  Actual: {circ_actual:.6f}")
print(f"  Match: {abs(circ_actual - circ_expected) < 1e-6}")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)
if abs(circ_actual) > 0.01:
    print(f"Circulation after scaling: {circ_actual:.6f}")
    print(f"This is the 'subspace_factor' used in E_coupling calculation.")
    print(f"With spatial_factor ~ 4.5, alpha = 10.5, A ~ 0.022:")
    print(f"  E_coupling ~ -10.5 × 4.5 × {circ_actual:.4f} × 0.022 = {-10.5 * 4.5 * circ_actual * 0.022:.6f} GeV")
else:
    print(f"PROBLEM: Circulation after scaling is too small: {circ_actual:.6e}")
    print(f"This is why E_coupling is negligible!")

