"""
Check if circulation survives 4D structure building.
"""

import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver

print("="*80)
print("CHECKING CIRCULATION THROUGH STAGE 1 AND 4D STRUCTURE")
print("="*80)

solver = UnifiedSFMSolver(verbose=False)

# Get shape from Stage 1
shape_result = solver.shape_solver.solve_lepton_shape(winding_k=1, generation_n=1)
chi_shape = shape_result.composite_shape

# Check circulation of Stage 1 output
N = len(chi_shape)
dsigma = 2 * np.pi / N

D1 = np.zeros((N, N), dtype=complex)
for i in range(N):
    D1[i, (i+1) % N] = 1.0
    D1[i, (i-1) % N] = -1.0
D1 = D1 / (2 * dsigma)

dchi_shape = D1 @ chi_shape
circ_shape = np.sum(np.conj(chi_shape) * dchi_shape) * dsigma

print(f"\nStage 1 output:")
print(f"  Norm: {np.sum(np.abs(chi_shape)**2) * dsigma:.6f}")
print(f"  Im(circulation): {np.imag(circ_shape):.6f}")

# Build 4D structure
structure_4d = solver.spatial_coupling.build_4d_structure(
    subspace_shape=chi_shape,
    n_target=1,
    l_target=0,
    m_target=0
)

print(f"\n4D Structure:")
print(f"  Number of components: {len(structure_4d)}")

# Check circulation of each component
print(f"\n  Component circulations:")
for (n, l, m), chi_nlm in structure_4d.items():
    norm = np.sum(np.abs(chi_nlm)**2) * dsigma
    dchi_nlm = D1 @ chi_nlm
    circ_nlm = np.sum(np.conj(chi_nlm) * dchi_nlm) * dsigma
    if norm > 1e-10:
        print(f"    ({n},{l},{m}): norm={norm:.6f}, Im(circ)={np.imag(circ_nlm):.6e}")

# Check total circulation
chi_total = sum(structure_4d.values())
norm_total = np.sum(np.abs(chi_total)**2) * dsigma
dchi_total = D1 @ chi_total
circ_total = np.sum(np.conj(chi_total) * dchi_total) * dsigma

print(f"\n  Total 4D structure:")
print(f"    Norm: {norm_total:.6f}")
print(f"    Im(circulation): {np.imag(circ_total):.6e}")

print(f"\n" + "="*80)
if abs(np.imag(circ_total)) > 0.5:
    print("SUCCESS: Circulation survives 4D structure building!")
else:
    print("PROBLEM: Circulation lost during 4D structure building!")
    print(f"  Stage 1 had Im(circ) = {np.imag(circ_shape):.6f}")
    print(f"  4D structure has Im(circ) = {np.imag(circ_total):.6e}")
    print(f"  Loss factor: {abs(np.imag(circ_total)/np.imag(circ_shape)):.6e}")

