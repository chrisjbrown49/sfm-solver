"""
Test shape solver directly to see circulation at each iteration.
"""

import numpy as np
from sfm_solver.core.shape_solver import DimensionlessShapeSolver

print("="*80)
print("TESTING SHAPE SOLVER CIRCULATION")
print("="*80)

solver = DimensionlessShapeSolver(
    g1_dimensionless=5000.0,
    g2_dimensionless=0.035,
    V0=1.0,
    V1=0.0,
    N_sigma=64,
    verbose=True  # Enable verbose to see iteration details
)

# Solve for electron
result = solver.solve_lepton_shape(winding_k=1, generation_n=1, max_iter=200, tol=1e-6)

# Check final circulation
chi = result.composite_shape
N = len(chi)
dsigma = 2 * np.pi / N

D1 = np.zeros((N, N), dtype=complex)
for i in range(N):
    D1[i, (i+1) % N] = 1.0
    D1[i, (i-1) % N] = -1.0
D1 = D1 / (2 * dsigma)

dchi = D1 @ chi
circulation = np.sum(np.conj(chi) * dchi) * dsigma

print(f"\n" + "="*80)
print(f"FINAL RESULT")
print(f"="*80)
print(f"Normalization: {np.sum(np.abs(chi)**2) * dsigma:.6f}")
print(f"Circulation (complex): {circulation}")
print(f"Im(circulation): {np.imag(circulation):.6f} (expected: ~1.0)")

