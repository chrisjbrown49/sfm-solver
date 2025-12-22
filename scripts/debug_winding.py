"""
Debug: What's happening to the winding during shape solver iterations?
"""

import numpy as np
from sfm_solver.core.shape_solver import DimensionlessShapeSolver

# Create shape solver
solver = DimensionlessShapeSolver(
    g1_dimensionless=5000.0,
    g2_dimensionless=0.035,
    V0=1.0,
    V1=0.0,
    N_sigma=64,
    verbose=False
)

print("="*80)
print("DEBUGGING WINDING IN SHAPE SOLVER")
print("="*80)

# Build operators
N = 64
sigma = np.linspace(0, 2*np.pi, N, endpoint=False)
dsigma = 2*np.pi / N

D1 = np.zeros((N, N), dtype=complex)
for i in range(N):
    D1[i, (i+1) % N] = 1.0
    D1[i, (i-1) % N] = -1.0
D1 = D1 / (2 * dsigma)

D2 = D1 @ D1

# Initial guess with winding k=1
winding_k = 1
chi_init = solver._gaussian_in_well(well=1, phase=0.0, winding=winding_k)
chi_init = chi_init / np.sqrt(np.sum(np.abs(chi_init)**2) * dsigma)

print(f"\nInitial wavefunction (k={winding_k}):")
dchi_init = D1 @ chi_init
circ_init = np.sum(np.conj(chi_init) * dchi_init) * dsigma
print(f"  Circulation (complex): {circ_init}")
print(f"  Im(circulation): {np.imag(circ_init):.6f}")

# Extract envelope
phi_init = chi_init / np.exp(1j * winding_k * sigma)
print(f"\nExtracted envelope phi:")
print(f"  max |phi|: {np.max(np.abs(phi_init)):.6f}")
print(f"  Is phi real? max(Im[phi]): {np.max(np.abs(np.imag(phi_init))):.6e}")

# Build effective Hamiltonian as in the shape solver
V_well = np.ones(N) * 1.0  # Simplified
V_em = np.zeros(N)
V_total = V_well + V_em + winding_k**2

H_eff = -D2 - 2j * winding_k * D1 + np.diag(V_total)

print(f"\nEffective Hamiltonian:")
print(f"  Shape: {H_eff.shape}")
print(f"  Is Hermitian? {np.allclose(H_eff, H_eff.conj().T)}")
print(f"  max |Im[H_eff]|: {np.max(np.abs(np.imag(H_eff))):.6e}")

# Solve eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eig(H_eff)

# Check first few eigenvalues
print(f"\nFirst 5 eigenvalues:")
idx_sorted = np.argsort(np.real(eigenvalues))
for i in range(5):
    idx = idx_sorted[i]
    ev = eigenvalues[idx]
    print(f"  {i}: {np.real(ev):.6f} + {np.imag(ev):.6e}i")

# Select ground state
idx_ground = idx_sorted[0]
phi_ground = eigenvectors[:, idx_ground]
phi_ground = phi_ground / np.sqrt(np.sum(np.abs(phi_ground)**2) * dsigma)

print(f"\nGround state envelope:")
print(f"  max |phi|: {np.max(np.abs(phi_ground)):.6f}")
print(f"  max |Im[phi]|: {np.max(np.abs(np.imag(phi_ground))):.6e}")

# Reconstruct full wavefunction
chi_ground = phi_ground * np.exp(1j * winding_k * sigma)
chi_ground = chi_ground / np.sqrt(np.sum(np.abs(chi_ground)**2) * dsigma)

print(f"\nReconstructed wavefunction:")
dchi_ground = D1 @ chi_ground
circ_ground = np.sum(np.conj(chi_ground) * dchi_ground) * dsigma
print(f"  Circulation (complex): {circ_ground}")
print(f"  Im(circulation): {np.imag(circ_ground):.6f}")
print(f"  Expected: ~{winding_k}")

if abs(np.imag(circ_ground)) > 0.5:
    print("\nSUCCESS: Envelope method preserves winding!")
else:
    print("\nPROBLEM: Winding still lost!")
    print("\nPossible cause:")
    print("  The -2ik∂/∂σ term makes H_eff non-Hermitian.")
    print("  np.linalg.eig should handle this, but maybe the eigenvectors")
    print("  are not preserving the winding structure correctly.")

