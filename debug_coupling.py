"""Debug script to understand why coupling factor is zero."""

import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver

# Create solver
solver = NonSeparableWavefunctionSolver(
    alpha=20.0, beta=100.0, kappa=0.0001,
    g1=5000.0, g2=0.004, V0=1.0,
)

# Solve for electron
structure = solver.solve_lepton(n_target=1, k_winding=1, verbose=True)

# Check which states have amplitude
print("\n=== States with amplitude ===")
keys_with_amplitude = []
for key, chi in structure.chi_components.items():
    norm = np.sum(np.abs(chi)**2) * solver.basis.dsigma
    if norm > 1e-10:
        keys_with_amplitude.append(key)
        print(f"  {key}: norm = {norm:.4f}")

# Check spatial coupling between these states
print("\n=== Spatial coupling between states with amplitude ===")
state_index_map = solver.get_state_index_map()
spatial_coupling = solver.get_spatial_coupling_matrix()

for key_i in keys_with_amplitude:
    idx_i = state_index_map.get(key_i)
    for key_j in keys_with_amplitude:
        if key_i == key_j:
            continue
        idx_j = state_index_map.get(key_j)
        
        R_ij = spatial_coupling[idx_i, idx_j]
        if abs(R_ij) > 1e-10:
            print(f"  R[{key_i}, {key_j}] = {R_ij:.6f}")

# Check the subspace integral directly
print("\n=== Subspace integrals between states ===")
D1 = np.zeros((solver.basis.N_sigma, solver.basis.N_sigma), dtype=complex)
dsigma = solver.basis.dsigma
for i in range(solver.basis.N_sigma):
    D1[i, (i+1) % solver.basis.N_sigma] = 1.0
    D1[i, (i-1) % solver.basis.N_sigma] = -1.0
D1 = D1 / (2 * dsigma)

for key_i in keys_with_amplitude:
    for key_j in keys_with_amplitude:
        if key_i == key_j:
            continue
        
        chi_i = structure.chi_components[key_i]
        chi_j = structure.chi_components[key_j]
        dchi_j = D1 @ chi_j
        
        integral = np.sum(np.conj(chi_i) * dchi_j) * dsigma
        if abs(integral) > 1e-10:
            print(f"  Im[int chi_{key_i}* (d chi_{key_j}/ds)] = {np.imag(integral):.6f}")

# Check total wavefunction k_eff
print(f"\n=== Total wavefunction k_eff = {structure.k_eff:.4f} ===")

