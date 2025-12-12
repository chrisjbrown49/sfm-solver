"""Test that coupling factor is now non-zero."""
import numpy as np
from sfm_solver.eigensolver.sfm_lepton_solver import SFMLeptonSolver

solver = SFMLeptonSolver()

print("=" * 60)
print("TESTING COUPLING FIX")
print("=" * 60)

for particle in ['electron', 'muon', 'tau']:
    print(f"\n{particle.upper()}:")
    result = solver.solve_lepton(particle=particle, verbose=False)
    mass_mev = solver.beta * result.amplitude_squared * 1000
    print(f"  n_spatial = {result.n_spatial}")
    print(f"  A^2 = {result.amplitude_squared:.6e}")
    print(f"  Mass = {mass_mev:.4f} MeV")
    print(f"  E_coupling = {result.energy_coupling:.6f}")
    print(f"  k_eff = {result.k_eff:.4f}")
    print(f"  Converged: {result.converged}")

print("\n" + "=" * 60)
print("Expected: E_coupling should be non-zero and negative")
print("Expected: Masses should differ (mass hierarchy)")
print("=" * 60)

