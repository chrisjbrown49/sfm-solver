"""
Debug: Check the actual values of spatial_factor and subspace_factor.
"""

import numpy as np
from scipy.special import genlaguerre

HBAR_C_GEV_FM = 0.197327  # GeV·fm

# Manually compute spatial_factor for n=1
n = 1
delta_x = 4.5  # fm (from results)

a = delta_x / np.sqrt(2 * n + 1)  # Scale parameter
r_grid = np.linspace(0.01, 20.0 * a, 500)
r = r_grid
x = (r / a) ** 2

# Laguerre polynomial for n=1 (k=0, alpha=0.5)
k = n - 1  # 0
alpha_lag = 0.5
L = genlaguerre(k, alpha_lag)(x)  # L_0^{1/2} = 1
if k >= 1:
    dL_dx = -genlaguerre(k - 1, alpha_lag + 1)(x)
else:
    dL_dx = np.zeros_like(x)

# Radial wavefunction
exp_factor = np.exp(-x / 2)
phi = L * exp_factor

# Derivative
dphi_dx = exp_factor * (dL_dx - L / 2)
dphi_dr = dphi_dx * (2 * r / a**2)

# Normalize
norm_sq = 4 * np.pi * np.trapz(phi**2 * r**2, r)
phi = phi / np.sqrt(norm_sq)
dphi_dr = dphi_dr / np.sqrt(norm_sq)

# Spatial factor
spatial_factor = 4 * np.pi * np.trapz(dphi_dr**2 * r**2, r)

print("="*80)
print("COUPLING FACTOR DIAGNOSTIC")
print("="*80)
print(f"\nSpatial factor calculation:")
print(f"  n = {n}")
print(f"  Delta_x = {delta_x} fm")
print(f"  a = {a:.6f} fm")
print(f"  spatial_factor = integral|grad phi|^2 d^3x = {spatial_factor:.6f}")

# Now check what the energy minimizer would compute
print(f"\nEnergy minimizer calculation:")
print(f"  alpha = 10.5 GeV")
print(f"  subspace_factor (circulation) ~ 1.0")
print(f"  A ~ 0.022")
print(f"  Expected E_coupling = -alpha × spatial_factor × circ × A")
print(f"                      = -10.5 × {spatial_factor:.4f} × 1.0 × 0.022")
print(f"                      = {-10.5 * spatial_factor * 1.0 * 0.022:.6f} GeV")

# But we're getting E_coupling ~ -1.2e-5 GeV
# That's a factor of:
actual = -1.2e-5
expected = -10.5 * spatial_factor * 1.0 * 0.022
print(f"\nActual E_coupling from solver: {actual:.6e} GeV")
print(f"Discrepancy factor: {expected/actual:.1f}×")

print(f"\nPossible causes:")
print(f"  1. Unit mismatch (spatial_factor in wrong units?)")
print(f"  2. Missing factor in coupling energy formula")
print(f"  3. Subspace_factor is actually much smaller than 1.0")
print(f"  4. Alpha needs different units/normalization")

