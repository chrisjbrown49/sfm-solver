"""
Estimate reasonable parameter values for constants.json based on physical constraints.
"""

import numpy as np

print("="*80)
print("PARAMETER ESTIMATION FOR SFM SOLVER")
print("="*80)

# Physical constants
hbar_c_GeV_fm = 0.1973269804  # GeV*fm
m_electron_GeV = 0.000510999  # GeV
m_muon_GeV = 0.1056583755  # GeV
m_tau_GeV = 1.77686  # GeV

print("\nExperimental masses:")
print(f"  Electron: {m_electron_GeV*1000:.3f} MeV")
print(f"  Muon: {m_muon_GeV*1000:.3f} MeV")
print(f"  Tau: {m_tau_GeV*1000:.3f} MeV")

print(f"\nMass ratios:")
print(f"  mu/e: {m_muon_GeV/m_electron_GeV:.1f}")
print(f"  tau/e: {m_tau_GeV/m_electron_GeV:.1f}")

print("\n" + "="*80)
print("APPROACH 1: Start from reasonable beta")
print("="*80)

# Assume beta ~ 100-1000 GeV (TeV scale feels too high, let's try intermediate)
beta_test = 100.0  # GeV

print(f"\nAssuming beta = {beta_test} GeV")

# Calculate amplitude for electron
A_e = np.sqrt(m_electron_GeV / beta_test)
A_mu = np.sqrt(m_muon_GeV / beta_test)
A_tau = np.sqrt(m_tau_GeV / beta_test)

print(f"\nAmplitudes from m = beta * A^2:")
print(f"  A_electron: {A_e:.6f}")
print(f"  A_muon: {A_mu:.6f}")
print(f"  A_tau: {A_tau:.6f}")

print(f"\nAmplitude ratios:")
print(f"  A_mu/A_e: {A_mu/A_e:.3f} (expected: ~{np.sqrt(206.8):.3f})")
print(f"  A_tau/A_e: {A_tau/A_e:.3f} (expected: ~{np.sqrt(3477):.3f})")

print("\n" + "="*80)
print("APPROACH 2: Estimate g_internal from spatial scales")
print("="*80)

# For electron, expect Delta_x ~ 0.1-1 fm (nuclear scale, around Compton wavelength)
# Compton wavelength = hbar / (m*c) = hbar*c / (m*c^2)
lambda_C_e = hbar_c_GeV_fm / m_electron_GeV
lambda_C_mu = hbar_c_GeV_fm / m_muon_GeV
lambda_C_tau = hbar_c_GeV_fm / m_tau_GeV

print(f"\nCompton wavelengths:")
print(f"  Electron: {lambda_C_e:.3f} fm")
print(f"  Muon: {lambda_C_mu:.6f} fm")
print(f"  Tau: {lambda_C_tau:.6f} fm")

# Target: Delta_x_electron ~ 0.5 fm (between Compton wavelength and nuclear scale)
Delta_x_target_fm = 0.5  # fm

print(f"\nTarget: Delta_x_electron = {Delta_x_target_fm} fm")

# From Delta_x = 1/(g_internal * A^6) [GeV^-1], convert to fm:
# Delta_x [fm] = (1/(g_internal * A^6)) * hbar_c
# g_internal = hbar_c / (Delta_x_fm * A^6)

g_internal_est = hbar_c_GeV_fm / (Delta_x_target_fm * A_e**6)

print(f"\nFrom Delta_x = (1/(g_internal * A^6)) * hbar_c:")
print(f"  g_internal = hbar_c / (Delta_x * A^6)")
print(f"  g_internal = {hbar_c_GeV_fm:.4f} / ({Delta_x_target_fm} * {A_e**6:.6e})")
print(f"  g_internal = {g_internal_est:.6e} GeV^4")

print("\n" + "="*80)
print("APPROACH 3: Try legacy-like value with correction factor")
print("="*80)

# The legacy code had g_internal ~ 0.0001, but with wrong units
# With unit correction, we need to scale by (hbar_c)^2 ~ 0.0389
legacy_g = 0.0001
corrected_g = legacy_g / (hbar_c_GeV_fm**2)

print(f"\nLegacy g_internal: {legacy_g}")
print(f"Unit correction factor: 1/(hbar_c)^2 = {1/(hbar_c_GeV_fm**2):.3f}")
print(f"Corrected g_internal: {corrected_g:.6e} GeV^4")

print("\n" + "="*80)
print("RECOMMENDED PARAMETERS")
print("="*80)

# Let's try a middle ground: order of magnitude between approaches
g_internal_rec = 100.0  # Start with 10^2, can adjust

print(f"\nRecommended starting values:")
print(f"  g_internal: {g_internal_rec} GeV^4")
print(f"  alpha: 10.0 GeV (spatial-subspace coupling)")
print(f"  g1: 50.0 GeV (nonlinear subspace coupling)")
print(f"  g2: 0.004 GeV (EM coupling, keep same)")
print(f"  V0: 1.0 GeV (three-well depth, keep same)")
print(f"  V1: 0.0 GeV (secondary well, keep same)")

print("\nPredicted electron Delta_x with these parameters:")
Delta_x_pred_nat = 1.0 / (g_internal_rec * A_e**6)
Delta_x_pred_fm = Delta_x_pred_nat * hbar_c_GeV_fm
print(f"  Delta_x = {Delta_x_pred_fm:.6f} fm")

print("\nNote: These are order-of-magnitude estimates.")
print("The optimizer will need to refine these values.")

