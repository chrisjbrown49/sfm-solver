"""
Quick diagnostic to check if coupling energy is generation-dependent.
"""

from sfm_solver.core.unified_solver import UnifiedSFMSolver

solver = UnifiedSFMSolver(verbose=False)

print("Solving leptons...")
result_e = solver.solve_lepton(winding_k=1, generation_n=1)
result_mu = solver.solve_lepton(winding_k=1, generation_n=2)
result_tau = solver.solve_lepton(winding_k=1, generation_n=3)

print("\n" + "="*80)
print("ENERGY BREAKDOWN COMPARISON")
print("="*80)

print(f"\n{'Component':<20} {'Electron (n=1)':<20} {'Muon (n=2)':<20} {'Tau (n=3)':<20}")
print("-"*80)

for key in ['E_spatial', 'E_curvature', 'E_coupling', 'E_sigma', 'E_total']:
    e_val = result_e.energy_components[key]
    mu_val = result_mu.energy_components[key]
    tau_val = result_tau.energy_components[key]
    print(f"{key:<20} {e_val:<20.6e} {mu_val:<20.6e} {tau_val:<20.6e}")

print("\n" + "="*80)
print("COUPLING ENERGY RATIOS (should be different!)")
print("="*80)
E_c_e = result_e.energy_components['E_coupling']
E_c_mu = result_mu.energy_components['E_coupling']
E_c_tau = result_tau.energy_components['E_coupling']

print(f"E_coupling(mu) / E_coupling(e)  = {E_c_mu/E_c_e:.4f}  (should be ~3.9 from spatial_factor)")
print(f"E_coupling(tau) / E_coupling(e) = {E_c_tau/E_c_e:.4f}  (should be ~8.6 from spatial_factor)")

print("\n" + "="*80)
print("AMPLITUDES")
print("="*80)
print(f"A_e   = {result_e.A:.6f}")
print(f"A_mu  = {result_mu.A:.6f}  (ratio: {result_mu.A/result_e.A:.4f})")
print(f"A_tau = {result_tau.A:.6f}  (ratio: {result_tau.A/result_e.A:.4f})")

