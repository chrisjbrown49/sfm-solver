"""Debug script to trace electron perturbative calculation."""
import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.universal_energy_minimizer import UniversalEnergyMinimizer
from sfm_solver.core.sfm_global import SFM_CONSTANTS

print("=" * 70)
print("ELECTRON PERTURBATIVE CALCULATION DIAGNOSTIC")
print("=" * 70)

# Get constants
constants = SFM_CONSTANTS
alpha = constants.alpha_coupling_base
beta = constants.beta_physical
kappa = constants.kappa_physical
g1 = constants.g1
g2 = constants.g2
V0 = constants.V0_physical

print(f"\nFundamental parameters:")
print(f"  alpha = {alpha:.6f} GeV")
print(f"  beta  = {beta:.6f} GeV")
print(f"  kappa = {kappa:.6f} GeV^-2")
print(f"  g1    = {g1:.6f}")
print(f"  V0    = {V0:.6f} GeV")

# Create wavefunction solver
wf_solver = NonSeparableWavefunctionSolver(
    alpha=alpha,
    beta=beta,
    kappa=kappa,
    g1=g1,
    g2=g2,
    V0=V0,
    n_max=5,
    l_max=2,
    N_sigma=64,
)

print(f"\n{'='*70}")
print("STEP 1: Spatial Basis States")
print("=" * 70)

for i, state in enumerate(wf_solver.basis.spatial_states):
    print(f"  State {i}: (n={state.n}, l={state.l}, m={state.m})")

print(f"\n{'='*70}")
print("STEP 2: Spatial Coupling Matrix for Electron (n=1, l=0)")
print("=" * 70)

n_target = 1
k_winding = 1
target_idx = wf_solver.basis.state_index(n_target, 0, 0)
E_target_0 = n_target ** 2

print(f"\nTarget state: (n={n_target}, l=0, m=0), index={target_idx}")
print(f"Target energy E_0 = n^2 = {E_target_0}")
print(f"\nCouplings to other states:")
print(f"{'State':<20} {'R_coupling':<15} {'E_state':<10} {'E_denom':<12} {'|R|/|E_d|':<12}")
print("-" * 70)

for i, state in enumerate(wf_solver.basis.spatial_states):
    if state.n == n_target and state.l == 0:
        continue  # Skip target state
    
    R_coupling = wf_solver.spatial_coupling[target_idx, i]
    E_state = state.n ** 2 + state.l * (state.l + 1) / 2
    E_denom = E_target_0 - E_state
    
    # Apply clamping as in the code
    E_denom_clamped = E_denom
    if abs(E_denom) < 0.5:
        E_denom_clamped = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
    
    ratio = abs(R_coupling) / abs(E_denom_clamped) if abs(E_denom_clamped) > 1e-10 else 0
    
    if abs(R_coupling) > 1e-10:
        print(f"({state.n},{state.l},{state.m}){'':<13} {R_coupling:>+12.6f}   {E_state:<10.2f} {E_denom_clamped:<12.2f} {ratio:<12.6f}")

print(f"\n{'='*70}")
print("STEP 3: Induced Components (Natural Perturbative Amplitudes)")
print("=" * 70)

# Manually trace through the solve_lepton logic
sigma = wf_solver.basis.sigma
dsigma = wf_solver.basis.dsigma
D1 = wf_solver._build_subspace_derivative_matrix()

# Primary s-wave
envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
chi_primary = envelope * np.exp(1j * k_winding * sigma)
norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
chi_primary = chi_primary / np.sqrt(norm_sq)

print(f"\nPrimary (n=1,l=0,m=0):")
print(f"  Winding k = {k_winding}")
print(f"  |chi_primary|^2 integral = {np.sum(np.abs(chi_primary)**2) * dsigma:.6f}")

k_induced = k_winding + 1
print(f"\nInduced states will have shifted winding k' = {k_induced}")

chi_components = {(n_target, 0, 0): chi_primary}

print(f"\nInduced component amplitudes:")
print(f"{'State':<20} {'R_coupling':<15} {'E_denom':<12} {'|induced|^2':<15} {'Sign of induced'}")
print("-" * 80)

for i, state in enumerate(wf_solver.basis.spatial_states):
    key = (state.n, state.l, state.m)
    if key == (n_target, 0, 0):
        continue
    
    R_coupling = wf_solver.spatial_coupling[target_idx, i]
    if abs(R_coupling) < 1e-10:
        chi_components[key] = np.zeros(wf_solver.basis.N_sigma, dtype=complex)
        continue
    
    E_state = state.n ** 2 + state.l * (state.l + 1) / 2
    E_denom = E_target_0 - E_state
    
    if abs(E_denom) < 0.5:
        E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
    
    # Induced component
    envelope_primary = np.exp(-(sigma - np.pi)**2 / 0.5)
    induced = -alpha * R_coupling * envelope_primary * np.exp(1j * k_induced * sigma) / E_denom
    chi_components[key] = induced
    
    induced_amp_sq = np.sum(np.abs(induced)**2) * dsigma
    
    # Sign analysis
    coeff = -alpha * R_coupling / E_denom
    sign_str = "+" if coeff > 0 else "-"
    
    print(f"({state.n},{state.l},{state.m}){'':<13} {R_coupling:>+12.6f}   {E_denom:<12.2f} {induced_amp_sq:<15.6e} {sign_str} (coeff={coeff:.6f})")

print(f"\n{'='*70}")
print("STEP 4: L-Composition Before Normalization")
print("=" * 70)

l_comp_raw = {}
total_amp = 0.0
for key, chi in chi_components.items():
    amp_sq = np.sum(np.abs(chi)**2) * dsigma
    total_amp += amp_sq
    l = key[1]
    l_comp_raw[l] = l_comp_raw.get(l, 0) + amp_sq

print(f"\nTotal amplitude^2 (before norm): {total_amp:.6e}")
print(f"\nL-composition (before norm):")
for l, amp in sorted(l_comp_raw.items()):
    print(f"  l={l}: {amp:.6e} ({100*amp/total_amp:.2f}%)")

print(f"\n{'='*70}")
print("STEP 5: Cross-Term Integrals Im[chi_i* d(chi_j)/dsigma]")
print("=" * 70)

print(f"\nComputing Im[int chi_i* (d chi_j/dsigma) dsigma] for state pairs:")
print(f"{'(i,j)':<30} {'Re[integral]':<15} {'Im[integral]':<15} {'R_ij':<12} {'R*Im':<12}")
print("-" * 85)

keys = list(chi_components.keys())
for key_i in keys:
    chi_i = chi_components[key_i]
    norm_i = np.sum(np.abs(chi_i)**2) * dsigma
    if norm_i < 1e-15:
        continue
    
    idx_i = wf_solver.basis.state_index(*key_i)
    
    for key_j in keys:
        if key_i == key_j:
            continue
        
        chi_j = chi_components[key_j]
        norm_j = np.sum(np.abs(chi_j)**2) * dsigma
        if norm_j < 1e-15:
            continue
        
        idx_j = wf_solver.basis.state_index(*key_j)
        R_ij = wf_solver.spatial_coupling[idx_i, idx_j]
        
        dchi_j = D1 @ chi_j
        integral = np.sum(np.conj(chi_i) * dchi_j) * dsigma
        
        contribution = np.real(R_ij) * np.imag(integral)
        
        if abs(np.imag(integral)) > 1e-10 or abs(R_ij) > 1e-10:
            print(f"{str(key_i):<15} -> {str(key_j):<12} {np.real(integral):<15.6f} {np.imag(integral):<15.6f} {R_ij:<12.6f} {contribution:<12.6f}")

print(f"\n{'='*70}")
print("STEP 6: Total Coupling Factor")
print("=" * 70)

# Normalize first (as the solver does)
total_norm = sum(np.sum(np.abs(chi)**2) * dsigma for chi in chi_components.values())
for key in chi_components:
    chi_components[key] = chi_components[key] / np.sqrt(total_norm)

coupling_factor = 0.0
for key_i in keys:
    chi_i = chi_components[key_i]
    norm_i = np.sum(np.abs(chi_i)**2) * dsigma
    if norm_i < 1e-15:
        continue
    
    idx_i = wf_solver.basis.state_index(*key_i)
    
    for key_j in keys:
        if key_i == key_j:
            continue
        
        chi_j = chi_components[key_j]
        norm_j = np.sum(np.abs(chi_j)**2) * dsigma
        if norm_j < 1e-15:
            continue
        
        idx_j = wf_solver.basis.state_index(*key_j)
        R_ij = wf_solver.spatial_coupling[idx_i, idx_j]
        
        dchi_j = D1 @ chi_j
        integral = np.sum(np.conj(chi_i) * dchi_j) * dsigma
        
        contribution = np.real(R_ij) * np.imag(integral)
        coupling_factor += contribution

print(f"\nTotal coupling_factor = {coupling_factor:.6f}")
print(f"E_coupling = -alpha * coupling_factor * A")
print(f"           = -{alpha:.6f} * {coupling_factor:.6f} * A")
print(f"           = {-alpha * coupling_factor:.6f} * A")

if coupling_factor > 0:
    print(f"\n*** E_coupling will be NEGATIVE (attractive) - CORRECT! ***")
else:
    print(f"\n*** E_coupling will be POSITIVE (repulsive) - WRONG! ***")

print(f"\n{'='*70}")
print("COMPARISON: Run the actual solver")
print("=" * 70)

# Now run the actual solver for comparison
structure = wf_solver.solve_lepton(n_target=1, k_winding=1, verbose=True)

print(f"\nSolver output:")
print(f"  k_eff = {structure.k_eff:.6f}")
print(f"  l_composition = {structure.l_composition}")
print(f"  converged = {structure.converged}")

# And the energy minimizer
minimizer = UniversalEnergyMinimizer(
    alpha=alpha, beta=beta, kappa=kappa, g1=g1, g2=g2, V0=V0
)

V_sigma = V0 * (1 - np.cos(3 * sigma))

result = minimizer.minimize(
    structure=structure,
    sigma_grid=sigma,
    V_sigma=V_sigma,
    spatial_coupling=wf_solver.spatial_coupling,
    state_index_map={state: i for i, state in enumerate(
        [(s.n, s.l, s.m) for s in wf_solver.basis.spatial_states]
    )},
    verbose=True,
)

print(f"\nMinimizer result:")
print(f"  A^2 = {result.amplitude_squared:.6e}")
print(f"  Mass = {result.predicted_mass_gev * 1000:.4f} MeV")
print(f"  E_coupling = {result.energy_coupling:.6f}")

print(f"\n{'='*70}")
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

