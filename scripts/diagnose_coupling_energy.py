"""
Diagnostic: Compute coupling energy directly from the Hamiltonian operator.

From Math Formulation Part A, Section 2, component 7:
    H_coupling = -α (∂²/∂x∂σ + ∂²/∂y∂σ + ∂²/∂z∂σ)

This script:
1. Builds the full 5D wavefunction Ψ(r, σ) = φ_n(r) × χ(σ)
2. Computes the coupling energy from the Hamiltonian: E = ⟨Ψ|H_coupling|Ψ⟩
3. Compares to the current analytical formula in the solver
"""

import numpy as np
from scipy.special import genlaguerre
from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.constants import load_constants_from_json

print("="*80)
print("COUPLING ENERGY DIAGNOSTIC: Hamiltonian vs Analytical Formula")
print("="*80)
print()

# Load constants
constants = load_constants_from_json()
alpha = constants['alpha']
print(f"Parameters: alpha={alpha:.2f} GeV, G_5D={constants['G_5D']:.2e}, g1={constants['g1']:.0f}")
print()

# Create solver
solver = UnifiedSFMSolver(verbose=False)

# Test parameters
generations = [1, 2, 3]
Delta_x = 10.0  # fm
Delta_sigma = 1.0
A = 1.0

print(f"Test configuration: Delta_x={Delta_x:.1f} fm, Delta_sigma={Delta_sigma:.1f}, A={A:.1f}")
print()

# Define grids
N_r = 100  # Radial grid points
N_sigma = 64  # Subspace grid points

r_max = 50.0  # fm (should be larger than Delta_x)
r_grid = np.linspace(0.01, r_max, N_r)
sigma_grid = np.linspace(0, 2*np.pi, N_sigma, endpoint=False)
dr = r_grid[1] - r_grid[0]
dsigma = 2*np.pi / N_sigma

print(f"Numerical grids: N_r={N_r}, N_sigma={N_sigma}")
print()

for generation_n in generations:
    print(f"{'='*80}")
    print(f"Generation n={generation_n} ({'Electron' if generation_n==1 else 'Muon' if generation_n==2 else 'Tau'})")
    print(f"{'='*80}")
    
    # =========================================================================
    # PART 1: Build spatial wavefunction φ_n(r)
    # =========================================================================
    
    # Harmonic oscillator scale parameter
    a = Delta_x / np.sqrt(2 * generation_n + 1)  # fm
    
    # Build φ_n using Laguerre polynomials (s-wave, l=0)
    k = generation_n - 1  # Polynomial degree
    alpha_lag = 0.5
    x = (r_grid / a) ** 2
    
    if k >= 0:
        L = genlaguerre(k, alpha_lag)(x)
    else:
        L = np.ones_like(x)
    
    phi_unnorm = L * np.exp(-x / 2)
    
    # Normalize: ∫ 4π φ² r² dr = 1
    norm_sq = 4 * np.pi * np.trapz(phi_unnorm**2 * r_grid**2, r_grid)
    phi = phi_unnorm / np.sqrt(norm_sq)
    
    # Compute radial derivative dφ/dr using finite differences
    dphi_dr = np.gradient(phi, r_grid)
    
    print(f"\nSpatial wavefunction phi_{generation_n}(r):")
    print(f"  Scale parameter a = {a:.3f} fm")
    print(f"  Normalization check: integral phi^2 4*pi*r^2 dr = {4*np.pi*np.trapz(phi**2 * r_grid**2, r_grid):.6f}")
    
    # =========================================================================
    # PART 2: Build subspace wavefunction χ(σ)
    # =========================================================================
    
    # Solve for subspace shape (dimensionless)
    shape_result = solver.shape_solver.solve_lepton_shape(
        winding_k=1,
        generation_n=generation_n,
        max_iter=200,
        tol=1e-6
    )
    
    # Build 4D structure
    structure_4d = solver.spatial_coupling.build_4d_structure(
        subspace_shape=shape_result.composite_shape,
        n_target=generation_n,
        l_target=0,
        m_target=0,
        Delta_x=Delta_x
    )
    
    # Extract χ(σ) and scale it
    chi_shape = shape_result.composite_shape
    
    # Scale: χ_scaled = (A/√Δσ) × χ_shape
    scaling_factor = A / np.sqrt(Delta_sigma)
    chi = scaling_factor * chi_shape
    
    # Compute ∂χ/∂σ using finite differences
    dchi_dsigma = np.gradient(chi, dsigma)
    
    # Compute circulation (for reference)
    circulation = np.sum(np.conj(chi) * dchi_dsigma) * dsigma
    
    print(f"\nSubspace wavefunction chi(sigma):")
    print(f"  Normalization: integral |chi|^2 dsigma = {np.sum(np.abs(chi)**2)*dsigma:.6f} (should be A^2 = {A**2:.2f})")
    print(f"  Circulation: Im[integral chi* d(chi)/dsigma dsigma] = {np.imag(circulation):.6f}")
    
    # =========================================================================
    # PART 3: Build full 5D wavefunction Ψ(r, σ) = φ(r) × χ(σ)
    # =========================================================================
    
    # Create 2D grid: (N_r, N_sigma)
    Psi = np.outer(phi, chi)  # Shape: (N_r, N_sigma)
    
    print(f"\nFull 5D wavefunction Psi(r,sigma):")
    print(f"  Grid shape: {Psi.shape}")
    print(f"  Total norm: integral |Psi|^2 4*pi*r^2 dr dsigma = {4*np.pi*np.sum(np.abs(Psi)**2 * r_grid[:, np.newaxis]**2)*dr*dsigma:.6f}")
    
    # =========================================================================
    # PART 4: Compute coupling energy from Hamiltonian operator
    # =========================================================================
    
    # H_coupling = -alpha (d2/dxdsigma + d2/dydsigma + d2/dzdsigma)
    # For s-wave (spherical symmetry), convert to radial: grad = (d/dr) r_hat
    # The mixed derivative: d2Psi/dxdsigma for spherical symmetry
    
    # Using integration by parts on the expectation value:
    # <Psi|H_coupling|Psi> = -alpha integral Psi* [d2Psi/dxdsigma + ...] d3r dsigma
    #                       = alpha integral [dPsi*/dx][dPsi/dsigma] + ... d3r dsigma
    #                       = alpha integral (gradPsi*)·(dPsi/dsigma) d3r dsigma
    
    # For spherical symmetry with s-wave:
    # (gradPsi) in 3D = (dPsi/dr) r_hat
    # The operator sums over x,y,z derivatives, but for s-wave these contribute equally
    
    # Compute dPsi/dr (varies with r, constant in sigma)
    dPsi_dr = np.outer(dphi_dr, chi)
    
    # Compute dPsi/dsigma (varies with sigma, constant in r)
    dPsi_dsigma = np.outer(phi, dchi_dsigma)
    
    # METHOD A: Direct computation using squared gradients
    # For spherically symmetric s-wave, the spatial gradient is radial only
    # E_coupling = alpha integral |dPsi/dr|^2 |dPsi/dsigma|^2 r^2 dr dsigma
    
    # Actually, the correct formula after integration by parts is:
    # E_coupling = alpha integral (dPsi*/dr)(dPsi/dsigma) + ... d3r dsigma
    # For complex Psi, this is: alpha Re[integral (dPsi*/dr)(dPsi/dsigma) 4*pi*r^2 dr dsigma]
    
    # Compute the integrand (for radial component only, multiply by 3 for x,y,z)
    # Actually, for s-wave: integral |gradPsi|^2 d3x = 4*pi integral (dPsi/dr)^2 r^2 dr
    
    # REVISED: The coupling involves CROSS TERMS between spatial and subspace derivatives
    # Let's compute different possible interpretations:
    
    # Interpretation 1: Factorized formula (current solver implementation)
    # E_coupling = -alpha * [integral |grad_phi|^2 d3r] * [Im(integral chi* d(chi)/dsigma dsigma)] * A
    spatial_factor_current = 4 * np.pi * np.trapz(dphi_dr**2 * r_grid**2, r_grid)  # 1/fm^2
    spatial_factor_dimensionless = spatial_factor_current * (Delta_x ** 2)  # Make dimensionless
    subspace_factor = np.imag(circulation) * (Delta_sigma / A)  # Current formula correction
    E_coupling_analytical = -alpha * spatial_factor_dimensionless * subspace_factor * A
    
    print(f"\nCurrent Analytical Formula:")
    print(f"  spatial_factor = {spatial_factor_dimensionless:.6f} (dimensionless)")
    print(f"  subspace_factor = {subspace_factor:.6f}")
    print(f"  E_coupling = {E_coupling_analytical:.6e} GeV")
    
    # Interpretation 2: Direct from mixed derivatives
    # Compute <Psi|d2/(drdsigma)|Psi> directly
    # Using product rule: d2(phi*chi)/(drdsigma) = (dphi/dr)(dchi/dsigma)
    # E = -alpha integral Psi* [(dphi/dr)(dchi/dsigma)] 4*pi*r^2 dr dsigma
    
    mixed_term = np.outer(phi * dphi_dr, np.conj(chi) * dchi_dsigma)
    E_coupling_direct_v1 = -alpha * 4 * np.pi * np.sum(mixed_term * r_grid[:, np.newaxis]**2) * dr * dsigma
    
    print(f"\nDirect Hamiltonian Computation (Method 1 - no integration by parts):")
    print(f"  E_coupling = {E_coupling_direct_v1:.6e} GeV")
    
    # Interpretation 3: After integration by parts
    # E = alpha integral (dPsi*/dr)(dPsi/dsigma) 4*pi*r^2 dr dsigma
    integrand_ibp = np.conj(dPsi_dr) * dPsi_dsigma
    E_coupling_direct_v2 = alpha * 4 * np.pi * np.sum(integrand_ibp * r_grid[:, np.newaxis]**2) * dr * dsigma
    
    print(f"\nDirect Hamiltonian Computation (Method 2 - after integration by parts):")
    print(f"  E_coupling = {np.real(E_coupling_direct_v2):.6e} GeV (real part)")
    print(f"  E_coupling = {np.imag(E_coupling_direct_v2):.6e} GeV (imag part)")
    
    # Interpretation 4: Magnitude of coupling
    # E = alpha * [integral |dphi/dr|^2 4*pi*r^2 dr] * [|integral chi* d(chi)/dsigma dsigma|]
    spatial_integral = 4 * np.pi * np.trapz(dphi_dr**2 * r_grid**2, r_grid)
    subspace_integral = np.abs(circulation)
    E_coupling_magnitude = alpha * spatial_integral * subspace_integral
    
    print(f"\nMagnitude-based Formula:")
    print(f"  E_coupling = {E_coupling_magnitude:.6e} GeV")
    
    # =========================================================================
    # PART 5: Compute using solver's method for comparison
    # =========================================================================
    
    E_coupling_solver = solver.energy_minimizer._compute_coupling_energy(
        chi_scaled=structure_4d,
        Delta_x=Delta_x,
        Delta_sigma=Delta_sigma,
        A=A,
        n_target=generation_n
    )
    
    print(f"\nSolver's _compute_coupling_energy:")
    print(f"  E_coupling = {E_coupling_solver:.6e} GeV")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print(f"\n{'-'*80}")
    print(f"COMPARISON:")
    print(f"  Solver (current):           {E_coupling_solver:.6e} GeV")
    print(f"  Analytical formula:         {E_coupling_analytical:.6e} GeV")
    print(f"  Hamiltonian Method 1:       {E_coupling_direct_v1:.6e} GeV")
    print(f"  Hamiltonian Method 2 (real):{np.real(E_coupling_direct_v2):.6e} GeV")
    print(f"  Magnitude-based:            {E_coupling_magnitude:.6e} GeV")
    print(f"{'-'*80}")
    print()

print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print()
print("Interpretation Guide:")
print("- Solver (current): Uses factorized formula with dimensionless spatial factor")
print("- Hamiltonian Method 1: Direct application of mixed derivative operator")
print("- Hamiltonian Method 2: After integration by parts")
print("- Magnitude-based: Uses magnitudes of spatial and subspace integrals")
print()
print("The correct formula should match the Hamiltonian operator definition.")
print("If there are large discrepancies, the analytical approximation may be incorrect.")

