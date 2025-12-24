"""
Diagnostic script to visualize energy components vs amplitude A for each lepton generation.
This will help us understand why tau collapses to unphysical values.
"""

import numpy as np
import matplotlib.pyplot as plt
from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.constants import load_constants_from_json

# Load current constants
constants = load_constants_from_json()
print("="*80)
print("ENERGY LANDSCAPE DIAGNOSTIC")
print("="*80)
print(f"Parameters: G_5D={constants['G_5D']:.2e}, alpha={constants['alpha']:.2f}, g1={constants['g1']:.0f}")
print()

# Create solver
solver = UnifiedSFMSolver(verbose=False)

# A values to scan
A_values = np.logspace(-1, 2.5, 100)  # 0.1 to ~300

# Generations to test
generations = [
    (1, "Electron (n=1)"),
    (2, "Muon (n=2)"),
    (3, "Tau (n=3)")
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(f'Energy Components vs Amplitude A\n(G_5D={constants["G_5D"]:.2e}, alpha={constants["alpha"]:.1f}, g1={constants["g1"]:.0f})', 
             fontsize=14, fontweight='bold')

for gen_idx, (generation_n, gen_name) in enumerate(generations):
    print(f"Analyzing {gen_name}...")
    
    # Solve for shape (do this once, it's dimensionless)
    shape_result = solver.shape_solver.solve_lepton_shape(
        winding_k=1,
        generation_n=generation_n,
        max_iter=200,
        tol=1e-6
    )
    
    # Build structure at a reference scale (we'll update Delta_x for each A)
    structure_4d = solver.spatial_coupling.build_4d_structure(
        subspace_shape=shape_result.composite_shape,
        n_target=generation_n,
        l_target=0,
        m_target=0,
        Delta_x=1.0  # Will recompute at each A
    )
    
    # Storage for energy components
    E_total_arr = []
    E_sigma_arr = []
    E_spatial_arr = []
    E_coupling_arr = []
    E_em_arr = []
    E_curv_arr = []
    
    # Scan over A values
    for A in A_values:
        # Estimate reasonable scales for this A
        # Use scaling relationships as initial guess
        Delta_x = 10.0  # Fixed for now to isolate A dependence
        Delta_sigma = 1.0  # Fixed for now
        
        try:
            # Rebuild structure at this scale
            structure_at_scale = solver.spatial_coupling.build_4d_structure(
                subspace_shape=shape_result.composite_shape,
                n_target=generation_n,
                l_target=0,
                m_target=0,
                Delta_x=Delta_x
            )
            
            # Compute energy components
            E_sigma = solver.energy_minimizer._compute_sigma_energy(
                shape_structure=structure_at_scale,
                Delta_sigma=Delta_sigma,
                A=A
            )
            
            E_spatial = solver.energy_minimizer._compute_spatial_energy(
                shape_structure=structure_at_scale,
                Delta_x=Delta_x,
                A=A
            )
            
            E_coupling = solver.energy_minimizer._compute_coupling_energy(
                shape_structure=structure_at_scale,
                Delta_x=Delta_x,
                A=A
            )
            
            E_em = solver.energy_minimizer._compute_em_energy(
                shape_structure=structure_at_scale,
                A=A
            )
            
            E_curv = solver.energy_minimizer._compute_curvature_energy(
                Delta_x=Delta_x,
                A=A
            )
            
            E_total = E_sigma + E_spatial + E_coupling + E_em + E_curv
            
            E_total_arr.append(E_total)
            E_sigma_arr.append(E_sigma)
            E_spatial_arr.append(E_spatial)
            E_coupling_arr.append(E_coupling)
            E_em_arr.append(E_em)
            E_curv_arr.append(E_curv)
            
        except Exception as e:
            # If computation fails, use NaN
            E_total_arr.append(np.nan)
            E_sigma_arr.append(np.nan)
            E_spatial_arr.append(np.nan)
            E_coupling_arr.append(np.nan)
            E_em_arr.append(np.nan)
            E_curv_arr.append(np.nan)
    
    # Convert to arrays
    E_total_arr = np.array(E_total_arr)
    E_sigma_arr = np.array(E_sigma_arr)
    E_spatial_arr = np.array(E_spatial_arr)
    E_coupling_arr = np.array(E_coupling_arr)
    E_em_arr = np.array(E_em_arr)
    E_curv_arr = np.array(E_curv_arr)
    
    # Find minimum
    valid_mask = ~np.isnan(E_total_arr)
    if np.any(valid_mask):
        min_idx = np.nanargmin(E_total_arr)
        A_min = A_values[min_idx]
        E_min = E_total_arr[min_idx]
        print(f"  Minimum at A = {A_min:.2f}, E_total = {E_min:.2e} GeV")
        print(f"    E_sigma:   {E_sigma_arr[min_idx]:>12.2e} GeV")
        print(f"    E_spatial: {E_spatial_arr[min_idx]:>12.2e} GeV")
        print(f"    E_coupling:{E_coupling_arr[min_idx]:>12.2e} GeV")
        print(f"    E_em:      {E_em_arr[min_idx]:>12.2e} GeV")
        print(f"    E_curv:    {E_curv_arr[min_idx]:>12.2e} GeV")
        print()
    
    # Plot individual components (left column)
    ax1 = axes[gen_idx, 0]
    ax1.semilogx(A_values, E_sigma_arr, label='E_sigma', linewidth=2)
    ax1.semilogx(A_values, E_spatial_arr, label='E_spatial', linewidth=2)
    ax1.semilogx(A_values, E_coupling_arr, label='E_coupling', linewidth=2)
    ax1.semilogx(A_values, E_em_arr, label='E_em', linewidth=2)
    ax1.semilogx(A_values, E_curv_arr, label='E_curv', linewidth=2)
    
    if np.any(valid_mask):
        ax1.axvline(A_min, color='red', linestyle='--', alpha=0.5, label=f'Min at A={A_min:.1f}')
    
    ax1.set_xlabel('Amplitude A', fontsize=10)
    ax1.set_ylabel('Energy (GeV)', fontsize=10)
    ax1.set_title(f'{gen_name} - Individual Components', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot total energy (right column)
    ax2 = axes[gen_idx, 1]
    ax2.semilogx(A_values, E_total_arr, 'k-', linewidth=2.5, label='E_total')
    
    if np.any(valid_mask):
        ax2.plot(A_min, E_min, 'ro', markersize=10, label=f'Min: A={A_min:.1f}')
    
    ax2.set_xlabel('Amplitude A', fontsize=10)
    ax2.set_ylabel('Total Energy (GeV)', fontsize=10)
    ax2.set_title(f'{gen_name} - Total Energy', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
output_path = 'energy_landscape_diagnostic.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
print()
print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print()
print("Key insights to look for:")
print("1. Which energy term dominates at large A?")
print("2. Is E_coupling becoming too attractive for higher generations?")
print("3. Do the minima occur at reasonable A values?")
print("4. Are there competing minima causing instability?")

