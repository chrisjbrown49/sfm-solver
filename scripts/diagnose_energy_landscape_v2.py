"""
Improved diagnostic: For each A, optimize Delta_x and Delta_sigma to find the true energy minimum.
This shows the actual energy landscape the solver sees.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.constants import load_constants_from_json

# Load current constants
constants = load_constants_from_json()
print("="*80)
print("ENERGY LANDSCAPE DIAGNOSTIC V2 (Optimized Scales)")
print("="*80)
print(f"Parameters: G_5D={constants['G_5D']:.2e}, alpha={constants['alpha']:.2f}, g1={constants['g1']:.0f}")
print()

# Create solver
solver = UnifiedSFMSolver(verbose=False)

# A values to scan (wider range, more points)
A_values = np.concatenate([
    np.linspace(0.1, 1.0, 20),    # Fine resolution near electron
    np.linspace(1.0, 10.0, 20),   # Medium amplitudes
    np.linspace(10, 50, 15),      # Muon range
    np.linspace(50, 100, 10)      # Tau range
])

# Generations to test
generations = [
    (1, "Electron (n=1)"),
    (2, "Muon (n=2)"),
    (3, "Tau (n=3)")
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(f'Energy Landscape: Optimized Scales at Each A\n(G_5D={constants["G_5D"]:.2e}, alpha={constants["alpha"]:.1f}, g1={constants["g1"]:.0f})', 
             fontsize=14, fontweight='bold')

for gen_idx, (generation_n, gen_name) in enumerate(generations):
    print(f"\nAnalyzing {gen_name}...")
    print("-" * 60)
    
    # Solve for shape (dimensionless)
    shape_result = solver.shape_solver.solve_lepton_shape(
        winding_k=1,
        generation_n=generation_n,
        max_iter=200,
        tol=1e-6
    )
    
    # Storage for energy components
    results = {
        'A': [],
        'E_total': [],
        'E_sigma': [],
        'E_spatial': [],
        'E_coupling': [],
        'E_em': [],
        'E_curv': [],
        'Delta_x_opt': [],
        'Delta_sigma_opt': []
    }
    
    # Scan over A values
    for i, A in enumerate(A_values):
        if i % 10 == 0:
            print(f"  Scanning A = {A:.2f}...")
        
        # For each A, optimize Delta_x and Delta_sigma
        def energy_for_fixed_A(params):
            Delta_x, Delta_sigma = params
            
            # Bounds check
            if Delta_x < 0.01 or Delta_x > 100 or Delta_sigma < 0.1 or Delta_sigma > 2.0:
                return 1e10
            
            try:
                # Rebuild structure at this scale
                structure = solver.spatial_coupling.build_4d_structure(
                    subspace_shape=shape_result.composite_shape,
                    n_target=generation_n,
                    l_target=0,
                    m_target=0,
                    Delta_x=Delta_x
                )
                
                # Compute total energy using the unified method
                E_total, _ = solver.energy_minimizer._compute_total_energy(
                    structure, Delta_x, Delta_sigma, A, generation_n
                )
                
                return E_total
            except:
                return 1e10
        
        # Initial guess for scales based on A
        if generation_n == 1:
            x0 = [10.0, 1.0]
        elif generation_n == 2:
            x0 = [5.0, 1.0]
        else:
            x0 = [20.0, 1.0]
        
        # Optimize scales for this A
        result = minimize(
            energy_for_fixed_A,
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 0.1, 'fatol': 0.01}
        )
        
        if result.success or result.fun < 1e9:
            Delta_x_opt, Delta_sigma_opt = result.x
            E_total = result.fun
            
            # Recompute components at optimal scales
            structure = solver.spatial_coupling.build_4d_structure(
                subspace_shape=shape_result.composite_shape,
                n_target=generation_n,
                l_target=0,
                m_target=0,
                Delta_x=Delta_x_opt
            )
            
            # Get all energy components
            _, components = solver.energy_minimizer._compute_total_energy(
                structure, Delta_x_opt, Delta_sigma_opt, A, generation_n
            )
            
            E_sig = components['E_sigma']
            E_sp = components['E_spatial']
            E_coup = components['E_coupling']
            E_em = components['E_em']
            E_cur = components['E_curvature']
            
            results['A'].append(A)
            results['E_total'].append(E_total)
            results['E_sigma'].append(E_sig)
            results['E_spatial'].append(E_sp)
            results['E_coupling'].append(E_coup)
            results['E_em'].append(E_em)
            results['E_curv'].append(E_cur)
            results['Delta_x_opt'].append(Delta_x_opt)
            results['Delta_sigma_opt'].append(Delta_sigma_opt)
    
    # Convert to arrays
    A_arr = np.array(results['A'])
    E_total_arr = np.array(results['E_total'])
    E_sigma_arr = np.array(results['E_sigma'])
    E_spatial_arr = np.array(results['E_spatial'])
    E_coupling_arr = np.array(results['E_coupling'])
    E_em_arr = np.array(results['E_em'])
    E_curv_arr = np.array(results['E_curv'])
    
    # Find minimum
    if len(E_total_arr) > 0:
        min_idx = np.argmin(E_total_arr)
        A_min = A_arr[min_idx]
        E_min = E_total_arr[min_idx]
        Dx_min = results['Delta_x_opt'][min_idx]
        Ds_min = results['Delta_sigma_opt'][min_idx]
        
        print(f"\n  MINIMUM FOUND:")
        print(f"    A = {A_min:.4f}, Delta_x = {Dx_min:.4f} fm, Delta_sigma = {Ds_min:.4f}")
        print(f"    E_total = {E_min:.4e} GeV")
        print(f"      E_sigma:   {E_sigma_arr[min_idx]:>12.4e} GeV")
        print(f"      E_spatial: {E_spatial_arr[min_idx]:>12.4e} GeV")
        print(f"      E_coupling:{E_coupling_arr[min_idx]:>12.4e} GeV")
        print(f"      E_em:      {E_em_arr[min_idx]:>12.4e} GeV")
        print(f"      E_curv:    {E_curv_arr[min_idx]:>12.4e} GeV")
    
    # Plot individual components (left column)
    ax1 = axes[gen_idx, 0]
    ax1.plot(A_arr, E_sigma_arr, 'o-', label='E_sigma', linewidth=2, markersize=3)
    ax1.plot(A_arr, E_spatial_arr, 's-', label='E_spatial', linewidth=2, markersize=3)
    ax1.plot(A_arr, E_coupling_arr, '^-', label='E_coupling', linewidth=2, markersize=3)
    ax1.plot(A_arr, E_em_arr, 'v-', label='E_em', linewidth=2, markersize=3)
    ax1.plot(A_arr, E_curv_arr, 'd-', label='E_curv', linewidth=2, markersize=3)
    
    if len(E_total_arr) > 0:
        ax1.axvline(A_min, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Min A={A_min:.1f}')
    
    ax1.set_xlabel('Amplitude A', fontsize=10)
    ax1.set_ylabel('Energy (GeV)', fontsize=10)
    ax1.set_title(f'{gen_name} - Individual Components', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Plot total energy (right column)
    ax2 = axes[gen_idx, 1]
    ax2.plot(A_arr, E_total_arr, 'ko-', linewidth=2.5, markersize=4, label='E_total')
    
    if len(E_total_arr) > 0:
        ax2.plot(A_min, E_min, 'ro', markersize=12, label=f'Min: A={A_min:.1f}', zorder=10)
    
    ax2.set_xlabel('Amplitude A', fontsize=10)
    ax2.set_ylabel('Total Energy (GeV)', fontsize=10)
    ax2.set_title(f'{gen_name} - Total Energy', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
output_path = 'energy_landscape_diagnostic_v2.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Plot saved to: {output_path}")
print(f"{'='*80}")

