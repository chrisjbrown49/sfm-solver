"""
Analyze and display energy components for all leptons.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.unified_solver import UnifiedSFMSolver
import numpy as np

def analyze_particle(solver, name, generation_n, winding_k=0):
    """Analyze energy components for a single particle."""
    print(f"\n{'='*80}")
    print(f"ANALYZING {name.upper()} (n={generation_n}, k={winding_k})")
    print(f"{'='*80}")
    
    # Solve
    result = solver.solve_lepton(generation_n=generation_n, winding_k=winding_k)
    
    if not (result.shape_converged and result.energy_converged):
        print(f"WARNING: {name} did not converge!")
        print(f"  Shape converged: {result.shape_converged}")
        print(f"  Energy converged: {result.energy_converged}")
        return None
    
    # Get final energy components
    components = result.energy_components
    
    # Extract values (in MeV)
    E_spatial = components.get('E_spatial', 0) * 1000  # GeV to MeV
    E_sigma = components.get('E_sigma', 0) * 1000
    E_sigma_kin = components.get('E_kinetic_sigma', 0) * 1000
    E_sigma_pot = components.get('E_potential_sigma', 0) * 1000
    E_sigma_nl = components.get('E_nonlinear_sigma', 0) * 1000
    E_coupling = components.get('E_coupling', 0) * 1000
    E_em = components.get('E_em', 0) * 1000
    E_spin_orbit = components.get('E_spin_orbit', 0) * 1000
    E_curvature = components.get('E_curvature', 0) * 1000
    E_total = components.get('E_total', 0) * 1000
    
    # Scale parameters
    A = result.A
    Delta_x = result.Delta_x
    Delta_sigma = result.Delta_sigma
    
    # Mass from Beautiful Equation
    G_5D = solver.G_5D
    mass_mev = G_5D * (A ** 2) * 1000.0
    
    print(f"\nScale Parameters:")
    print(f"  A (amplitude):    {A:.6f}")
    print(f"  Delta_x:          {Delta_x:.6f} fm")
    print(f"  Delta_sigma:      {Delta_sigma:.6f}")
    
    print(f"\nMass:")
    print(f"  m = G_5D × A² = {G_5D} × {A:.6f}² = {mass_mev:.3f} MeV")
    
    print(f"\nEnergy Components (MeV):")
    print(f"  {'Component':<20} {'Value (MeV)':>15} {'% of |Total|':>12}")
    print(f"  {'-'*20} {'-'*15} {'-'*12}")
    
    abs_total = abs(E_total)
    
    print(f"  {'E_spatial':<20} {E_spatial:>15.3f} {100*abs(E_spatial)/abs_total:>11.1f}%")
    print(f"  {'E_sigma (total)':<20} {E_sigma:>15.3f} {100*abs(E_sigma)/abs_total:>11.1f}%")
    print(f"    {'E_sigma_kin':<18} {E_sigma_kin:>15.3f} {100*abs(E_sigma_kin)/abs_total:>11.1f}%")
    print(f"    {'E_sigma_pot':<18} {E_sigma_pot:>15.3f} {100*abs(E_sigma_pot)/abs_total:>11.1f}%")
    print(f"    {'E_sigma_nl':<18} {E_sigma_nl:>15.3f} {100*abs(E_sigma_nl)/abs_total:>11.1f}%")
    print(f"  {'E_coupling':<20} {E_coupling:>15.3f} {100*abs(E_coupling)/abs_total:>11.1f}%")
    print(f"  {'E_em':<20} {E_em:>15.3f} {100*abs(E_em)/abs_total:>11.1f}%")
    print(f"  {'E_spin_orbit':<20} {E_spin_orbit:>15.3f} {100*abs(E_spin_orbit)/abs_total:>11.1f}%")
    print(f"  {'E_curvature':<20} {E_curvature:>15.3f} {100*abs(E_curvature)/abs_total:>11.1f}%")
    print(f"  {'-'*20} {'-'*15} {'-'*12}")
    print(f"  {'E_total':<20} {E_total:>15.3f} {'100.0%':>12}")
    
    return {
        'name': name,
        'n': generation_n,
        'A': A,
        'Delta_x': Delta_x,
        'Delta_sigma': Delta_sigma,
        'mass_mev': mass_mev,
        'E_spatial': E_spatial,
        'E_sigma': E_sigma,
        'E_sigma_kin': E_sigma_kin,
        'E_sigma_pot': E_sigma_pot,
        'E_sigma_nl': E_sigma_nl,
        'E_coupling': E_coupling,
        'E_em': E_em,
        'E_spin_orbit': E_spin_orbit,
        'E_curvature': E_curvature,
        'E_total': E_total
    }

def main():
    print("="*80)
    print("ENERGY COMPONENT ANALYSIS FOR LEPTONS")
    print("="*80)
    
    # Create solver
    print("\nInitializing solver...")
    solver = UnifiedSFMSolver(verbose=False)
    
    G_5D = solver.G_5D
    alpha = solver.alpha
    V0 = solver.V0
    g1 = solver.g1
    
    print(f"\nFundamental Constants:")
    print(f"  G_5D = {G_5D}")
    print(f"  alpha = {alpha}")
    print(f"  V0 = {V0}")
    print(f"  g1 = {g1}")
    
    # Analyze all leptons
    results = []
    for name, n in [("Electron", 1), ("Muon", 2), ("Tau", 3)]:
        result = analyze_particle(solver, name, n)
        if result:
            results.append(result)
    
    # Summary comparison table
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}\n")
    
    if len(results) == 3:
        print(f"{'Component':<20} {'Electron':>15} {'Muon':>15} {'Tau':>15}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        
        print(f"{'Generation n':<20} {results[0]['n']:>15} {results[1]['n']:>15} {results[2]['n']:>15}")
        print(f"{'Amplitude A':<20} {results[0]['A']:>15.6f} {results[1]['A']:>15.6f} {results[2]['A']:>15.6f}")
        print(f"{'Delta_x (fm)':<20} {results[0]['Delta_x']:>15.6f} {results[1]['Delta_x']:>15.6f} {results[2]['Delta_x']:>15.6f}")
        print(f"{'Delta_sigma':<20} {results[0]['Delta_sigma']:>15.6f} {results[1]['Delta_sigma']:>15.6f} {results[2]['Delta_sigma']:>15.6f}")
        print(f"{'Mass (MeV)':<20} {results[0]['mass_mev']:>15.3f} {results[1]['mass_mev']:>15.3f} {results[2]['mass_mev']:>15.3f}")
        print()
        
        print("Energy Components (MeV):")
        for key in ['E_spatial', 'E_sigma_kin', 'E_sigma_pot', 'E_sigma_nl', 
                    'E_coupling', 'E_em', 'E_spin_orbit', 'E_curvature', 'E_total']:
            label = key.replace('_', ' ')
            print(f"  {label:<18} {results[0][key]:>15.3f} {results[1][key]:>15.3f} {results[2][key]:>15.3f}")
        
        print(f"\n{'Mass Ratios:':<20}")
        print(f"  {'mu/e':<18} {results[1]['mass_mev']/results[0]['mass_mev']:>15.3f} {'(exp: 206.8)':>30}")
        print(f"  {'tau/e':<18} {results[2]['mass_mev']/results[0]['mass_mev']:>15.3f} {'(exp: 3477.2)':>30}")
        print(f"  {'tau/mu':<18} {results[2]['mass_mev']/results[1]['mass_mev']:>15.3f} {'(exp: 16.8)':>30}")

if __name__ == "__main__":
    main()

