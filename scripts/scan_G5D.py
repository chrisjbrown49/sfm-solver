"""
Scan G_5D values to find correct operating regime.

After fixing the energy formulas to use G_5D directly (eliminating beta),
we need to find the correct value of G_5D that gives proper energy balance.

This script scans G_5D across many orders of magnitude and reports:
- Amplitude values and convergence
- Energy component balance
- Mass hierarchy quality
- Bound-hitting behavior
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.calculate_beta import calibrate_beta_from_electron
from sfm_solver.core.constants import ELECTRON_MASS_GEV, MUON_MASS_GEV, TAU_MASS_GEV

# Convert to MeV for display
ELECTRON_MASS_MEV = ELECTRON_MASS_GEV * 1000
MUON_MASS_MEV = MUON_MASS_GEV * 1000
TAU_MASS_MEV = TAU_MASS_GEV * 1000


def test_G5D_value(G5D_value, verbose=False):
    """
    Test a specific G_5D value and return comprehensive diagnostics.
    
    Args:
        G5D_value: G_5D value to test (in GeV^-2)
        verbose: Print detailed information
        
    Returns:
        Dictionary with results for all three leptons
    """
    # Create solver with this G_5D
    solver = UnifiedSFMSolver(G_5D=G5D_value, verbose=False)
    
    results = {}
    
    # Solve for all three leptons
    for gen_n, name in [(1, "electron"), (2, "muon"), (3, "tau")]:
        try:
            result = solver.solve_lepton(
                winding_k=1,
                generation_n=gen_n,
                max_iter=200,
                tol=1e-6,
                max_iter_outer=50,
                tol_outer=1e-3
            )
            
            # Store results
            results[name] = {
                'converged': result.converged and result.outer_converged,
                'A': result.A,
                'Delta_x': result.Delta_x,
                'Delta_sigma': result.Delta_sigma,
                'E_total': result.E_total,
                'E_spatial': result.E_spatial,
                'E_curvature': result.E_curvature,
                'E_sigma': result.E_sigma,
                'E_coupling': result.E_coupling,
                'E_em': result.E_em,
                'outer_iterations': result.outer_iterations,
                'hit_min_A': result.A <= 0.001001,  # Near minimum bound
                'hit_max_Dx': result.Delta_x >= 99.9,  # Near maximum bound
            }
        except Exception as e:
            results[name] = {
                'converged': False,
                'error': str(e),
                'A': None
            }
    
    # Calculate beta from electron
    if results['electron']['A'] is not None and results['electron']['converged']:
        beta = calibrate_beta_from_electron(
            A_electron=results['electron']['A'],
            target_mass_MeV=ELECTRON_MASS_MEV
        )
        
        # Calculate masses
        for name in ['electron', 'muon', 'tau']:
            if results[name]['A'] is not None:
                results[name]['mass_MeV'] = beta * results[name]['A']**2
    else:
        beta = None
    
    results['beta'] = beta
    results['G_5D'] = G5D_value
    
    return results


def analyze_energy_balance(results):
    """
    Analyze energy component balance to assess physics quality.
    
    Returns a score (0-100) where 100 is perfect balance.
    """
    if not results['electron']['converged']:
        return 0
    
    # Check for bound-hitting (bad sign)
    if any(results[name]['hit_min_A'] or results[name]['hit_max_Dx'] 
           for name in ['electron', 'muon', 'tau']):
        return 0
    
    # Check amplitude hierarchy (should increase with generation)
    A_e = results['electron']['A']
    A_mu = results['muon']['A']
    A_tau = results['tau']['A']
    
    if not (A_e < A_mu < A_tau):
        return 10  # Wrong hierarchy
    
    # Check energy component balance for electron
    # Ideal: spatial and curvature energies should be comparable
    E_x = results['electron']['E_spatial']
    E_curv = results['electron']['E_curvature']
    
    if E_x == 0 or E_curv == 0:
        return 0
    
    ratio = E_x / E_curv if E_x < E_curv else E_curv / E_x
    
    # Score based on balance (ratio close to 1 is good)
    if ratio > 0.1:
        balance_score = 50 + 40 * ratio  # 50-90 points
    else:
        balance_score = ratio * 500  # 0-50 points for very unbalanced
    
    # Bonus points for correct hierarchy
    hierarchy_score = 10
    
    return min(100, balance_score + hierarchy_score)


def main():
    print("=" * 80)
    print("G_5D PARAMETER SCAN")
    print("=" * 80)
    print()
    print("Scanning G_5D values across many orders of magnitude...")
    print("Goal: Find regime where energy balance is reasonable and")
    print("      amplitude hierarchy emerges naturally.")
    print()
    
    # Scan from 1e-15 to 1e10 in logarithmic steps
    G5D_exponents = np.arange(-15, 11, 1)  # -15 to 10 in steps of 1
    G5D_values = [10.0 ** exp for exp in G5D_exponents]
    
    print(f"Testing {len(G5D_values)} values from 1e-15 to 1e10 GeV^-2")
    print()
    
    # Store all results
    all_results = []
    
    # Run scans
    for i, G5D in enumerate(G5D_values):
        print(f"[{i+1}/{len(G5D_values)}] Testing G_5D = {G5D:.2e} GeV^-2...", end=" ")
        
        results = test_G5D_value(G5D)
        score = analyze_energy_balance(results)
        results['quality_score'] = score
        all_results.append(results)
        
        # Quick status
        status = ""
        if not results['electron']['converged']:
            status = "FAILED"
        elif results['electron']['hit_min_A'] or results['electron']['hit_max_Dx']:
            status = "BOUND"
        elif score > 60:
            status = f"GOOD (score={score:.0f})"
        else:
            status = f"score={score:.0f}"
        
        print(status)
    
    print()
    print("=" * 80)
    print("SCAN RESULTS")
    print("=" * 80)
    print()
    
    # Find best candidates
    valid_results = [r for r in all_results if r['quality_score'] > 50]
    valid_results.sort(key=lambda r: r['quality_score'], reverse=True)
    
    if not valid_results:
        print("WARNING: No G_5D values found with good quality score (>50)!")
        print()
        print("Showing all results with any convergence:")
        print()
        valid_results = [r for r in all_results if r['electron']['converged']]
        valid_results.sort(key=lambda r: r['quality_score'], reverse=True)
    
    # Display top candidates
    print(f"Top {min(10, len(valid_results))} candidates:")
    print()
    print("-" * 80)
    print(f"{'G_5D':<12} {'Score':<8} {'A_e':<10} {'A_mu':<10} {'A_tau':<10} {'Status':<15}")
    print("-" * 80)
    
    for r in valid_results[:10]:
        G5D_str = f"{r['G_5D']:.1e}"
        score_str = f"{r['quality_score']:.0f}"
        A_e_str = f"{r['electron']['A']:.4f}" if r['electron']['A'] else "FAIL"
        A_mu_str = f"{r['muon']['A']:.4f}" if r['muon']['A'] else "FAIL"
        A_tau_str = f"{r['tau']['A']:.4f}" if r['tau']['A'] else "FAIL"
        
        status = ""
        if r['electron']['hit_min_A']:
            status += "MIN_A "
        if r['electron']['hit_max_Dx']:
            status += "MAX_Dx "
        if not status:
            status = "OK"
        
        print(f"{G5D_str:<12} {score_str:<8} {A_e_str:<10} {A_mu_str:<10} {A_tau_str:<10} {status:<15}")
    
    print("-" * 80)
    print()
    
    # Detailed analysis of best candidate
    if valid_results:
        print("=" * 80)
        print("DETAILED ANALYSIS OF BEST CANDIDATE")
        print("=" * 80)
        print()
        
        best = valid_results[0]
        print(f"G_5D = {best['G_5D']:.6e} GeV^-2")
        print(f"Quality Score: {best['quality_score']:.1f}/100")
        print(f"Beta (from electron): {best['beta']:.6f} GeV")
        print()
        
        print("-" * 80)
        print("AMPLITUDES & CONVERGENCE")
        print("-" * 80)
        for name in ['electron', 'muon', 'tau']:
            r = best[name]
            print(f"{name.capitalize():8s}: A={r['A']:8.4f}, Dx={r['Delta_x']:6.2f} fm, "
                  f"Ds={r['Delta_sigma']:6.3f}, "
                  f"iters={r['outer_iterations']:2d}, conv={r['converged']}")
        print()
        
        print("-" * 80)
        print("ENERGY COMPONENTS (Electron)")
        print("-" * 80)
        e = best['electron']
        print(f"  E_spatial:     {e['E_spatial']:12.6e} GeV")
        print(f"  E_curvature:   {e['E_curvature']:12.6e} GeV")
        print(f"  E_sigma:       {e['E_sigma']:12.6e} GeV")
        print(f"  E_coupling:    {e['E_coupling']:12.6e} GeV")
        print(f"  E_em:          {e['E_em']:12.6e} GeV")
        print(f"  ----------")
        print(f"  E_total:       {e['E_total']:12.6e} GeV")
        print()
        
        # Energy balance ratio
        if e['E_curvature'] != 0:
            ratio = e['E_spatial'] / e['E_curvature']
            print(f"  E_spatial/E_curvature ratio: {ratio:.4f}")
            if 0.1 < ratio < 10:
                print(f"    -> Good balance!")
            elif ratio < 0.1:
                print(f"    -> Curvature dominates (may favor delocalization)")
            else:
                print(f"    -> Spatial dominates (may favor extreme confinement)")
        print()
        
        print("-" * 80)
        print("MASS PREDICTIONS")
        print("-" * 80)
        exp_masses = [ELECTRON_MASS_MEV, MUON_MASS_MEV, TAU_MASS_MEV]
        for i, name in enumerate(['electron', 'muon', 'tau']):
            pred = best[name]['mass_MeV']
            exp = exp_masses[i]
            err = abs(pred - exp) / exp * 100
            print(f"{name.capitalize():8s}: {pred:10.4f} MeV  (exp: {exp:10.4f} MeV)  "
                  f"Error: {err:6.2f}%")
        print()
        
        # Mass ratios
        m_e = best['electron']['mass_MeV']
        m_mu = best['muon']['mass_MeV']
        m_tau = best['tau']['mass_MeV']
        
        print("-" * 80)
        print("MASS RATIOS")
        print("-" * 80)
        print(f"  mu/e:   {m_mu/m_e:8.2f}  (exp: 206.77)")
        print(f"  tau/e:  {m_tau/m_e:8.2f}  (exp: 3477.23)")
        print(f"  tau/mu: {m_tau/m_mu:8.2f}  (exp: 16.82)")
        print()
    else:
        print("No valid candidates found!")
        print()
        print("This suggests G_5D may need to be outside the tested range,")
        print("or other parameters (g1, alpha, etc.) may need adjustment.")
    
    print("=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


