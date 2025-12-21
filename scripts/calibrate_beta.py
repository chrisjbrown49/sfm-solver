"""
Utility script to calibrate β from electron mass.

This demonstrates the β-independent energy minimization and calibration workflow.
The calibrated β can then be used for all other particle calculations.
"""

import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver


def main():
    print("="*80)
    print("BETA CALIBRATION FROM ELECTRON MASS")
    print("="*80)
    print("\nThis script demonstrates the correct use of beta in the SFM framework:")
    print("  1. Energy minimization uses g_internal (fundamental parameter)")
    print("  2. beta is calibrated from the electron mass: beta = m_e / A_e^2")
    print("  3. All other particle masses use: m = beta x A^2")
    print("  4. Mass ratios emerge from amplitude ratios")
    
    # Create solver without beta (will calibrate)
    print("\n" + "-"*80)
    print("Initializing solver without beta...")
    print("-"*80)
    
    solver = UnifiedSFMSolver(
        beta=None,  # No beta yet - will calibrate
        n_max=5,
        l_max=2,
        N_sigma=64,
        verbose=False
    )
    
    print("Solver initialized with:")
    print(f"  g_internal = {solver.g_internal:.6f} (FUNDAMENTAL)")
    print(f"  alpha = {solver.alpha:.6f} GeV")
    print(f"  g1 = {solver.g1:.3f}")
    print(f"  g2 = {solver.g2:.6f}")
    print(f"  beta = Not yet calibrated")
    
    # Calibrate from electron
    print("\n" + "="*80)
    print("STEP 1: CALIBRATING BETA FROM ELECTRON")
    print("="*80)
    
    beta_calibrated = solver.calibrate_beta_from_electron(verbose=True)
    
    print(f"\n{'='*80}")
    print("CALIBRATION COMPLETE")
    print("="*80)
    print(f"Calibrated beta = {beta_calibrated:.6f} GeV")
    print(f"\nThis beta is now fixed for all subsequent calculations.")
    print(f"All particle masses will use: m = beta x A^2")
    
    # Verify with muon
    print(f"\n{'='*80}")
    print("STEP 2: VERIFYING WITH MUON")
    print("="*80)
    print(f"\nSolving for muon using calibrated beta...")
    
    result_mu = solver.solve_lepton(winding_k=1, generation_n=2)
    
    m_mu_pred = result_mu.mass * 1000  # MeV
    m_e = 0.510999  # MeV (experimental)
    m_mu_exp = 105.658  # MeV (experimental)
    
    print(f"\nResults:")
    print(f"  Muon amplitude: A_mu = {result_mu.A:.6f}")
    print(f"  Predicted mass: m_mu = beta x A_mu^2 = {m_mu_pred:.3f} MeV")
    print(f"  Experimental:   m_mu = {m_mu_exp:.3f} MeV")
    print(f"  Error: {abs(m_mu_pred - m_mu_exp)/m_mu_exp*100:.1f}%")
    
    print(f"\nMass ratio:")
    print(f"  Predicted m_mu/m_e = {m_mu_pred/m_e:.1f}")
    print(f"  Experimental m_mu/m_e = {m_mu_exp/m_e:.1f}")
    print(f"  (Should emerge from amplitude ratio A_mu^2/A_e^2)")
    
    # Verify with tau
    print(f"\n{'='*80}")
    print("STEP 3: VERIFYING WITH TAU")
    print("="*80)
    print(f"\nSolving for tau using calibrated beta...")
    
    result_tau = solver.solve_lepton(winding_k=1, generation_n=3)
    
    m_tau_pred = result_tau.mass * 1000  # MeV
    m_tau_exp = 1776.86  # MeV (experimental)
    
    print(f"\nResults:")
    print(f"  Tau amplitude: A_tau = {result_tau.A:.6f}")
    print(f"  Predicted mass: m_tau = beta x A_tau^2 = {m_tau_pred:.3f} MeV")
    print(f"  Experimental:   m_tau = {m_tau_exp:.3f} MeV")
    print(f"  Error: {abs(m_tau_pred - m_tau_exp)/m_tau_exp*100:.1f}%")
    
    print(f"\nMass ratio:")
    print(f"  Predicted m_tau/m_e = {m_tau_pred/m_e:.1f}")
    print(f"  Experimental m_tau/m_e = {m_tau_exp/m_e:.1f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"\nCalibrated beta = {beta_calibrated:.6f} GeV")
    print(f"\nPhysics insights:")
    print(f"  - g_internal = {solver.g_internal:.6f} is the FUNDAMENTAL parameter")
    print(f"  - beta is DERIVED from electron calibration")
    print(f"  - Energy minimization uses g_internal, NOT beta")
    print(f"  - beta only appears in final mass calculation: m = beta x A^2")
    print(f"  - Mass ratios = Amplitude ratios (independent of beta)")
    print(f"\nLepton masses:")
    print(f"  Electron: {0.511:.3f} MeV (exact by calibration)")
    print(f"  Muon:     {m_mu_pred:.3f} MeV (predicted)")
    print(f"  Tau:      {m_tau_pred:.3f} MeV (predicted)")
    
    print(f"\nThe generation hierarchy must emerge from amplitude ratios.")
    print(f"If all masses are similar, the physics needs adjustment (not beta!).")


if __name__ == '__main__':
    main()

