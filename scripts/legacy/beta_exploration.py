"""
Manual beta parameter exploration for SFM.

Run this script to test specific beta values and see:
1. All derived parameters (kappa, g1, alpha)
2. Predicted particle masses
3. Comparison with experimental values

USAGE:
------
# Mode 1: Derive all parameters from beta (first-principles)
python scripts/beta_exploration.py --beta 0.1

# Mode 2: Specify all 4 parameters directly (manual override)
python scripts/beta_exploration.py --beta 0.1 --alpha 0.05 --kappa 100.0 --g1 1.43

# Scan mode
python scripts/beta_exploration.py --scan --min 0.01 --max 1.0 --points 20
"""

import numpy as np
from sfm_solver.optimization.parameter_optimizer import (
    derive_all_parameters_from_beta,
    SFMParameterOptimizer,
    CALIBRATION_PARTICLES,
    VALIDATION_PARTICLES,
)


def explore_beta(
    beta_gev: float,
    alpha_override: float = None,
    kappa_override: float = None,
    g1_override: float = None,
    verbose: bool = True
):
    """
    Manually explore a specific beta value.
    
    Args:
        beta_gev: The beta value to test (in GeV)
        alpha_override: If provided, use this instead of derived alpha
        kappa_override: If provided, use this instead of derived kappa
        g1_override: If provided, use this instead of derived g1
        verbose: Print detailed output
        
    Returns:
        Tuple of (params_dict, results_dict)
    """
    # Step 1: Derive parameters from beta (first-principles)
    derived_params = derive_all_parameters_from_beta(beta_gev)
    
    # Check if we're using manual overrides
    using_overrides = any(x is not None for x in [alpha_override, kappa_override, g1_override])
    
    # Build final parameter set
    params = {
        'beta': beta_gev,
        'alpha': alpha_override if alpha_override is not None else derived_params['alpha'],
        'kappa': kappa_override if kappa_override is not None else derived_params['kappa'],
        'g1': g1_override if g1_override is not None else derived_params['g1'],
        'L0': derived_params['L0'],
    }
    
    if verbose:
        print("=" * 70)
        if using_overrides:
            print(f"MANUAL PARAMETER OVERRIDE: β = {beta_gev} GeV")
            print("(Using user-specified values, NOT first-principles derivation)")
        else:
            print(f"FIRST-PRINCIPLES EXPLORATION: β = {beta_gev} GeV")
            print("(All parameters derived from β using SFM relationships)")
        print("=" * 70)
        print("\nPARAMETERS:")
        print(f"  β (beta)  = {params['beta']:.6f} GeV")
        print(f"  α (alpha) = {params['alpha']:.6f} GeV", end="")
        if alpha_override is not None:
            print(f"  [OVERRIDE - derived would be {derived_params['alpha']:.6f}]")
        else:
            print()
        print(f"  κ (kappa) = {params['kappa']:.6e} GeV⁻²", end="")
        if kappa_override is not None:
            print(f"  [OVERRIDE - derived would be {derived_params['kappa']:.6e}]")
        else:
            print()
        print(f"  g₁        = {params['g1']:.4f}", end="")
        if g1_override is not None:
            print(f"  [OVERRIDE - derived would be {derived_params['g1']:.4f}]")
        else:
            print()
        print(f"  L₀        = {params['L0']:.6f} GeV⁻¹")
        print()
    
    # Step 2: Create optimizer (just to use its predict_mass method)
    optimizer = SFMParameterOptimizer(
        calibration_particles=CALIBRATION_PARTICLES,
        validation_particles=VALIDATION_PARTICLES,
        N_grid=64,
        verbose=False,
    )
    
    # Step 3: Predict masses for calibration particles
    if verbose:
        print("CALIBRATION PARTICLE PREDICTIONS:")
        print("-" * 50)
    
    results = {}
    for particle in CALIBRATION_PARTICLES:
        m_pred = optimizer.predict_mass(
            particle,
            beta=params['beta'],
            alpha=params['alpha'],
            kappa=params['kappa'],
            g1=params['g1'],
        )
        
        m_exp = particle.mass_gev
        
        if m_pred is not None and m_pred > 0:
            error_pct = abs(m_pred - m_exp) / m_exp * 100
            results[particle.name] = {
                'predicted': m_pred,
                'experimental': m_exp,
                'error_pct': error_pct,
            }
            if verbose:
                status = '✅' if error_pct < 10 else '❌'
                print(f"  {particle.name:12s}: pred={m_pred*1000:10.3f} MeV, "
                      f"exp={m_exp*1000:8.3f} MeV, error={error_pct:6.1f}% {status}")
        else:
            results[particle.name] = {'predicted': None, 'experimental': m_exp}
            if verbose:
                print(f"  {particle.name:12s}: SOLVER FAILED (exp={m_exp*1000:.3f} MeV)")
    
    # Step 4: Predict masses for validation particles
    if verbose:
        print("\nVALIDATION PARTICLE PREDICTIONS:")
        print("-" * 50)
    
    for particle in VALIDATION_PARTICLES:
        m_pred = optimizer.predict_mass(
            particle,
            beta=params['beta'],
            alpha=params['alpha'],
            kappa=params['kappa'],
            g1=params['g1'],
        )
        
        m_exp = particle.mass_gev
        
        if m_pred is not None and m_pred > 0:
            error_pct = abs(m_pred - m_exp) / m_exp * 100
            results[particle.name] = {
                'predicted': m_pred,
                'experimental': m_exp,
                'error_pct': error_pct,
            }
            if verbose:
                status = '✅' if error_pct < 10 else '❌'
                print(f"  {particle.name:12s}: pred={m_pred*1000:10.3f} MeV, "
                      f"exp={m_exp*1000:8.3f} MeV, error={error_pct:6.1f}% {status}")
        else:
            results[particle.name] = {'predicted': None, 'experimental': m_exp}
            if verbose:
                print(f"  {particle.name:12s}: SOLVER FAILED (exp={m_exp*1000:.3f} MeV)")
    
    if verbose:
        print("=" * 70)
    
    return params, results


def scan_beta_range(beta_min: float, beta_max: float, n_points: int = 10):
    """
    Scan a range of beta values and show derived parameters.
    
    Args:
        beta_min: Minimum beta (GeV)
        beta_max: Maximum beta (GeV)
        n_points: Number of points to sample
    """
    print("=" * 70)
    print(f"BETA SCAN: {beta_min} to {beta_max} GeV ({n_points} points)")
    print("=" * 70)
    print()
    print(f"{'Beta (GeV)':>12} | {'kappa (GeV⁻²)':>14} | {'g1':>10} | {'alpha (GeV)':>12}")
    print("-" * 60)
    
    # Use log spacing if range spans orders of magnitude
    if beta_max / beta_min > 10:
        betas = np.logspace(np.log10(beta_min), np.log10(beta_max), n_points)
    else:
        betas = np.linspace(beta_min, beta_max, n_points)
    
    for beta in betas:
        params = derive_all_parameters_from_beta(beta)
        print(f"{beta:12.4f} | {params['kappa']:14.6e} | {params['g1']:10.4f} | {params['alpha']:12.6f}")
    
    print("=" * 70)


# ============================================================
# MAIN: Run with command-line arguments
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Explore SFM beta parameter values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
---------
  # Mode 1: Derive all parameters from beta (first-principles)
  python scripts/beta_exploration.py --beta 0.1

  # Mode 2: Specify all 4 parameters directly (manual override)
  python scripts/beta_exploration.py --beta 0.1 --alpha 0.05 --kappa 100.0 --g1 1.43

  # Scan a range of beta values
  python scripts/beta_exploration.py --scan --min 0.001 --max 1.0 --points 20
        """
    )
    
    # Single value exploration
    parser.add_argument('--beta', type=float, default=None,
                        help='Beta value to test (in GeV). If only beta is provided, '
                             'alpha/kappa/g1 are derived from first principles.')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Override alpha value (in GeV). Requires --beta.')
    parser.add_argument('--kappa', type=float, default=None,
                        help='Override kappa value (in GeV⁻²). Requires --beta.')
    parser.add_argument('--g1', type=float, default=None,
                        help='Override g1 value. Requires --beta.')
    
    # Scan mode
    parser.add_argument('--scan', action='store_true',
                        help='Run a scan over a range of beta values')
    parser.add_argument('--min', type=float, default=0.01,
                        help='Minimum beta for scan (default: 0.01 GeV)')
    parser.add_argument('--max', type=float, default=1.0,
                        help='Maximum beta for scan (default: 1.0 GeV)')
    parser.add_argument('--points', type=int, default=10,
                        help='Number of points for scan (default: 10)')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if any([args.alpha, args.kappa, args.g1]) and args.beta is None:
        parser.error("--alpha, --kappa, and --g1 require --beta to be specified")
    
    if args.beta is not None:
        # Single beta exploration (with optional overrides)
        has_any_override = any([args.alpha, args.kappa, args.g1])
        has_all_overrides = all([args.alpha, args.kappa, args.g1])
        
        if has_any_override and not has_all_overrides:
            # Partial override - warn but allow (missing ones will be derived)
            print("WARNING: Partial parameter override detected.")
            print("         Missing parameters will be derived from beta.")
            print()
        
        explore_beta(
            beta_gev=args.beta,
            alpha_override=args.alpha,
            kappa_override=args.kappa,
            g1_override=args.g1,
        )
        
    elif args.scan:
        # Scan mode
        scan_beta_range(beta_min=args.min, beta_max=args.max, n_points=args.points)
        
    else:
        # Default: show both scan and a few test values
        print("\n" + "=" * 70)
        print("STEP 1: Parameter relationships across beta range")
        print("=" * 70 + "\n")
        scan_beta_range(beta_min=args.min, beta_max=args.max, n_points=args.points)
        
        print("\n" + "=" * 70)
        print("STEP 2: Detailed predictions for specific beta values")
        print("=" * 70 + "\n")
        for beta in [0.01, 0.05, 0.1, 0.5, 1.0]:
            explore_beta(beta)
            print()
