"""
Helper function for determining mass scale from electron.

This module provides the calibrate_beta_from_electron() function which should
be called from test scripts to determine the mass scale beta from the experimental
electron mass.

The mass scale beta converts dimensionless amplitudes to physical masses: m = beta × A²

This function is NOT called automatically by the solver - it must be invoked
explicitly by test scripts when mass scale determination is needed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sfm_solver.core.unified_solver import UnifiedSFMSolver


def calibrate_beta_from_electron(
    solver: 'UnifiedSFMSolver',
    electron_mass_exp: float = 0.000510999,  # GeV
    max_iter: int = 200,
    max_iter_outer: int = 50,
    tol_outer: float = 1e-3
) -> float:
    """
    Determine mass scale by solving for electron and matching experimental mass.
    
    The electron's amplitude A_e is found through the full solver with outer loop
    iteration (same method used for solving other particles).
    The mass scale beta is then determined to match the experimental electron mass:
    beta = m_e_exp / A_e^2
    
    This mass scale can then be used to convert all particle amplitudes to masses.
    
    Process:
        1. Solve electron with full outer loop iteration (same as other leptons)
        2. Extract converged amplitude A_e
        3. Determine mass scale: beta = m_e_exp / A_e^2
        4. Return mass scale for use in test scripts
    
    Args:
        solver: UnifiedSFMSolver instance to use for solving electron
        electron_mass_exp: Experimental electron mass (GeV)
        max_iter: Maximum iterations for shape solver (default: 200)
        max_iter_outer: Maximum outer loop iterations (default: 50)
        tol_outer: Outer loop convergence tolerance (default: 1e-3)
        
    Returns:
        beta: Mass scale factor (GeV) for converting amplitudes to masses
        
    Example:
        >>> from sfm_solver.core.unified_solver import UnifiedSFMSolver
        >>> from sfm_solver.core.calculate_beta import calibrate_beta_from_electron
        >>> 
        >>> solver = UnifiedSFMSolver()
        >>> beta = calibrate_beta_from_electron(solver)
        >>> print(f"Mass scale: {beta:.6f} GeV")
        >>> 
        >>> # Now solve other particles and convert amplitudes to masses
        >>> result = solver.solve_lepton(generation_n=2, max_iter_outer=50)
        >>> mass_mu = beta * result.A**2
        >>> print(f"Muon mass: {mass_mu*1000:.3f} MeV")
    """
    # Solve electron with full outer loop (same method as other leptons)
    result = solver.solve_lepton(
        winding_k=1,
        generation_n=1,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        tol_outer=tol_outer
    )
    
    A_electron = result.A
    
    # Determine mass scale from amplitude
    beta_calibrated = electron_mass_exp / (A_electron**2)
    
    return beta_calibrated

