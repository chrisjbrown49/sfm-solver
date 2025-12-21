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
    max_iter: int = 200
) -> float:
    """
    Determine mass scale by solving for electron and matching experimental mass.
    
    The electron's amplitude A_e is found through pure energy minimization.
    The mass scale beta is then determined to match the experimental electron mass:
    beta = m_e_exp / A_e^2
    
    This mass scale can then be used to convert all particle amplitudes to masses.
    
    Process:
        1. Solve electron shape (Stage 1, dimensionless field configuration)
        2. Find optimal amplitude A_e through energy minimization (scale-independent)
        3. Determine mass scale: beta = m_e_exp / A_e^2
        4. Return mass scale for use in test scripts
    
    Args:
        solver: UnifiedSFMSolver instance to use for solving electron
        electron_mass_exp: Experimental electron mass (GeV)
        max_iter: Maximum iterations for shape solver
        
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
        >>> result = solver.solve_lepton(generation_n=2)
        >>> mass_mu = beta * result.A**2
        >>> print(f"Muon mass: {mass_mu*1000:.3f} MeV")
    """
    # Solve electron shape (dimensionless)
    shape_result = solver.shape_solver.solve_lepton_shape(
        generation_n=1,
        winding_k=1,
        max_iter=max_iter,
        tol=1e-6
    )
    
    # Build 4D structure
    structure_4d = solver.spatial_coupling.build_4d_structure(
        subspace_shape=shape_result.composite_shape,
        n_target=1,
        l_target=0,
        m_target=0
    )
    
    # Find optimal amplitude through energy minimization
    # (Energy minimizer works without mass scale - pure amplitude optimization)
    optimization_result = solver.energy_minimizer.minimize_lepton_energy(
        shape_structure=structure_4d,
        generation_n=1
    )
    
    A_electron = optimization_result.A
    
    # Determine mass scale from amplitude
    beta_calibrated = electron_mass_exp / (A_electron**2)
    
    return beta_calibrated

