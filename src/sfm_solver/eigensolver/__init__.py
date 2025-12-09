"""
Eigensolver module for SFM Solver.

Provides solvers for the eigenvalue problems in the Single-Field Model:

Core Infrastructure:
- SpectralOperators: FFT-based differentiation operators (used by all solvers)

Physics-Based Lepton Solver:
- SFMLeptonSolver: Uses four-term energy functional with NO fitted parameters
  Mass ratios EMERGE from energy minimization, consistent with meson/baryon solvers

Legacy solvers (LinearEigensolver, NonlinearEigensolver, fitted amplitude solver,
Gross-Pitaevskii solver, coupled solver) have been moved to sfm_solver.legacy
and are no longer part of the main eigensolver API.
"""

from sfm_solver.eigensolver.spectral import SpectralOperators

# Physics-based lepton solver
from sfm_solver.eigensolver.sfm_lepton_solver import (
    SFMLeptonSolver,
    SFMLeptonState,
    solve_lepton_masses,
    LEPTON_WINDING,
    LEPTON_SPATIAL_MODE,
)

__all__ = [
    # Core infrastructure
    "SpectralOperators",
    # Physics-based lepton solver
    "SFMLeptonSolver",
    "SFMLeptonState",
    "solve_lepton_masses",
    "LEPTON_WINDING",
    "LEPTON_SPATIAL_MODE",
]

