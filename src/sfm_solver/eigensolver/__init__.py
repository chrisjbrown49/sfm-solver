"""
Eigensolver module for SFM Solver.

Provides solvers for the eigenvalue problems in the Single-Field Model:

Core Infrastructure:
- LinearEigensolver: Standard eigenvalue solver for subspace
- NonlinearEigensolver: Self-consistent solver with g|χ|² term and DIIS mixing
- SpectralOperators: FFT-based differentiation operators

Physics-Based Lepton Solver (RECOMMENDED):
- SFMLeptonSolver: Uses four-term energy functional with NO fitted parameters
  Mass ratios EMERGE from energy minimization, consistent with meson/baryon solvers

Legacy Solvers:
- SFMAmplitudeSolver: Uses fitted scaling law m(n) = m₀ × n^a × exp(b×n)
  (Deprecated - use SFMLeptonSolver for physics-based approach)

Legacy / experimental solvers such as the Gross-Pitaevskii solver and the full
coupled (r,σ) eigenvalue solver have been moved to sfm_solver.legacy and are
no longer part of the main public eigensolver API.
"""

from sfm_solver.eigensolver.linear import LinearEigensolver
from sfm_solver.eigensolver.nonlinear import NonlinearEigensolver, ConvergenceInfo
from sfm_solver.eigensolver.spectral import SpectralOperators

# Physics-based lepton solver (RECOMMENDED)
from sfm_solver.eigensolver.sfm_lepton_solver import (
    SFMLeptonSolver,
    SFMLeptonState,
    solve_lepton_masses,
    LEPTON_WINDING,
    LEPTON_SPATIAL_MODE,
)

# Legacy amplitude solver (deprecated - uses fitted scaling law)
from sfm_solver.eigensolver.sfm_amplitude_solver import (
    SFMAmplitudeSolver,
    SFMAmplitudeState,
    solve_sfm_lepton_masses,
)

__all__ = [
    # Core infrastructure
    "LinearEigensolver",
    "NonlinearEigensolver",
    "ConvergenceInfo",
    "SpectralOperators",
    # Physics-based lepton solver (RECOMMENDED)
    "SFMLeptonSolver",
    "SFMLeptonState",
    "solve_lepton_masses",
    "LEPTON_WINDING",
    "LEPTON_SPATIAL_MODE",
    # Legacy amplitude solver (deprecated)
    "SFMAmplitudeSolver",
    "SFMAmplitudeState",
    "solve_sfm_lepton_masses",
]

