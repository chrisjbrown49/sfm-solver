"""
SFM Solver - Single-Field Model Eigenvalue Solver

A numerical solver for computing eigenvalues and eigenfunctions
in the Single-Field Model framework.

This top-level package exposes the core infrastructure used in the validated
Tier 1 / Tier 1b / Tier 2 / Tier 2b pipelines:
- Subspace-only eigensolvers (linear and nonlinear)
- Three-well subspace potential
- Core SFM parameters and grids

Legacy and experimental components (GP solver, full coupled (r,σ) solver,
spin-orbit effective potentials, α-fitting utilities, and their tests) have
been moved to sfm_solver.legacy and are no longer part of the public API.
"""

from sfm_solver.core import (
    HBAR,
    C,
    E_CHARGE,
    SFMParameters,
    SpectralGrid,
)
from sfm_solver.potentials import ThreeWellPotential
from sfm_solver.eigensolver import (
    LinearEigensolver,
    NonlinearEigensolver,
    SpectralOperators,
    SFMAmplitudeSolver,
    SFMAmplitudeState,
    solve_sfm_lepton_masses,
)
from sfm_solver.spatial.radial import RadialGrid, RadialOperators
from sfm_solver.reporting import ResultsReporter

__version__ = "0.1.0"

__all__ = [
    # Constants
    "HBAR",
    "C",
    "E_CHARGE",
    # Core classes
    "SFMParameters",
    "SpectralGrid",
    # Spatial grid
    "RadialGrid",
    "RadialOperators",
    # Potentials
    "ThreeWellPotential",
    # Eigensolvers
    "LinearEigensolver",
    "NonlinearEigensolver",
    "SpectralOperators",
    "SFMAmplitudeSolver",
    "SFMAmplitudeState",
    "solve_sfm_lepton_masses",
    # Reporting
    "ResultsReporter",
]

