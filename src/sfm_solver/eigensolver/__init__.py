"""
Eigensolver module for SFM Solver.

Provides solvers for the eigenvalue problems in the Single-Field Model:

Core Infrastructure:
- LinearEigensolver: Standard eigenvalue solver for subspace
- NonlinearEigensolver: Self-consistent solver with g|χ|² term and DIIS mixing
- SpectralOperators: FFT-based differentiation operators

Working Mass Ratio Solvers:
- SFMAmplitudeSolver: Uses scaling law m(n) = m₀ × n^a × exp(b×n) [MAIN SOLUTION]

Legacy / experimental solvers such as the Gross-Pitaevskii solver and the full
coupled (r,σ) eigenvalue solver have been moved to sfm_solver.legacy and are
no longer part of the main public eigensolver API.
"""

from sfm_solver.eigensolver.linear import LinearEigensolver
from sfm_solver.eigensolver.nonlinear import NonlinearEigensolver, ConvergenceInfo
from sfm_solver.eigensolver.spectral import SpectralOperators
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
    # Main solution - SFM amplitude solver
    "SFMAmplitudeSolver",
    "SFMAmplitudeState",
    "solve_sfm_lepton_masses",
]

