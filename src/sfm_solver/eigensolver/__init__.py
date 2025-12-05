"""
Eigensolver module for SFM Solver.

Provides solvers for the eigenvalue problems in the Single-Field Model:

Core Infrastructure:
- LinearEigensolver: Standard eigenvalue solver for subspace
- NonlinearEigensolver: Self-consistent solver with g|χ|² term and DIIS mixing
- SpectralOperators: FFT-based differentiation operators

Working Mass Ratio Solvers:
- SFMAmplitudeSolver: Uses scaling law m(n) = m₀ × n^a × exp(b×n) [MAIN SOLUTION]
- GrossPitaevskiiSolver: Non-normalized wavefunctions with particle number N

Supporting Infrastructure:
- CoupledEigensolver: Joint (r,σ) eigenvalue problem
- CoupledGrids, CoupledHamiltonian: 2D grid infrastructure
"""

from sfm_solver.eigensolver.linear import LinearEigensolver
from sfm_solver.eigensolver.nonlinear import NonlinearEigensolver, ConvergenceInfo
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.eigensolver.gp_solver import (
    GrossPitaevskiiSolver,
    GPSolution,
    compute_mass_ratios_from_gp,
)
from sfm_solver.eigensolver.coupled_solver import (
    CoupledGrids,
    CoupledHamiltonian,
    CoupledEigensolver,
    CoupledSolution,
    compute_mass_ratios,
)
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
    # GP solver (alternative approach)
    "GrossPitaevskiiSolver",
    "GPSolution",
    "compute_mass_ratios_from_gp",
    # Coupled solver infrastructure
    "CoupledGrids",
    "CoupledHamiltonian",
    "CoupledEigensolver",
    "CoupledSolution",
    "compute_mass_ratios",
]
