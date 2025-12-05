"""
Eigensolver module for SFM Solver.

Provides solvers for the nonlinear eigenvalue problem:
    (H_0 + g|χ|²) χ = E χ
    
Includes:
- Linear eigensolver for subspace-only problems
- Nonlinear (self-consistent) solver for subspace with g|χ|² term
- Amplitude-quantized solver for mass hierarchy via different A branches
- Coupled solver for full spacetime-subspace problem with mass hierarchy
"""

from sfm_solver.eigensolver.linear import LinearEigensolver
from sfm_solver.eigensolver.nonlinear import NonlinearEigensolver, ConvergenceInfo
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.eigensolver.amplitude_solver import (
    AmplitudeQuantizedSolver,
    AmplitudeSolution,
    ImaginaryTimeEvolution,
    find_lepton_mass_hierarchy,
)
from sfm_solver.eigensolver.nonlinear_coupled import (
    NonlinearCoupledSolver,
    NonlinearCoupledSolution,
    find_amplitude_quantization,
)
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

__all__ = [
    "LinearEigensolver",
    "NonlinearEigensolver",
    "ConvergenceInfo",
    "SpectralOperators",
    "AmplitudeQuantizedSolver",
    "AmplitudeSolution",
    "ImaginaryTimeEvolution",
    "find_lepton_mass_hierarchy",
    "NonlinearCoupledSolver",
    "NonlinearCoupledSolution",
    "find_amplitude_quantization",
    "GrossPitaevskiiSolver",
    "GPSolution",
    "compute_mass_ratios_from_gp",
    "CoupledGrids",
    "CoupledHamiltonian",
    "CoupledEigensolver",
    "CoupledSolution",
    "compute_mass_ratios",
]
