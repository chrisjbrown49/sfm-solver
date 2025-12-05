"""
Eigensolver module for SFM Solver.

Contains linear and nonlinear eigenvalue solvers for the 
Schrödinger-like equation on S¹.
"""

from sfm_solver.eigensolver.linear import LinearEigensolver
from sfm_solver.eigensolver.nonlinear import NonlinearEigensolver
from sfm_solver.eigensolver.spectral import SpectralOperators

__all__ = [
    "LinearEigensolver",
    "NonlinearEigensolver",
    "SpectralOperators",
]
