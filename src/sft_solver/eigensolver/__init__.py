"""
Eigensolver module for SFT Solver.

Contains linear and nonlinear eigenvalue solvers for the 
Schrödinger-like equation on S¹.
"""

from sft_solver.eigensolver.linear import LinearEigensolver
from sft_solver.eigensolver.nonlinear import NonlinearEigensolver
from sft_solver.eigensolver.spectral import SpectralOperators

__all__ = [
    "LinearEigensolver",
    "NonlinearEigensolver",
    "SpectralOperators",
]

