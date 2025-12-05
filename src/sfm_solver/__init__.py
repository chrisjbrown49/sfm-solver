"""
SFM Solver - Single-Field Model Tier 1 Eigenvalue Solver

A numerical solver for computing eigenvalues and eigenfunctions
in the Single-Field Model framework.
"""

from sfm_solver.core import (
    HBAR,
    C,
    E_CHARGE,
    SFMParameters,
    SpectralGrid,
)
from sfm_solver.potentials import ThreeWellPotential, EffectivePotential
from sfm_solver.eigensolver import LinearEigensolver, NonlinearEigensolver

__version__ = "0.1.0"

__all__ = [
    # Constants
    "HBAR",
    "C",
    "E_CHARGE",
    # Core classes
    "SFMParameters",
    "SpectralGrid",
    # Potentials
    "ThreeWellPotential",
    "EffectivePotential",
    # Eigensolvers
    "LinearEigensolver",
    "NonlinearEigensolver",
]
