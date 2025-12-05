"""
SFT Solver - Single-Field Theory Tier 1 Eigenvalue Solver

A numerical solver for computing eigenvalues and eigenfunctions
in the Single-Field Theory framework.
"""

from sft_solver.core import (
    HBAR,
    C,
    E_CHARGE,
    SFTParameters,
    SpectralGrid,
)
from sft_solver.potentials import ThreeWellPotential, EffectivePotential
from sft_solver.eigensolver import LinearEigensolver, NonlinearEigensolver

__version__ = "0.1.0"

__all__ = [
    # Constants
    "HBAR",
    "C",
    "E_CHARGE",
    # Core classes
    "SFTParameters",
    "SpectralGrid",
    # Potentials
    "ThreeWellPotential",
    "EffectivePotential",
    # Eigensolvers
    "LinearEigensolver",
    "NonlinearEigensolver",
]

