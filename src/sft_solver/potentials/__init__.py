"""
Potentials module for SFT Solver.

Contains potential energy functions for the S¹ subspace.
"""

from sft_solver.potentials.three_well import ThreeWellPotential
from sft_solver.potentials.effective import EffectivePotential, SpinOrbitPotential

__all__ = [
    "ThreeWellPotential",
    "EffectivePotential",
    "SpinOrbitPotential",
]

