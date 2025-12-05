"""
Potentials module for SFM Solver.

Contains potential energy functions for the S¹ subspace.
"""

from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.potentials.effective import EffectivePotential, SpinOrbitPotential

__all__ = [
    "ThreeWellPotential",
    "EffectivePotential",
    "SpinOrbitPotential",
]
