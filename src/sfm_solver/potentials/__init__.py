"""
Potentials module for SFM Solver.

Contains potential energy functions for the S¹ subspace.

The main production potential used in the Tier 1 / Tier 1b / Tier 2 / Tier 2b
pipelines is the periodic three-well potential.

More elaborate effective potentials with spin-orbit terms have been moved to
sfm_solver.legacy.effective and are no longer part of the main public API.
"""

from sfm_solver.potentials.three_well import ThreeWellPotential

__all__ = [
    "ThreeWellPotential",
]

