"""
Analysis module for SFM Solver.

Contains mass spectrum analysis and comparison tools.
"""

from sfm_solver.analysis.mass_spectrum import (
    MassSpectrum,
    calculate_mass_from_amplitude,
    calibrate_beta_from_electron,
    calculate_mass_ratios,
)

__all__ = [
    "MassSpectrum",
    "calculate_mass_from_amplitude",
    "calibrate_beta_from_electron",
    "calculate_mass_ratios",
]
