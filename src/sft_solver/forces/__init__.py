"""
Forces module for SFT Solver.

Contains electromagnetic force calculations based on circulation integrals.
"""

from sft_solver.forces.electromagnetic import (
    calculate_circulation,
    calculate_winding_number,
    calculate_envelope_asymmetry,
    calculate_em_energy,
    calculate_nonlinear_energy,
    calculate_total_interaction_energy,
    EMForceCalculator,
)

__all__ = [
    "calculate_circulation",
    "calculate_winding_number",
    "calculate_envelope_asymmetry",
    "calculate_em_energy",
    "calculate_nonlinear_energy",
    "calculate_total_interaction_energy",
    "EMForceCalculator",
]

