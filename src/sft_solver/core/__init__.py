"""
Core module for SFT Solver.

Contains physical constants, parameter definitions, and grid utilities.
"""

from sft_solver.core.constants import (
    HBAR,
    HBAR_EV,
    C,
    E_CHARGE,
    G_NEWTON,
    ALPHA_EM,
    GEV_TO_KG,
    ELECTRON_MASS_GEV,
    MUON_MASS_GEV,
    TAU_MASS_GEV,
)
from sft_solver.core.parameters import SFTParameters
from sft_solver.core.grid import SpectralGrid

__all__ = [
    # Constants
    "HBAR",
    "HBAR_EV",
    "C",
    "E_CHARGE",
    "G_NEWTON",
    "ALPHA_EM",
    "GEV_TO_KG",
    "ELECTRON_MASS_GEV",
    "MUON_MASS_GEV",
    "TAU_MASS_GEV",
    # Classes
    "SFTParameters",
    "SpectralGrid",
]

