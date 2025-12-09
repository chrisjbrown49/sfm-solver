"""
Core module for SFM Solver.

Contains physical constants, parameter definitions, and grid utilities.

Global Constants:
- SFM_CONSTANTS: Single source of truth for β, L₀, κ (from Beautiful Equation)
- Use SFM_CONSTANTS.calibrate_from_electron() to set global β
"""

from sfm_solver.core.constants import (
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
from sfm_solver.core.parameters import SFMParameters
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import (
    SFMGlobalConstants,
    SFM_CONSTANTS,
    reset_global_constants,
)

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
    # Global SFM constants (single source of truth)
    "SFMGlobalConstants",
    "SFM_CONSTANTS",
    "reset_global_constants",
    # Classes
    "SFMParameters",
    "SpectralGrid",
]
