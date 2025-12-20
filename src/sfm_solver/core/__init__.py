"""
Core module for SFM Solver.

Contains physical constants, parameter definitions, grid utilities, and solvers.

RECOMMENDED: Use the new two-stage solver architecture:
- UnifiedSFMSolver: Main interface for solving particles
- DimensionlessShapeSolver: Stage 1 (shape)
- UniversalEnergyMinimizer: Stage 2 (scale)

Legacy solvers are available in sfm_solver.legacy.core but are deprecated.

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

# New two-stage solver architecture (MAIN INTERFACE)
from sfm_solver.core.unified_solver import UnifiedSFMSolver, UnifiedSolverResult
from sfm_solver.core.shape_solver import DimensionlessShapeSolver, DimensionlessShapeResult
from sfm_solver.core.spatial_coupling import SpatialCouplingBuilder, SpatialState
from sfm_solver.core.energy_minimizer import UniversalEnergyMinimizer, EnergyMinimizationResult

# Legacy utility classes (optional imports for backward compatibility)
try:
    from sfm_solver.core.parameters import SFMParameters
    from sfm_solver.core.grid import SpectralGrid
    from sfm_solver.core.sfm_global import (
        SFMGlobalConstants,
        SFM_CONSTANTS,
        reset_global_constants,
    )
    _LEGACY_AVAILABLE = True
except ImportError:
    # Legacy modules not available - this is OK for new architecture
    _LEGACY_AVAILABLE = False
    SFMParameters = None
    SpectralGrid = None
    SFMGlobalConstants = None
    SFM_CONSTANTS = None
    reset_global_constants = None

# Build __all__ list dynamically
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
    # New Two-Stage Solver Architecture (MAIN INTERFACE)
    "UnifiedSFMSolver",
    "UnifiedSolverResult",
    "DimensionlessShapeSolver",
    "DimensionlessShapeResult",
    "SpatialCouplingBuilder",
    "SpatialState",
    "UniversalEnergyMinimizer",
    "EnergyMinimizationResult",
]

# Add legacy classes if available
if _LEGACY_AVAILABLE:
    __all__.extend([
        "SFMGlobalConstants",
        "SFM_CONSTANTS",
        "reset_global_constants",
        "SFMParameters",
        "SpectralGrid",
    ])
