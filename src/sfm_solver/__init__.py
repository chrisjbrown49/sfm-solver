"""
SFM Solver - Single-Field Model Two-Stage Solver

A first-principles solver for computing particle masses in the
Single-Field Model framework.

NEW TWO-STAGE ARCHITECTURE (RECOMMENDED):
==========================================
Stage 1: Solve for dimensionless wavefunction shape
Stage 2: Minimize energy over scale parameters (Delta_x, Delta_sigma, A)

Main Interface:
    from sfm_solver import UnifiedSFMSolver
    
    solver = UnifiedSFMSolver()  # Uses defaults from constants.json
    result = solver.solve_baryon(quark_windings=(5,5,-3))
    print(f"Predicted mass: {result.mass} GeV")

Components:
- UnifiedSFMSolver: Main interface combining both stages
- DimensionlessShapeSolver: Stage 1 (shape at unit scale)
- UniversalEnergyMinimizer: Stage 2 (optimize scales)
- SpatialCouplingBuilder: Build 4D structure from shape

Legacy solvers are available in sfm_solver.legacy but are deprecated.
"""

# Import core constants
from sfm_solver.core import (
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

# Import new two-stage solver architecture (MAIN INTERFACE)
from sfm_solver.core import (
    UnifiedSFMSolver,
    UnifiedSolverResult,
    DimensionlessShapeSolver,
    DimensionlessShapeResult,
    SpatialCouplingBuilder,
    SpatialState,
    UniversalEnergyMinimizer,
    EnergyMinimizationResult,
)

__version__ = "0.2.0"  # Updated for new architecture

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

