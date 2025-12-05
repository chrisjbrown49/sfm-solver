"""
SFM Solver - Single-Field Model Eigenvalue Solver

A numerical solver for computing eigenvalues and eigenfunctions
in the Single-Field Model framework.

Includes:
- Subspace-only eigensolvers (linear and nonlinear)
- Coupled subspace-spacetime solver for mass hierarchy
- Parameter fitting for coupling constant α
- Mass ratio predictions
"""

from sfm_solver.core import (
    HBAR,
    C,
    E_CHARGE,
    SFMParameters,
    SpectralGrid,
)
from sfm_solver.potentials import ThreeWellPotential, EffectivePotential
from sfm_solver.eigensolver import (
    LinearEigensolver, 
    NonlinearEigensolver,
    CoupledEigensolver,
    CoupledGrids,
    CoupledSolution,
    compute_mass_ratios,
)
from sfm_solver.spatial.radial import RadialGrid, RadialOperators
from sfm_solver.fitting.alpha_fit import (
    LeptonMassFitter,
    fit_alpha_to_mass_ratio,
    predict_tau_mass,
)
from sfm_solver.reporting import ResultsReporter

__version__ = "0.1.0"

__all__ = [
    # Constants
    "HBAR",
    "C",
    "E_CHARGE",
    # Core classes
    "SFMParameters",
    "SpectralGrid",
    # Spatial grid
    "RadialGrid",
    "RadialOperators",
    # Potentials
    "ThreeWellPotential",
    "EffectivePotential",
    # Eigensolvers
    "LinearEigensolver",
    "NonlinearEigensolver",
    "CoupledEigensolver",
    "CoupledGrids",
    "CoupledSolution",
    "compute_mass_ratios",
    # Fitting
    "LeptonMassFitter",
    "fit_alpha_to_mass_ratio",
    "predict_tau_mass",
    # Reporting
    "ResultsReporter",
]
