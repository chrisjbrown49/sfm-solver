"""
Legacy and experimental modules for SFM Solver.

This package contains older solver implementations and experiments that are
no longer part of the main Tier 1 / Tier 1b / Tier 2 / Tier 2b pipelines.

They are kept for reference and potential future reuse, but are not imported
by the public sfm_solver API or any tier-completion tests.
"""

from sfm_solver.legacy.gp_solver import (  # noqa: F401
    GrossPitaevskiiSolver,
    GPSolution,
    compute_mass_ratios_from_gp,
)
from sfm_solver.legacy.coupled_solver import (  # noqa: F401
    CoupledGrids,
    CoupledHamiltonian,
    CoupledEigensolver,
    CoupledSolution,
    compute_mass_ratios,
)
from sfm_solver.legacy.alpha_fit import (  # noqa: F401
    LeptonMassFitter,
    FittingResult,
    fit_alpha_to_mass_ratio,
    predict_tau_mass,
)
from sfm_solver.legacy.effective import (  # noqa: F401
    SpinOrbitPotential,
    EffectivePotential,
    NonlinearEffectivePotential,
)

__all__ = [
    # GP solver
    "GrossPitaevskiiSolver",
    "GPSolution",
    "compute_mass_ratios_from_gp",
    # Coupled solver
    "CoupledGrids",
    "CoupledHamiltonian",
    "CoupledEigensolver",
    "CoupledSolution",
    "compute_mass_ratios",
    # Alpha fitting
    "LeptonMassFitter",
    "FittingResult",
    "fit_alpha_to_mass_ratio",
    "predict_tau_mass",
    # Effective potentials
    "SpinOrbitPotential",
    "EffectivePotential",
    "NonlinearEffectivePotential",
]


