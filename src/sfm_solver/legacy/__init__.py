"""
Legacy and experimental modules for SFM Solver.

This package contains older solver implementations and experiments that are
no longer part of the main Tier 1 / Tier 1b / Tier 2 / Tier 2b pipelines.

They are kept for reference and potential future reuse, but are not imported
by the public sfm_solver API or any tier-completion tests.

Legacy Eigensolvers:
- LinearEigensolver: Direct eigenvalue solver (replaced by energy minimization)
- NonlinearEigensolver: Self-consistent eigenvalue solver with DIIS
- SFMAmplitudeSolver: Uses fitted scaling law m(n) = m₀ × n^a × exp(b×n)
  All replaced by SFMLeptonSolver in sfm_solver.eigensolver

Other Legacy:
- GrossPitaevskiiSolver: Early GP-based approach
- CoupledEigensolver: Full (r,σ) coupled eigenvalue problem
- LeptonMassFitter: Alpha fitting to match mass ratios
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
from sfm_solver.legacy.sfm_amplitude_solver import (  # noqa: F401
    SFMAmplitudeSolver,
    SFMAmplitudeState,
    solve_sfm_lepton_masses,
)
from sfm_solver.legacy.linear import LinearEigensolver  # noqa: F401
from sfm_solver.legacy.nonlinear import (  # noqa: F401
    NonlinearEigensolver,
    ConvergenceInfo,
    DIISMixer,
    AndersonMixer,
)

__all__ = [
    # Legacy eigensolvers (replaced by SFMLeptonSolver)
    "LinearEigensolver",
    "NonlinearEigensolver",
    "ConvergenceInfo",
    "DIISMixer",
    "AndersonMixer",
    # Legacy amplitude solver (replaced by SFMLeptonSolver)
    "SFMAmplitudeSolver",
    "SFMAmplitudeState",
    "solve_sfm_lepton_masses",
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


