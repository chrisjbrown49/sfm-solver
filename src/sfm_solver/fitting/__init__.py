"""
Fitting module for SFM Solver.

Provides tools for fitting model parameters to experimental observables,
particularly the spacetime-subspace coupling constant α which determines
the lepton mass hierarchy.

Key components:
- alpha_fit: Fit coupling constant to reproduce m_μ/m_e ratio
- predict_tau: Predict tau mass from fitted parameters
"""

from sfm_solver.fitting.alpha_fit import (
    fit_alpha_to_mass_ratio,
    predict_tau_mass,
    LeptonMassFitter,
)

__all__ = [
    'fit_alpha_to_mass_ratio',
    'predict_tau_mass',
    'LeptonMassFitter',
]

