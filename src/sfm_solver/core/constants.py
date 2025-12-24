"""
Physical constants for SFM Solver.

All values from CODATA 2018/SI 2019 definitions where applicable.

Fundamental SFM constants (hbar, c, beta, alpha, kappa, g1) are loaded from
constants.json if available, otherwise default values are used.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

# =============================================================================
# Load Fundamental Constants from JSON
# =============================================================================

# Path to constants.json (same directory as this file)
_CONSTANTS_JSON_PATH = Path(__file__).parent / "constants.json"

# Default values for fundamental constants (used if constants.json is missing)
_DEFAULT_CONSTANTS: Dict[str, Any] = {
    "hbar": 1.0545718176461565e-34,  # J·s (reduced Planck constant)
    "c": 299792458,  # m/s (speed of light)
    "alpha": 10.5,  # GeV (spatial-subspace coupling)
    "G_5D": 1.0e7,  # GeV^-2 (FUNDAMENTAL: 5D gravitational constant)
    "g1": 5000.0,  # dimensionless (nonlinear self-interaction)
    "g2": 0.035,  # dimensionless (circulation/EM coupling)
    "lambda_so": 0.2,  # dimensionless (spin-orbit coupling strength)
    "V0": 1.0,  # GeV (primary three-well potential depth)
    "V1": 0.0,  # GeV (secondary six-well potential depth)
    "N_r_grid": 100,  # Grid points for radial wavefunction
    "N_sigma_grid": 64,  # Grid points for subspace wavefunction
    "r_max_fm": 50.0,  # Maximum radius in fm
}


def load_constants_from_json() -> Dict[str, Any]:
    """
    Load fundamental constants from constants.json.
    
    If the file doesn't exist or is invalid, returns default values.
    
    Returns:
        Dictionary with constant names as keys and values.
    """
    if _CONSTANTS_JSON_PATH.exists():
        try:
            with open(_CONSTANTS_JSON_PATH, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                result = _DEFAULT_CONSTANTS.copy()
                result.update(loaded)
                return result
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load constants.json: {e}. Using defaults.")
            return _DEFAULT_CONSTANTS.copy()
    else:
        return _DEFAULT_CONSTANTS.copy()


def save_constants_to_json(constants: Dict[str, Any]) -> None:
    """
    Save fundamental constants to constants.json.
    
    Args:
        constants: Dictionary with constant names and values.
                   Should contain: hbar, c, beta, alpha, g_eff, g1
                   Note: g_eff is the fundamental gravitational coupling.
                   kappa is derived as kappa = g_eff × beta².
    """
    with open(_CONSTANTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(constants, f, indent=4)


def get_constants_json_path() -> Path:
    """Return the path to the constants.json file."""
    return _CONSTANTS_JSON_PATH


# Load constants at module import time
_LOADED_CONSTANTS = load_constants_from_json()

# =============================================================================
# SFM Fundamental Constants (loaded from JSON or defaults)
# =============================================================================

# Reduced Planck constant (from JSON or default)
HBAR_LOADED: float = _LOADED_CONSTANTS["hbar"]

# Speed of light (from JSON or default)
C_LOADED: float = _LOADED_CONSTANTS["c"]

# Beta: mass-amplitude coupling constant
# Note: Beta is now derived from electron mass after solving: β = m_e / A_e²
# This default value (100 GeV) is for backward compatibility only.
# The actual physical β is ~0.0005 GeV to give m_e = 0.511 MeV
BETA: float = _LOADED_CONSTANTS.get("beta", 100.0)

# Alpha: spatial-subspace coupling strength
ALPHA: float = _LOADED_CONSTANTS["alpha"]

# G_5D: FUNDAMENTAL 5D gravitational constant (in GeV^-2)
# This is the fundamental gravitational coupling in 5D spacetime.
# Related to mass scale via: beta = G_5D * c
# Controls spatial confinement energy and curvature energy.
# β only converts amplitude to physical mass at the end: m = β × A²
G_5D_CONSTANT: float = _LOADED_CONSTANTS.get("G_5D", 1.0e7)

# G1: nonlinear self-interaction coupling
G1: float = _LOADED_CONSTANTS["g1"]

# =============================================================================
# G2: Circulation/Electromagnetic Coupling
# =============================================================================
# Status: CALIBRATED (theoretically derivable - work remaining)
#
# G2 controls the electromagnetic self-energy through the circulation integral:
#     E_EM = g2 × |J_normalized|²
#
# where J = ∫ χ* ∂χ/∂σ dσ is the circulation (related to charge/winding).
#
# THEORETICAL BASIS:
#     The SFM predicts that g2 should emerge from the 5D geometry and be
#     related to the fine structure constant α_EM ≈ 1/137. Specifically:
#         g2 ~ α_EM × (characteristic scale factors)
#
#     Current value calibrated to match proton-neutron mass splitting.
#
# PATH TO FIRST-PRINCIPLES:
#     Derive g2 from: (1) 5D metric structure, (2) compactification geometry,
#     (3) relationship to α_EM and other coupling constants.
# =============================================================================
G2: float = _LOADED_CONSTANTS.get("g2", 0.035)

# =============================================================================
# LAMBDA_SO: Spin-Orbit Coupling Strength
# =============================================================================
# Status: CALIBRATED (theoretically derivable - work remaining)
#
# Lambda_so controls the spin-orbit coupling in the SFM Hamiltonian:
#     H_so = λ (∂/∂σ ⊗ σ_z)
#
# This creates different effective potentials for particles with different
# winding numbers k (which determine charge):
#     V_eff = V(σ) ∓ λk
#
# The ± depends on spin orientation. This shifts envelope centers and widths
# for different quarks (u: k=+5, d: k=-3), creating mass splitting.
#
# THEORETICAL BASIS:
#     Lambda_so has dimensions of action ([λ] = ℏ). From dimensional analysis
#     and the SFM structure, we expect:
#         λ ~ ℏ × (geometric factors)
#
#     The natural k-scaling is automatic: the effect is λ×k, so higher
#     winding numbers (larger |k|) have proportionally larger effects.
#
# PATH TO FIRST-PRINCIPLES:
#     Derive λ from: (1) 5D spin-geometry coupling, (2) relationship to
#     fine structure constant α_EM, (3) constraints from neutrino physics
#     (neutrinos have λ_ν ~ 10⁻⁹ λ_charged).
# =============================================================================
LAMBDA_SO: float = _LOADED_CONSTANTS.get("lambda_so", 0.2)

# =============================================================================
# V0: Primary Three-Well Potential Depth
# =============================================================================
# Status: FIXED (geometric origin from 5D compactification)
#
# V0 sets the depth of the primary three-well potential in the subspace:
#     V(σ) = V0 × (1 - cos(3σ)) + V1 × (1 - cos(6σ))
#
# The cos(3σ) term creates three minima at σ = 0, 2π/3, 4π/3 corresponding
# to the three color charges (red, green, blue). These are the locations
# where quarks can be localized.
#
# THEORETICAL BASIS:
#     V0 emerges from the 5D geometry and should be related to β through
#     the compactification scale. Currently fixed at 1.0 GeV.
#
# PATH TO FIRST-PRINCIPLES:
#     Derive V0 from: (1) 5D metric structure, (2) relationship to β,
#     (3) constraints from QCD color confinement.
# =============================================================================
V0: float = _LOADED_CONSTANTS.get("V0", 1.0)

# =============================================================================
# V1: Secondary Six-Well Potential Depth
# =============================================================================
# Status: EXPLORATORY (geometric origin unclear)
#
# V1 adds a secondary six-well structure to the subspace potential:
#     V(σ) = V0 × (1 - cos(3σ)) + V1 × (1 - cos(6σ))
#
# The cos(6σ) term creates six secondary minima, potentially allowing
# more fine structure in quark localization. Currently set to 0.0.
#
# THEORETICAL BASIS:
#     The need for V1 is uncertain. It may arise from:
#     (1) Higher-order corrections to the 5D potential
#     (2) Quantum corrections from quark interactions
#     (3) May not be needed at all if primary wells suffice
#
# PATH TO FIRST-PRINCIPLES:
#     Determine if V1 is required from first principles or if it's a
#     phenomenological artifact that should be eliminated.
# =============================================================================
V1: float = _LOADED_CONSTANTS.get("V1", 0.0)

# =============================================================================
# 5D Field Grid Configuration
# =============================================================================
# Grid resolution for computing energies from full 5D wavefunction Ψ(r,σ)

# Number of radial grid points
N_R_GRID: int = _LOADED_CONSTANTS.get("N_r_grid", 100)

# Number of subspace grid points (periodic dimension σ ∈ [0, 2π])
N_SIGMA_GRID: int = _LOADED_CONSTANTS.get("N_sigma_grid", 64)

# Maximum radius for radial grid in fm
R_MAX_FM: float = _LOADED_CONSTANTS.get("r_max_fm", 50.0)

# =============================================================================
# Single-Field Model Fundamental Constants (SI units)
# =============================================================================

# Speed of light (loaded from JSON or exact SI definition)
C: float = C_LOADED  # m/s

# Planck constant (exact, SI definition)
H: float = 6.626_070_15e-34  # J·s

# Reduced Planck constant (loaded from JSON or derived from H)
HBAR: float = HBAR_LOADED  # J·s = 1.054571817...e-34 J·s

# Reduced Planck constant in eV·s
HBAR_EV: float = 6.582_119_569e-16  # eV·s

# Beta: subspace-spacetime coupling constant

# L0: subspace radius

# IMPORTANT:
# SFM defines Beta and L0 as fundamental constants, not derived.  Challenge is to solve.
# the field equations and adjust Beta and L0 so that predictions match experimental values.

# =============================================================================
# Standard Model Fundamental Constants (SI units)
# =============================================================================

# Elementary charge (exact, SI definition)
E_CHARGE: float = 1.602_176_634e-19  # C

# Vacuum permittivity (exact, SI definition)
# ε₀ = 1/(μ₀c²) where μ₀ = 4π×10⁻⁷ H/m exactly
EPSILON_0: float = 8.854_187_8128e-12  # F/m (farads per meter)

# Vacuum permeability (exact, SI definition)
MU_0: float = 1.256_637_062_12e-6  # H/m (henries per meter)

# Gravitational constant (CODATA 2018)
G_NEWTON: float = 6.674_30e-11  # m³/(kg·s²)

# Fine structure constant at low energy α(m_e) (CODATA 2018)
# α = e²/(4πε₀ℏc) ≈ 1/137.036
ALPHA_EM: float = 1.0 / 137.035_999_084
ALPHA_EM_INVERSE: float = 137.035_999_084  # 1/α for convenience

# Fermi coupling constant G_F/(ℏc)³
G_FERMI: float = 1.166_378_7e-5  # GeV⁻²

# =============================================================================
# Unit Conversions
# =============================================================================

# GeV to kg conversion
GEV_TO_JOULE: float = 1.602_176_634e-10  # J/GeV
GEV_TO_KG: float = GEV_TO_JOULE / (C * C)  # kg/GeV

# eV to Joule
EV_TO_JOULE: float = E_CHARGE  # J/eV

# Planck mass
M_PLANCK: float = np.sqrt(HBAR * C / G_NEWTON)  # kg
M_PLANCK_GEV: float = M_PLANCK / GEV_TO_KG  # GeV ≈ 1.22e19 GeV

# =============================================================================
# Particle Masses (Directly Measured - PDG 2024)
# =============================================================================

# Lepton masses in GeV
ELECTRON_MASS_GEV: float = 0.000_510_998_950_00  # GeV (0.30 ppb precision)
MUON_MASS_GEV: float = 0.105_658_375_5  # GeV (0.22 ppm precision)
TAU_MASS_GEV: float = 1.776_86  # GeV (68 ppm precision)

# Lepton mass ratios (critical tests for SFM)
MUON_ELECTRON_RATIO: float = 206.768_283_0  # m_μ/m_e
TAU_MUON_RATIO: float = 16.816_7  # m_τ/m_μ
TAU_ELECTRON_RATIO: float = 3477.15  # m_τ/m_e

# Gauge boson masses in GeV
W_MASS_GEV: float = 80.369_1  # GeV (29 ppm precision)
Z_MASS_GEV: float = 91.188_0  # GeV (22 ppm precision)
HIGGS_MASS_GEV: float = 125.09  # GeV

# Hadron masses in GeV
PROTON_MASS_GEV: float = 0.938_272_088_16  # GeV (0.3 ppb precision)
NEUTRON_MASS_GEV: float = 0.939_565_421_99  # GeV (0.6 ppb precision)
PION_CHARGED_MASS_GEV: float = 0.139_570_39  # GeV
PION_NEUTRAL_MASS_GEV: float = 0.134_976_8  # GeV

# Quarkonium masses in GeV (for Tier 2 validation)
JPSI_MASS_GEV: float = 3.096_900  # J/ψ (cc̄) - charmonium ground state
UPSILON_1S_MASS_GEV: float = 9.460_30  # Υ(1S) (bb̄) - bottomonium ground state (future)

# =============================================================================
# Natural Units Option
# =============================================================================
# For calculations in natural units where ℏ = c = 1
# Energy is measured in GeV, length in GeV⁻¹, time in GeV⁻¹

class NaturalUnits:
    """Natural units where ℏ = c = 1."""
    
    HBAR: float = 1.0
    C: float = 1.0
    
    # Conversion factors
    GEV_TO_FM: float = 0.197_326_980  # 1 GeV⁻¹ ≈ 0.197 fm (ℏc ≈ 197 MeV·fm)
    FM_TO_GEV_INV: float = 1.0 / GEV_TO_FM
    
    @staticmethod
    def length_to_gev_inv(length_meters: float) -> float:
        """Convert length in meters to GeV⁻¹."""
        return length_meters * C / HBAR_EV / 1e9  # meters to GeV⁻¹
    
    @staticmethod
    def gev_inv_to_length(gev_inv: float) -> float:
        """Convert GeV⁻¹ to length in meters."""
        return gev_inv * HBAR_EV * 1e9 / C

