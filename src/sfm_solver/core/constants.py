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
    "g_internal": 0.003,  # FUNDAMENTAL: gravitational self-confinement in amplitude units
    "g1": 5000.0,  # dimensionless (nonlinear self-interaction)
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

# G_INTERNAL: FUNDAMENTAL gravitational self-confinement constant
# This controls self-confinement: Δx = 1/(G_internal × A⁶)^(1/3)
# G_internal is independent of beta - works directly with amplitude A
# β only converts amplitude to physical mass at the end: m = β × A²
G_INTERNAL: float = _LOADED_CONSTANTS.get("g_internal", 0.003)

# G1: nonlinear self-interaction coupling
G1: float = _LOADED_CONSTANTS["g1"]

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

