"""
Physical constants for SFM Solver.

All values from CODATA 2018/SI 2019 definitions where applicable.
"""

import numpy as np

# =============================================================================
# Single-Field Model Fundamental Constants (SI units)
# =============================================================================

# Speed of light (exact, SI definition)
C: float = 299_792_458  # m/s

# Planck constant (exact, SI definition)
H: float = 6.626_070_15e-34  # J·s

# Reduced Planck constant
HBAR: float = H / (2 * np.pi)  # J·s = 1.054571817...e-34 J·s

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

