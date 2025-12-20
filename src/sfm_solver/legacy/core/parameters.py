"""
SFM Parameters dataclass for configuring the solver.

The parameters define the properties of the three-well potential
and coupling constants for the Single-Field Model.

NOTE: g1 and g2 coupling constants now default to first-principles values
derived from the fine structure constant α via SFM_CONSTANTS.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from sfm_solver.core.constants import HBAR, C, ALPHA_EM


# Sentinel value for "use SFM_CONSTANTS default"
_USE_DEFAULT = -1.0


@dataclass
class SFMParameters:
    """
    Parameters for the Single-Field Model solver.
    
    The Beautiful Equation: β L₀ c = ℏ
    
    Attributes:
        beta: Mass coupling constant (GeV). Relates subspace amplitude to mass.
              Typical range: 40-100 GeV.
        V0: Primary well depth (GeV). Controls the depth of the three main wells.
            Typical value: ~1 GeV.
        V1: Secondary modulation (GeV). Controls the 6-fold modulation.
            Typical value: 0-0.5 GeV, should be <= V0.
        g1: Nonlinear coupling constant. Controls |χ|² self-interaction.
            Default: α (fine structure constant) ≈ 0.0073, derived from first principles.
        g2: Circulation coupling constant. Controls EM-like interactions.
            Default: α (for self-energy) ≈ 0.0073, derived from first principles.
        lambda_so: Spin-orbit coupling constant.
            Typical value: 0.01-0.5.
        R: Effective subspace "radius" parameter (dimensionless scaling).
            Default: 1.0.
        m_sigma: Effective mass parameter in subspace direction.
            Default: 1.0 (dimensionless).
        L0: Subspace radius (meters). Derived from β via Beautiful Equation.
            Typical value: ~3-5e-18 m for β ~40-100 GeV.
    """
    
    # Primary parameters
    beta: float = 50.0  # GeV - Mass coupling
    V0: float = 1.0     # GeV - Primary well depth
    V1: float = 0.1     # GeV - Secondary modulation
    
    # Coupling constants (dimensionless) - now derived from first principles!
    # Use sentinel value _USE_DEFAULT to indicate "use SFM_CONSTANTS"
    g1: float = _USE_DEFAULT  # Nonlinear |χ|² coupling → defaults to α
    g2: float = _USE_DEFAULT  # Circulation coupling (EM) → defaults to α
    lambda_so: float = 0.1  # Spin-orbit coupling
    
    # Subspace parameters
    R: float = 1.0      # Effective radius scaling
    m_sigma: float = 1.0  # Effective mass in σ direction
    
    # Derived parameter (computed from Beautiful Equation)
    L0: Optional[float] = field(default=None, init=False)
    
    def __post_init__(self):
        """Compute derived parameters after initialization."""
        # Apply first-principles defaults from SFM_CONSTANTS
        self._apply_defaults()
        self._compute_L0()
        self._validate()
    
    def _apply_defaults(self):
        """Apply first-principles defaults for g1 and g2 from SFM_CONSTANTS."""
        # Import here to avoid circular dependency
        from sfm_solver.core.sfm_global import SFM_CONSTANTS
        
        if self.g1 == _USE_DEFAULT:
            self.g1 = SFM_CONSTANTS.g1  # α from first principles
        if self.g2 == _USE_DEFAULT:
            self.g2 = SFM_CONSTANTS.g2_alpha  # α for self-energy from first principles
    
    def _compute_L0(self):
        """
        Compute subspace radius L0 from the Beautiful Equation.
        
        β L₀ c = ℏ  =>  L₀ = ℏ / (β c)
        
        With β in GeV, we need to convert:
        L₀ = ℏ(J·s) / (β(GeV) × GeV_to_J × c(m/s))
        """
        GeV_to_J = 1.602_176_634e-10  # J/GeV
        beta_J = self.beta * GeV_to_J  # Convert β to J (energy units)
        self.L0 = HBAR / (beta_J / C)  # L₀ = ℏc / β in meters
    
    def _validate(self):
        """Validate parameter values."""
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.V0 < 0:
            raise ValueError(f"V0 must be non-negative, got {self.V0}")
        if self.V1 < 0:
            raise ValueError(f"V1 must be non-negative, got {self.V1}")
        if self.V1 > self.V0:
            raise ValueError(f"V1 ({self.V1}) should be <= V0 ({self.V0})")
        if self.g1 < 0:
            raise ValueError(f"g1 must be non-negative, got {self.g1}")
        if self.g2 < 0:
            raise ValueError(f"g2 must be non-negative, got {self.g2}")
    
    @property
    def barrier_height(self) -> float:
        """
        Calculate the barrier height between wells.
        
        For V(σ) = V₀[1-cos(3σ)] + V₁[1-cos(6σ)],
        the maximum occurs at σ = π/3, 2π/3, π, etc.
        Maximum value ≈ 2V₀ + 2V₁ (when both terms at maximum)
        """
        return 2 * self.V0 + 2 * self.V1
    
    @property
    def well_positions(self) -> np.ndarray:
        """
        Return the positions of the three wells in σ.
        
        Wells are located at σ = 0, 2π/3, 4π/3.
        """
        return np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    
    def verify_beautiful_equation(self) -> float:
        """
        Verify the Beautiful Equation: beta * L0 = hbar * c.
        
        Returns the ratio (beta * L0) / (hbar * c) which should equal 1.
        """
        GeV_to_J = 1.602_176_634e-10
        beta_J = self.beta * GeV_to_J
        return beta_J * self.L0 / (HBAR * C)
    
    def mass_from_amplitude(self, A_squared: float) -> float:
        """
        Calculate mass from subspace amplitude squared.
        
        m = β A²_χ
        
        Args:
            A_squared: The integrated amplitude ∫|χ(σ)|² dσ
            
        Returns:
            Mass in GeV
        """
        return self.beta * A_squared
    
    def amplitude_from_mass(self, mass: float) -> float:
        """
        Calculate required amplitude squared from target mass.
        
        A²_χ = m / β
        
        Args:
            mass: Target mass in GeV
            
        Returns:
            Required integrated amplitude squared
        """
        return mass / self.beta
    
    def __repr__(self) -> str:
        return (
            f"SFMParameters(\n"
            f"  beta = {self.beta:.2f} GeV,\n"
            f"  V0 = {self.V0:.4f} GeV,\n"
            f"  V1 = {self.V1:.4f} GeV,\n"
            f"  g1 = {self.g1:.4f},\n"
            f"  g2 = {self.g2:.4f},\n"
            f"  lambda_so = {self.lambda_so:.4f},\n"
            f"  L0 = {self.L0:.3e} m\n"
            f")"
        )
