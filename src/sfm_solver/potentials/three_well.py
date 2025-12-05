"""
Three-Well Potential for the S¹ subspace.

The fundamental potential structure in SFM with three minima
at σ = 0, 2π/3, 4π/3.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple


class ThreeWellPotential:
    """
    Three-well potential on S¹.
    
    V(σ) = V₀[1 - cos(3σ)] + V₁[1 - cos(6σ)]
    
    The primary term creates three wells at σ = 0, 2π/3, 4π/3.
    The secondary term adds a higher-frequency modulation.
    
    Attributes:
        V0: Primary well depth (GeV).
        V1: Secondary modulation depth (GeV).
    """
    
    def __init__(self, V0: float = 1.0, V1: float = 0.1):
        """
        Initialize the three-well potential.
        
        Args:
            V0: Primary well depth (GeV). Must be non-negative.
            V1: Secondary modulation (GeV). Must be non-negative and <= V0.
        """
        if V0 < 0:
            raise ValueError(f"V0 must be non-negative, got {V0}")
        if V1 < 0:
            raise ValueError(f"V1 must be non-negative, got {V1}")
        
        self.V0 = V0
        self.V1 = V1
    
    def __call__(
        self, 
        sigma: Union[float, NDArray[np.floating]]
    ) -> Union[float, NDArray[np.floating]]:
        """
        Evaluate the potential at given σ value(s).
        
        Args:
            sigma: Subspace coordinate(s) in radians.
            
        Returns:
            Potential value(s) in GeV.
        """
        return self.V0 * (1 - np.cos(3 * sigma)) + self.V1 * (1 - np.cos(6 * sigma))
    
    def derivative(
        self, 
        sigma: Union[float, NDArray[np.floating]]
    ) -> Union[float, NDArray[np.floating]]:
        """
        Compute the first derivative dV/dσ.
        
        dV/dσ = 3V₀ sin(3σ) + 6V₁ sin(6σ)
        
        Args:
            sigma: Subspace coordinate(s) in radians.
            
        Returns:
            Derivative value(s) in GeV.
        """
        return 3 * self.V0 * np.sin(3 * sigma) + 6 * self.V1 * np.sin(6 * sigma)
    
    def second_derivative(
        self, 
        sigma: Union[float, NDArray[np.floating]]
    ) -> Union[float, NDArray[np.floating]]:
        """
        Compute the second derivative d²V/dσ².
        
        d²V/dσ² = 9V₀ cos(3σ) + 36V₁ cos(6σ)
        
        Args:
            sigma: Subspace coordinate(s) in radians.
            
        Returns:
            Second derivative value(s) in GeV.
        """
        return 9 * self.V0 * np.cos(3 * sigma) + 36 * self.V1 * np.cos(6 * sigma)
    
    @property
    def well_positions(self) -> NDArray[np.floating]:
        """
        Return the positions of the three wells.
        
        The wells (minima of V) are located at σ = 0, 2π/3, 4π/3.
        """
        return np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    
    @property 
    def barrier_positions(self) -> NDArray[np.floating]:
        """
        Return the positions of the three barriers (maxima).
        
        The barriers are located at σ = π/3, π, 5π/3.
        """
        return np.array([np.pi / 3, np.pi, 5 * np.pi / 3])
    
    @property
    def well_depth(self) -> float:
        """
        Return the depth of the wells (minimum of V).
        
        At the well positions (σ = 0, 2π/3, 4π/3):
        - cos(3σ) = 1
        - cos(6σ) = 1
        
        So V_min = V₀(1-1) + V₁(1-1) = 0
        """
        return 0.0
    
    @property
    def barrier_height(self) -> float:
        """
        Return the height of the barriers (maximum of V).
        
        At the barrier positions (σ = π/3, π, 5π/3):
        - cos(3σ) = -1
        - cos(6σ) = 1 (for π/3 and 5π/3) or 1 (for π)
        
        For σ = π/3: cos(π) = -1, cos(2π) = 1
        For σ = π: cos(3π) = -1, cos(6π) = 1
        
        So V_max = V₀(1-(-1)) + V₁(1-1) = 2V₀
        
        Wait, need to recalculate for 6σ term:
        - σ = π/3: 6σ = 2π → cos(2π) = 1 → V₁ term = 0
        - σ = π: 6σ = 6π → cos(6π) = 1 → V₁ term = 0
        
        The pure barrier height from V₀ term is 2V₀.
        The V₁ term adds structure but doesn't increase the main barriers.
        """
        # Find the actual maximum numerically
        sigma_fine = np.linspace(0, 2 * np.pi, 1000)
        V_fine = self(sigma_fine)
        return float(np.max(V_fine))
    
    def well_curvature(self, well_index: int = 0) -> float:
        """
        Return the curvature (second derivative) at a well.
        
        At the well positions, d²V/dσ² = 9V₀ + 36V₁.
        This gives the "spring constant" for harmonic approximation.
        
        Args:
            well_index: Which well (0, 1, or 2).
            
        Returns:
            Curvature at the well in GeV.
        """
        sigma_well = self.well_positions[well_index]
        return self.second_derivative(sigma_well)
    
    def harmonic_frequency(self, well_index: int = 0, m_eff: float = 1.0) -> float:
        """
        Return the harmonic oscillator frequency at a well.
        
        For small oscillations, V ≈ V₀ + ½κ(σ - σ_well)²
        where κ = d²V/dσ².
        
        ω = √(κ/m_eff)
        
        Args:
            well_index: Which well (0, 1, or 2).
            m_eff: Effective mass parameter.
            
        Returns:
            Angular frequency ω in GeV (natural units).
        """
        kappa = self.well_curvature(well_index)
        return np.sqrt(kappa / m_eff)
    
    def tunneling_distance(self) -> float:
        """
        Return the distance between adjacent wells.
        
        The wells are separated by 2π/3 radians.
        """
        return 2 * np.pi / 3
    
    def on_grid(self, sigma: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Evaluate potential on a grid (alias for __call__).
        
        Args:
            sigma: Grid points.
            
        Returns:
            Potential values on the grid.
        """
        return self(sigma)
    
    def as_diagonal_matrix(
        self, 
        sigma: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Return the potential as a diagonal matrix.
        
        Useful for constructing Hamiltonian in matrix form.
        
        Args:
            sigma: Grid points.
            
        Returns:
            Diagonal matrix with V(σ) on diagonal.
        """
        return np.diag(self(sigma))
    
    def __repr__(self) -> str:
        return f"ThreeWellPotential(V0={self.V0:.4f} GeV, V1={self.V1:.4f} GeV)"

