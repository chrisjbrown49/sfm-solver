"""
Effective potentials including spin-orbit coupling.

These potentials modify the base three-well potential to account
for the interaction between the winding mode and spin.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Optional

from sfm_solver.potentials.three_well import ThreeWellPotential


class SpinOrbitPotential:
    """
    Spin-orbit coupling term for the effective potential.
    
    V_so = ∓λ k
    
    The sign depends on whether it's a positive or negative charge.
    This term couples the winding number k to the spin.
    
    Attributes:
        lambda_so: Spin-orbit coupling strength (GeV).
    """
    
    def __init__(self, lambda_so: float = 0.1):
        """
        Initialize the spin-orbit coupling.
        
        Args:
            lambda_so: Coupling strength in GeV.
        """
        if lambda_so < 0:
            raise ValueError(f"lambda_so must be non-negative, got {lambda_so}")
        self.lambda_so = lambda_so
    
    def __call__(
        self, 
        k: int, 
        charge_sign: int = 1
    ) -> float:
        """
        Calculate the spin-orbit energy shift.
        
        Args:
            k: Winding number (integer).
            charge_sign: +1 for positive charge, -1 for negative charge.
            
        Returns:
            Energy shift in GeV.
        """
        return -charge_sign * self.lambda_so * k
    
    def __repr__(self) -> str:
        return f"SpinOrbitPotential(lambda_so={self.lambda_so:.4f} GeV)"


class EffectivePotential:
    """
    Effective potential including spin-orbit coupling.
    
    V_eff(σ; k, ±) = V(σ) ∓ λ k
    
    where:
    - V(σ) is the base three-well potential
    - ± indicates the charge sign
    - λ is the spin-orbit coupling
    - k is the winding number
    
    For spin-1/2 particles, the two spinor components see different
    effective potentials.
    
    Attributes:
        base_potential: The underlying ThreeWellPotential.
        spin_orbit: The SpinOrbitPotential coupling.
    """
    
    def __init__(
        self,
        base_potential: Optional[ThreeWellPotential] = None,
        lambda_so: float = 0.1,
        V0: float = 1.0,
        V1: float = 0.1
    ):
        """
        Initialize the effective potential.
        
        Args:
            base_potential: Pre-constructed ThreeWellPotential, or None to create one.
            lambda_so: Spin-orbit coupling strength (GeV).
            V0: Primary well depth if creating new potential.
            V1: Secondary modulation if creating new potential.
        """
        if base_potential is not None:
            self.base_potential = base_potential
        else:
            self.base_potential = ThreeWellPotential(V0, V1)
        
        self.spin_orbit = SpinOrbitPotential(lambda_so)
    
    def __call__(
        self,
        sigma: Union[float, NDArray[np.floating]],
        k: int = 1,
        charge_sign: int = 1
    ) -> Union[float, NDArray[np.floating]]:
        """
        Evaluate the effective potential.
        
        Args:
            sigma: Subspace coordinate(s).
            k: Winding number.
            charge_sign: +1 for positive charge, -1 for negative.
            
        Returns:
            Effective potential value(s) in GeV.
        """
        V_base = self.base_potential(sigma)
        V_so = self.spin_orbit(k, charge_sign)
        return V_base + V_so
    
    def for_positive_charge(
        self,
        sigma: Union[float, NDArray[np.floating]],
        k: int = 1
    ) -> Union[float, NDArray[np.floating]]:
        """
        Effective potential for positive charge (e.g., positron, u quark).
        
        V_eff,+ = V(σ) - λk
        """
        return self(sigma, k, charge_sign=+1)
    
    def for_negative_charge(
        self,
        sigma: Union[float, NDArray[np.floating]],
        k: int = 1
    ) -> Union[float, NDArray[np.floating]]:
        """
        Effective potential for negative charge (e.g., electron, d quark).
        
        V_eff,- = V(σ) + λk
        """
        return self(sigma, k, charge_sign=-1)
    
    def spinor_potentials(
        self,
        sigma: Union[float, NDArray[np.floating]],
        k: int = 1
    ) -> Tuple[Union[float, NDArray], Union[float, NDArray]]:
        """
        Return both spinor component potentials.
        
        For a spin-1/2 particle, the two spinor components [ψ₊, ψ₋]
        see slightly different effective potentials due to spin-orbit coupling.
        
        Args:
            sigma: Subspace coordinate(s).
            k: Winding number.
            
        Returns:
            Tuple of (V_plus, V_minus) for spin-up and spin-down components.
        """
        V_base = self.base_potential(sigma)
        V_so = self.spin_orbit.lambda_so * k
        
        # Spin-up component sees V - λk
        V_plus = V_base - V_so
        # Spin-down component sees V + λk
        V_minus = V_base + V_so
        
        return V_plus, V_minus
    
    def energy_splitting(self, k: int = 1) -> float:
        """
        Calculate the spin-orbit energy splitting.
        
        ΔE = 2λk
        
        This is the energy difference between spin-up and spin-down states.
        
        Args:
            k: Winding number.
            
        Returns:
            Energy splitting in GeV.
        """
        return 2 * self.spin_orbit.lambda_so * abs(k)
    
    @property
    def lambda_so(self) -> float:
        """Get the spin-orbit coupling strength."""
        return self.spin_orbit.lambda_so
    
    @property
    def V0(self) -> float:
        """Get the primary well depth."""
        return self.base_potential.V0
    
    @property
    def V1(self) -> float:
        """Get the secondary modulation."""
        return self.base_potential.V1
    
    @property
    def well_positions(self) -> NDArray[np.floating]:
        """Get the well positions."""
        return self.base_potential.well_positions
    
    def __repr__(self) -> str:
        return (
            f"EffectivePotential(\n"
            f"  base={self.base_potential},\n"
            f"  {self.spin_orbit}\n"
            f")"
        )


class NonlinearEffectivePotential(EffectivePotential):
    """
    Effective potential with nonlinear self-interaction term.
    
    V_eff(σ) = V(σ) ∓ λk + g₁|χ(σ)|²
    
    The nonlinear term g₁|χ|² provides self-interaction that
    is important for confinement and mass generation.
    
    Attributes:
        g1: Nonlinear coupling constant.
    """
    
    def __init__(
        self,
        base_potential: Optional[ThreeWellPotential] = None,
        lambda_so: float = 0.1,
        g1: float = 0.1,
        V0: float = 1.0,
        V1: float = 0.1
    ):
        """
        Initialize the nonlinear effective potential.
        
        Args:
            base_potential: Pre-constructed ThreeWellPotential.
            lambda_so: Spin-orbit coupling strength.
            g1: Nonlinear |χ|² coupling constant.
            V0: Primary well depth.
            V1: Secondary modulation.
        """
        super().__init__(base_potential, lambda_so, V0, V1)
        
        if g1 < 0:
            raise ValueError(f"g1 must be non-negative, got {g1}")
        self.g1 = g1
    
    def with_density(
        self,
        sigma: Union[float, NDArray[np.floating]],
        chi_squared: Union[float, NDArray[np.floating]],
        k: int = 1,
        charge_sign: int = 1
    ) -> Union[float, NDArray[np.floating]]:
        """
        Evaluate potential with the nonlinear density term.
        
        V_eff = V(σ) ± λk + g₁|χ|²
        
        Args:
            sigma: Subspace coordinate(s).
            chi_squared: |χ(σ)|² density values.
            k: Winding number.
            charge_sign: +1 for positive, -1 for negative charge.
            
        Returns:
            Effective potential including nonlinear term.
        """
        V_linear = super().__call__(sigma, k, charge_sign)
        return V_linear + self.g1 * chi_squared
    
    def __repr__(self) -> str:
        return (
            f"NonlinearEffectivePotential(\n"
            f"  base={self.base_potential},\n"
            f"  {self.spin_orbit},\n"
            f"  g1={self.g1:.4f}\n"
            f")"
        )

