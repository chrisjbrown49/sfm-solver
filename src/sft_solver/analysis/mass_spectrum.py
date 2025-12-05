"""
Mass spectrum analysis for SFT solutions.

Calculates particle masses from wavefunction amplitudes and
compares with experimental values.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sft_solver.core.grid import SpectralGrid
from sft_solver.core.constants import (
    ELECTRON_MASS_GEV,
    MUON_MASS_GEV,
    TAU_MASS_GEV,
    MUON_ELECTRON_RATIO,
    TAU_ELECTRON_RATIO,
)


def calculate_mass_from_amplitude(
    A_squared: float,
    beta: float
) -> float:
    """
    Calculate mass from subspace amplitude squared.
    
    m = β A²_χ
    
    Args:
        A_squared: Integrated amplitude squared ∫|χ|² dσ.
        beta: Mass coupling constant (GeV).
        
    Returns:
        Mass in GeV.
    """
    return beta * A_squared


def calibrate_beta_from_electron(
    A_squared_electron: float,
    m_electron: float = ELECTRON_MASS_GEV
) -> float:
    """
    Calibrate β from the electron mass.
    
    β = m_e / A²_χ(electron)
    
    Args:
        A_squared_electron: Amplitude squared for electron solution.
        m_electron: Target electron mass in GeV.
        
    Returns:
        Calibrated β in GeV.
    """
    if A_squared_electron <= 0:
        raise ValueError("A² must be positive")
    return m_electron / A_squared_electron


def calculate_mass_ratios(
    amplitudes: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate mass ratios from amplitude ratios.
    
    Since m ∝ A², we have m₁/m₂ = A₁²/A₂².
    
    Args:
        amplitudes: Dictionary mapping particle names to A² values.
        
    Returns:
        Dictionary of mass ratios relative to electron.
    """
    if 'electron' not in amplitudes:
        raise ValueError("Must include 'electron' for ratio calculation")
    
    A_e = amplitudes['electron']
    if A_e <= 0:
        raise ValueError("Electron amplitude must be positive")
    
    ratios = {}
    for name, A_sq in amplitudes.items():
        ratios[f"{name}/electron"] = A_sq / A_e
    
    return ratios


@dataclass
class ParticleSolution:
    """Data class for a solved particle state."""
    name: str
    winding_number: int
    energy: float  # GeV
    amplitude_squared: float  # A²_χ
    wavefunction: NDArray[np.complexfloating]
    mass: Optional[float] = None  # GeV, computed from β


class MassSpectrum:
    """
    Mass spectrum analyzer for SFT solutions.
    
    Collects particle solutions, calibrates β, and compares
    computed masses with experimental values.
    
    Attributes:
        grid: SpectralGrid instance.
        beta: Mass coupling constant (can be calibrated).
        solutions: Dictionary of particle solutions.
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        beta: Optional[float] = None
    ):
        """
        Initialize the mass spectrum analyzer.
        
        Args:
            grid: SpectralGrid instance.
            beta: Initial β value, or None to calibrate later.
        """
        self.grid = grid
        self.beta = beta
        self.solutions: Dict[str, ParticleSolution] = {}
    
    def add_solution(
        self,
        name: str,
        winding_number: int,
        energy: float,
        wavefunction: NDArray[np.complexfloating]
    ) -> ParticleSolution:
        """
        Add a solved particle state.
        
        Args:
            name: Particle name (e.g., 'electron', 'muon').
            winding_number: Winding number k.
            energy: Eigenvalue energy in GeV.
            wavefunction: Solved wavefunction on grid.
            
        Returns:
            ParticleSolution object.
        """
        # Compute amplitude squared
        A_sq = np.real(self.grid.integrate(np.abs(wavefunction)**2))
        
        # Compute mass if β is set
        mass = self.beta * A_sq if self.beta is not None else None
        
        solution = ParticleSolution(
            name=name,
            winding_number=winding_number,
            energy=energy,
            amplitude_squared=A_sq,
            wavefunction=wavefunction,
            mass=mass
        )
        
        self.solutions[name] = solution
        return solution
    
    def calibrate_beta(
        self,
        reference: str = 'electron',
        target_mass: Optional[float] = None
    ) -> float:
        """
        Calibrate β from a reference particle.
        
        Args:
            reference: Name of reference particle.
            target_mass: Target mass in GeV, or None for experimental electron mass.
            
        Returns:
            Calibrated β value.
        """
        if reference not in self.solutions:
            raise ValueError(f"Reference particle '{reference}' not in solutions")
        
        if target_mass is None:
            target_mass = ELECTRON_MASS_GEV
        
        A_sq = self.solutions[reference].amplitude_squared
        self.beta = calibrate_beta_from_electron(A_sq, target_mass)
        
        # Update all masses
        for sol in self.solutions.values():
            sol.mass = self.beta * sol.amplitude_squared
        
        return self.beta
    
    def get_mass_ratios(self) -> Dict[str, float]:
        """
        Get mass ratios relative to electron.
        
        Returns:
            Dictionary of mass ratios.
        """
        if 'electron' not in self.solutions:
            raise ValueError("Need electron solution for ratios")
        
        A_e = self.solutions['electron'].amplitude_squared
        
        ratios = {}
        for name, sol in self.solutions.items():
            if name != 'electron':
                ratios[f"{name}/electron"] = sol.amplitude_squared / A_e
        
        return ratios
    
    def compare_with_experiment(self) -> Dict[str, Dict]:
        """
        Compare computed masses with experimental values.
        
        Returns:
            Dictionary with comparison for each particle.
        """
        experimental = {
            'electron': ELECTRON_MASS_GEV,
            'muon': MUON_MASS_GEV,
            'tau': TAU_MASS_GEV,
        }
        
        experimental_ratios = {
            'muon/electron': MUON_ELECTRON_RATIO,
            'tau/electron': TAU_ELECTRON_RATIO,
        }
        
        comparison = {}
        
        for name, sol in self.solutions.items():
            if sol.mass is not None:
                exp_mass = experimental.get(name)
                if exp_mass is not None:
                    error = (sol.mass - exp_mass) / exp_mass * 100
                    comparison[name] = {
                        'computed_mass': sol.mass,
                        'experimental_mass': exp_mass,
                        'error_percent': error,
                        'A_squared': sol.amplitude_squared,
                    }
        
        # Add ratio comparisons
        ratios = self.get_mass_ratios()
        for ratio_name, computed_ratio in ratios.items():
            exp_ratio = experimental_ratios.get(ratio_name)
            if exp_ratio is not None:
                error = (computed_ratio - exp_ratio) / exp_ratio * 100
                comparison[f"ratio_{ratio_name}"] = {
                    'computed': computed_ratio,
                    'experimental': exp_ratio,
                    'error_percent': error,
                }
        
        return comparison
    
    def summary(self) -> str:
        """
        Generate a summary string of the mass spectrum.
        
        Returns:
            Formatted summary string.
        """
        lines = ["=" * 60]
        lines.append("SFT Mass Spectrum Summary")
        lines.append("=" * 60)
        
        if self.beta is not None:
            lines.append(f"β = {self.beta:.6e} GeV")
        else:
            lines.append("β not calibrated")
        lines.append("")
        
        lines.append(f"{'Particle':<12} {'k':>4} {'A²':>12} {'Mass (GeV)':>14}")
        lines.append("-" * 46)
        
        for name, sol in sorted(self.solutions.items()):
            mass_str = f"{sol.mass:.6e}" if sol.mass else "N/A"
            lines.append(
                f"{name:<12} {sol.winding_number:>4} {sol.amplitude_squared:>12.6e} {mass_str:>14}"
            )
        
        lines.append("")
        
        # Add experimental comparison if available
        comparison = self.compare_with_experiment()
        if comparison:
            lines.append("Comparison with Experiment:")
            lines.append("-" * 46)
            for name, data in comparison.items():
                if 'error_percent' in data:
                    lines.append(f"{name}: error = {data['error_percent']:+.2f}%")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def validate_mass_hierarchy(self) -> Tuple[bool, str]:
        """
        Validate that the mass hierarchy is correct.
        
        Checks:
        - m_μ/m_e should be ~206.77
        - m_τ/m_e should be ~3477.15
        
        Returns:
            Tuple of (passed, message).
        """
        ratios = self.get_mass_ratios()
        
        messages = []
        passed = True
        
        if 'muon/electron' in ratios:
            computed = ratios['muon/electron']
            expected = MUON_ELECTRON_RATIO
            error = abs(computed - expected) / expected * 100
            
            if error < 10:  # Within 10%
                messages.append(f"✓ m_μ/m_e = {computed:.2f} (error: {error:.1f}%)")
            else:
                passed = False
                messages.append(f"✗ m_μ/m_e = {computed:.2f} (error: {error:.1f}%, expected ~{expected:.2f})")
        
        if 'tau/electron' in ratios:
            computed = ratios['tau/electron']
            expected = TAU_ELECTRON_RATIO
            error = abs(computed - expected) / expected * 100
            
            if error < 10:  # Within 10%
                messages.append(f"✓ m_τ/m_e = {computed:.2f} (error: {error:.1f}%)")
            else:
                passed = False
                messages.append(f"✗ m_τ/m_e = {computed:.2f} (error: {error:.1f}%, expected ~{expected:.2f})")
        
        return passed, "\n".join(messages)
    
    def to_dict(self) -> Dict:
        """
        Export spectrum data as dictionary.
        
        Returns:
            Dictionary with all spectrum data.
        """
        return {
            'beta': self.beta,
            'solutions': {
                name: {
                    'winding_number': sol.winding_number,
                    'energy': sol.energy,
                    'amplitude_squared': sol.amplitude_squared,
                    'mass': sol.mass,
                }
                for name, sol in self.solutions.items()
            },
            'ratios': self.get_mass_ratios() if 'electron' in self.solutions else {},
        }

