"""
Color verification utilities for SFM Solver.

Provides tools to verify that the three-phase color structure {0, 2π/3, 4π/3}
emerges naturally from energy minimization in baryon systems.

Key physics:
- "Color" in SFM is the phase relationship between quarks in a baryon
- Color neutrality means: Σ exp(iφᵢ) = 0
- This is satisfied by phases {0, 2π/3, 4π/3} which sum to zero
- CRITICAL: Phases must EMERGE from dynamics, not be imposed as initial conditions
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid


@dataclass
class ColorVerificationResult:
    """Result of color phase verification."""
    is_color_neutral: bool
    phases: List[float]
    phase_differences: Tuple[float, float]  # (Δφ₁₂, Δφ₂₃)
    color_sum_magnitude: float
    emergence_verified: bool
    notes: str = ""


def extract_phases(
    chi1: NDArray[np.complexfloating],
    chi2: NDArray[np.complexfloating],
    chi3: NDArray[np.complexfloating],
    grid: SpectralGrid,
    method: str = 'peak'
) -> Tuple[float, float, float]:
    """
    Extract relative phases from three quark wavefunctions.
    
    Methods:
    - 'peak': Phase at wavefunction peak position
    - 'weighted': Weighted average phase using |χ|² as weight
    - 'fft': Dominant Fourier component phase
    
    Args:
        chi1, chi2, chi3: Three quark wavefunctions
        grid: SpectralGrid instance
        method: Phase extraction method
        
    Returns:
        Tuple of (φ₁, φ₂, φ₃) phases in radians
    """
    wavefunctions = [chi1, chi2, chi3]
    phases = []
    
    for chi in wavefunctions:
        if method == 'peak':
            # Find the peak of |χ|² and extract phase there
            density = np.abs(chi)**2
            peak_idx = np.argmax(density)
            phase = np.angle(chi[peak_idx])
            
        elif method == 'weighted':
            # Weighted average phase using density as weight
            density = np.abs(chi)**2
            norm = np.sum(density) * grid.dsigma
            if norm > 1e-14:
                # Use atan2 for proper phase unwrapping
                sin_avg = np.sum(density * np.sin(np.angle(chi))) * grid.dsigma / norm
                cos_avg = np.sum(density * np.cos(np.angle(chi))) * grid.dsigma / norm
                phase = np.arctan2(sin_avg, cos_avg)
            else:
                phase = 0.0
                
        elif method == 'fft':
            # Extract phase of dominant Fourier component
            chi_hat = np.fft.fft(chi)
            # Find dominant mode (ignoring DC)
            magnitudes = np.abs(chi_hat)
            magnitudes[0] = 0  # Ignore DC
            dominant_idx = np.argmax(magnitudes)
            phase = np.angle(chi_hat[dominant_idx])
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        phases.append(float(phase))
    
    return tuple(phases)


def verify_color_neutrality(
    chi1: NDArray[np.complexfloating],
    chi2: NDArray[np.complexfloating],
    chi3: NDArray[np.complexfloating],
    grid: SpectralGrid,
    tolerance: float = 0.01
) -> Tuple[bool, List[float], float, float]:
    """
    Verify that color neutrality is satisfied: Σ exp(iφᵢ) ≈ 0.
    
    Args:
        chi1, chi2, chi3: Three quark wavefunctions
        grid: SpectralGrid instance
        tolerance: Maximum allowed magnitude of color sum
        
    Returns:
        Tuple of:
        - is_color_neutral: True if |Σe^(iφᵢ)| < tolerance
        - phases: List of [φ₁, φ₂, φ₃]
        - delta_12: Phase difference φ₂ - φ₁ (mod 2π)
        - delta_23: Phase difference φ₃ - φ₂ (mod 2π)
    """
    # Extract phases
    phi1, phi2, phi3 = extract_phases(chi1, chi2, chi3, grid)
    
    # Calculate color sum
    color_sum = np.exp(1j * phi1) + np.exp(1j * phi2) + np.exp(1j * phi3)
    magnitude = np.abs(color_sum)
    
    # Check neutrality - convert to native Python bool
    is_neutral = bool(magnitude < tolerance)
    
    # Calculate phase differences (wrapped to [0, 2π])
    delta_12 = float((phi2 - phi1) % (2 * np.pi))
    delta_23 = float((phi3 - phi2) % (2 * np.pi))
    
    return is_neutral, [phi1, phi2, phi3], delta_12, delta_23


def verify_phase_emergence(
    initial_phases: Tuple[float, float, float],
    final_phases: Tuple[float, float, float],
    tolerance: float = 0.3
) -> Tuple[bool, str]:
    """
    Verify that three-phase structure emerged from dynamics.
    
    The three-phase structure {0, 2π/3, 4π/3} must emerge from energy
    minimization, NOT be imposed as initial conditions.
    
    This function checks that:
    1. Initial phases were NOT already in the three-phase configuration
    2. Final phases ARE in the three-phase configuration
    
    Args:
        initial_phases: (φ₁, φ₂, φ₃) at iteration start
        final_phases: (φ₁, φ₂, φ₃) at convergence
        tolerance: Phase difference tolerance in radians
        
    Returns:
        Tuple of:
        - emerged: True if structure emerged from non-trivial initial
        - message: Explanation of result
    """
    target_spacing = 2 * np.pi / 3  # Expected spacing
    
    # Check initial state - should NOT be in three-phase configuration
    init_d12 = (initial_phases[1] - initial_phases[0]) % (2 * np.pi)
    init_d23 = (initial_phases[2] - initial_phases[1]) % (2 * np.pi)
    
    init_is_target = (
        abs(init_d12 - target_spacing) < tolerance and
        abs(init_d23 - target_spacing) < tolerance
    )
    
    # Check final state - SHOULD be in three-phase configuration
    final_d12 = (final_phases[1] - final_phases[0]) % (2 * np.pi)
    final_d23 = (final_phases[2] - final_phases[1]) % (2 * np.pi)
    
    final_is_target = (
        abs(final_d12 - target_spacing) < tolerance and
        abs(final_d23 - target_spacing) < tolerance
    )
    
    # Also allow the reverse order (4π/3 spacing = -2π/3 = same as 4π/3)
    if not final_is_target:
        final_is_target = (
            abs(final_d12 - 2 * target_spacing) < tolerance and
            abs(final_d23 - 2 * target_spacing) < tolerance
        )
    
    if init_is_target:
        return False, "Initial phases already in three-phase config (test invalid)"
    
    if final_is_target:
        return True, f"Color phases emerged: Δφ₁₂={final_d12:.4f}, Δφ₂₃={final_d23:.4f}"
    
    return False, f"Color phases did NOT emerge: Δφ₁₂={final_d12:.4f}, Δφ₂₃={final_d23:.4f}"


class ColorVerification:
    """
    Utilities for verifying color structure emergence.
    
    Provides comprehensive tools for analyzing the three-quark
    phase relationships that constitute "color" in SFM.
    """
    
    def __init__(self, grid: SpectralGrid, tolerance: float = 0.01):
        """
        Initialize color verification utilities.
        
        Args:
            grid: SpectralGrid instance
            tolerance: Default tolerance for neutrality checks
        """
        self.grid = grid
        self.tolerance = tolerance
    
    def full_verification(
        self,
        chi1: NDArray[np.complexfloating],
        chi2: NDArray[np.complexfloating],
        chi3: NDArray[np.complexfloating],
        initial_phases: Optional[Tuple[float, float, float]] = None
    ) -> ColorVerificationResult:
        """
        Perform complete color verification.
        
        Args:
            chi1, chi2, chi3: Three quark wavefunctions
            initial_phases: Optional initial phases to verify emergence
            
        Returns:
            ColorVerificationResult with all verification data
        """
        # Check neutrality
        is_neutral, phases, d12, d23 = verify_color_neutrality(
            chi1, chi2, chi3, self.grid, self.tolerance
        )
        
        # Calculate color sum magnitude - convert to native Python float
        color_sum = sum(np.exp(1j * phi) for phi in phases)
        magnitude = float(np.abs(color_sum))
        
        # Check emergence if initial phases provided
        if initial_phases is not None:
            emerged, message = verify_phase_emergence(
                initial_phases, tuple(phases)
            )
        else:
            emerged = False
            message = "Initial phases not provided - emergence not verified"
        
        return ColorVerificationResult(
            is_color_neutral=bool(is_neutral),  # Ensure native bool
            phases=phases,
            phase_differences=(d12, d23),
            color_sum_magnitude=magnitude,
            emergence_verified=emerged,
            notes=message
        )
    
    @staticmethod
    def expected_phase_spacing() -> float:
        """Return the expected phase spacing 2π/3."""
        return 2 * np.pi / 3
    
    @staticmethod
    def calculate_color_sum(phases: List[float]) -> complex:
        """Calculate the color sum Σ exp(iφᵢ)."""
        return sum(np.exp(1j * phi) for phi in phases)
    
    def plot_phases(
        self, 
        phases: List[float],
        ax=None,
        title: str = "Color Phases"
    ):
        """
        Visualize color phases as vectors on unit circle.
        
        Args:
            phases: List of [φ₁, φ₂, φ₃] phases
            ax: Matplotlib axis (creates new if None)
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
        
        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']
        
        for phi, color, label in zip(phases, colors, labels):
            ax.arrow(phi, 0, 0, 0.9, head_width=0.1, color=color, 
                    linewidth=2, label=label)
        
        # Add color sum vector
        color_sum = self.calculate_color_sum(phases)
        ax.arrow(np.angle(color_sum), 0, 0, np.abs(color_sum), 
                head_width=0.1, color='black', linewidth=2, 
                linestyle='--', label=f'Sum (|S|={np.abs(color_sum):.3f})')
        
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        return ax

