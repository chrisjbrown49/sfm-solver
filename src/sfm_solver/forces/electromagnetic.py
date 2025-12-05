"""
Electromagnetic force calculations.

The EM force in SFM emerges from the circulation term:
    Ĥ_circ = g₂ |∫ χ* ∂χ/∂σ dσ|²

For two particles, like charges (same winding sign) lead to
energy penalty → repulsion, while opposite charges (opposite 
winding signs) have reduced penalty → attraction.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict

from sfm_solver.core.grid import SpectralGrid


def calculate_circulation(
    chi: NDArray[np.complexfloating],
    grid: SpectralGrid
) -> complex:
    """
    Calculate the circulation integral for a wavefunction.
    
    J = ∫₀^(2π) χ*(σ) ∂χ/∂σ dσ
    
    For χ(σ) = A exp(ikσ) f(σ), this gives approximately ik times the norm.
    
    Args:
        chi: Wavefunction on the grid.
        grid: SpectralGrid instance.
        
    Returns:
        Complex circulation integral value.
    """
    dchi = grid.first_derivative(chi)
    return grid.integrate(np.conj(chi) * dchi)


def calculate_winding_number(
    chi: NDArray[np.complexfloating],
    grid: SpectralGrid
) -> float:
    """
    Extract the winding number from the circulation integral.
    
    For a normalized wavefunction with ∫|χ|² dσ = 2π,
    k = Im[J] / (2π)
    
    More generally:
    k = Im[J] / ∫|χ|² dσ
    
    Args:
        chi: Wavefunction on the grid.
        grid: SpectralGrid instance.
        
    Returns:
        Approximate winding number (should be close to integer).
    """
    J = calculate_circulation(chi, grid)
    norm_sq = np.real(grid.integrate(np.abs(chi)**2))
    
    if norm_sq < 1e-14:
        return 0.0
    
    return np.imag(J) / norm_sq


def calculate_envelope_asymmetry(
    chi_plus: NDArray[np.complexfloating],
    chi_minus: NDArray[np.complexfloating],
    grid: SpectralGrid
) -> Tuple[float, float]:
    """
    Calculate envelope asymmetry parameters η and δ.
    
    For two particles with opposite charges (winding numbers k and -k),
    the envelopes may differ due to spin-orbit effects.
    
    η = (A₊ - A₋) / (A₊ + A₋)  (amplitude asymmetry)
    δ = σ₊ - σ₋                 (position offset)
    
    Args:
        chi_plus: Wavefunction for positive charge.
        chi_minus: Wavefunction for negative charge.
        grid: SpectralGrid instance.
        
    Returns:
        Tuple of (η, δ) asymmetry parameters.
    """
    # Compute amplitudes
    A_plus = np.sqrt(np.real(grid.integrate(np.abs(chi_plus)**2)))
    A_minus = np.sqrt(np.real(grid.integrate(np.abs(chi_minus)**2)))
    
    # Amplitude asymmetry
    if A_plus + A_minus > 1e-10:
        eta = (A_plus - A_minus) / (A_plus + A_minus)
    else:
        eta = 0.0
    
    # Position asymmetry: find centers of mass of |χ|²
    density_plus = np.abs(chi_plus)**2
    density_minus = np.abs(chi_minus)**2
    
    # Weighted average position (on circle, need to handle periodicity)
    # Use circular mean: θ_mean = atan2(Σ sin(θ), Σ cos(θ))
    cos_sigma = np.cos(grid.sigma)
    sin_sigma = np.sin(grid.sigma)
    
    norm_plus = np.sum(density_plus) * grid.dsigma
    if norm_plus > 1e-10:
        cos_mean_plus = np.sum(density_plus * cos_sigma) * grid.dsigma / norm_plus
        sin_mean_plus = np.sum(density_plus * sin_sigma) * grid.dsigma / norm_plus
        sigma_plus = np.arctan2(sin_mean_plus, cos_mean_plus)
    else:
        sigma_plus = 0.0
    
    norm_minus = np.sum(density_minus) * grid.dsigma
    if norm_minus > 1e-10:
        cos_mean_minus = np.sum(density_minus * cos_sigma) * grid.dsigma / norm_minus
        sin_mean_minus = np.sum(density_minus * sin_sigma) * grid.dsigma / norm_minus
        sigma_minus = np.arctan2(sin_mean_minus, cos_mean_minus)
    else:
        sigma_minus = 0.0
    
    # Position difference (wrapped to [-π, π])
    delta = sigma_plus - sigma_minus
    if delta > np.pi:
        delta -= 2 * np.pi
    elif delta < -np.pi:
        delta += 2 * np.pi
    
    return float(eta), float(delta)


def calculate_em_energy(
    chi1: NDArray[np.complexfloating],
    chi2: NDArray[np.complexfloating],
    grid: SpectralGrid,
    g2: float
) -> float:
    """
    Calculate the electromagnetic energy from circulation terms.
    
    For two particles:
    E_EM = g₂ |J₁ + J₂|²
    
    where J_i = ∫ χᵢ* ∂χᵢ/∂σ dσ.
    
    Like charges (same winding): J₁ + J₂ adds up → large energy → repulsion
    Opposite charges (opposite winding): J₁ + J₂ cancels → small energy → attraction
    
    Args:
        chi1: First wavefunction.
        chi2: Second wavefunction.
        grid: SpectralGrid instance.
        g2: Circulation coupling constant.
        
    Returns:
        EM energy in natural units.
    """
    J1 = calculate_circulation(chi1, grid)
    J2 = calculate_circulation(chi2, grid)
    
    total_circulation = J1 + J2
    
    return g2 * np.abs(total_circulation)**2


def calculate_nonlinear_energy(
    chi1: NDArray[np.complexfloating],
    chi2: NDArray[np.complexfloating],
    grid: SpectralGrid,
    g1: float
) -> float:
    """
    Calculate nonlinear interaction energy from density overlap.
    
    E_nl = g₁ ∫ |χ₁|² |χ₂|² dσ
    
    This represents the short-range contact interaction between particles.
    
    Args:
        chi1: First wavefunction.
        chi2: Second wavefunction.
        grid: SpectralGrid instance.
        g1: Nonlinear coupling constant.
        
    Returns:
        Nonlinear interaction energy.
    """
    density1 = np.abs(chi1)**2
    density2 = np.abs(chi2)**2
    
    overlap = grid.integrate(density1 * density2)
    
    return g1 * np.real(overlap)


def calculate_total_interaction_energy(
    chi1: NDArray[np.complexfloating],
    chi2: NDArray[np.complexfloating],
    grid: SpectralGrid,
    g1: float,
    g2: float
) -> Dict[str, float]:
    """
    Calculate total interaction energy between two particles.
    
    E_int = E_nl + E_EM
         = g₁ ∫|χ₁|²|χ₂|² dσ + g₂ |J₁ + J₂|²
    
    Args:
        chi1: First wavefunction.
        chi2: Second wavefunction.
        grid: SpectralGrid instance.
        g1: Nonlinear coupling constant.
        g2: Circulation coupling constant.
        
    Returns:
        Dictionary with 'nonlinear', 'em', and 'total' energies.
    """
    E_nl = calculate_nonlinear_energy(chi1, chi2, grid, g1)
    E_em = calculate_em_energy(chi1, chi2, grid, g2)
    
    return {
        'nonlinear': E_nl,
        'em': E_em,
        'total': E_nl + E_em
    }


class EMForceCalculator:
    """
    Calculator for electromagnetic forces between particles.
    
    Provides methods for analyzing EM-like interactions in SFT,
    including force directions, energy landscapes, and charge effects.
    
    Attributes:
        grid: SpectralGrid instance.
        g1: Nonlinear coupling constant.
        g2: Circulation (EM) coupling constant.
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        g1: float = 0.1,
        g2: float = 0.1
    ):
        """
        Initialize the EM force calculator.
        
        Args:
            grid: SpectralGrid instance.
            g1: Nonlinear coupling constant.
            g2: Circulation coupling constant.
        """
        self.grid = grid
        self.g1 = g1
        self.g2 = g2
    
    def circulation_energy(
        self,
        wavefunctions: list,
    ) -> float:
        """
        Calculate total circulation energy for multiple particles.
        
        E_circ = g₂ |Σᵢ Jᵢ|²
        
        Args:
            wavefunctions: List of wavefunctions.
            
        Returns:
            Total circulation energy.
        """
        total_J = 0.0j
        for chi in wavefunctions:
            total_J += calculate_circulation(chi, self.grid)
        
        return self.g2 * np.abs(total_J)**2
    
    def total_winding(self, wavefunctions: list) -> float:
        """
        Calculate total winding number (net charge).
        
        Args:
            wavefunctions: List of wavefunctions.
            
        Returns:
            Sum of winding numbers.
        """
        total_k = 0.0
        for chi in wavefunctions:
            total_k += calculate_winding_number(chi, self.grid)
        return total_k
    
    def charge_from_winding(self, k: int) -> float:
        """
        Calculate electric charge from winding number.
        
        Q = ±e/k (in units where e = 1)
        
        Args:
            k: Winding number.
            
        Returns:
            Electric charge in units of e.
        """
        if k == 0:
            return 0.0
        return 1.0 / k
    
    def force_type(
        self,
        chi1: NDArray[np.complexfloating],
        chi2: NDArray[np.complexfloating]
    ) -> str:
        """
        Determine if force is attractive or repulsive.
        
        Like charges (same winding sign) → repulsive
        Opposite charges (opposite winding sign) → attractive
        
        Args:
            chi1: First wavefunction.
            chi2: Second wavefunction.
            
        Returns:
            'repulsive', 'attractive', or 'neutral'.
        """
        k1 = calculate_winding_number(chi1, self.grid)
        k2 = calculate_winding_number(chi2, self.grid)
        
        if abs(k1) < 0.1 or abs(k2) < 0.1:
            return 'neutral'
        
        if k1 * k2 > 0:
            return 'repulsive'
        else:
            return 'attractive'
    
    def energy_vs_separation(
        self,
        chi_template: NDArray[np.complexfloating],
        k1: int,
        k2: int,
        n_points: int = 50
    ) -> Tuple[NDArray, NDArray]:
        """
        Calculate interaction energy as a function of separation.
        
        Creates two localized modes at varying separations and
        computes the total interaction energy.
        
        Args:
            chi_template: Template wavefunction for envelope shape.
            k1: Winding number of first particle.
            k2: Winding number of second particle.
            n_points: Number of separation points.
            
        Returns:
            Tuple of (separations, energies) arrays.
        """
        # Get envelope from template
        envelope = np.abs(chi_template)
        
        separations = np.linspace(0.1, np.pi, n_points)
        energies = np.zeros(n_points)
        
        for i, sep in enumerate(separations):
            # Create two localized modes at positions 0 and sep
            chi1 = envelope * np.exp(1j * k1 * self.grid.sigma)
            
            # Shift envelope for second particle
            envelope2 = np.roll(envelope, int(sep / self.grid.dsigma))
            chi2 = envelope2 * np.exp(1j * k2 * self.grid.sigma)
            
            # Normalize
            chi1 = self.grid.normalize(chi1)
            chi2 = self.grid.normalize(chi2)
            
            # Calculate energy
            E = calculate_total_interaction_energy(
                chi1, chi2, self.grid, self.g1, self.g2
            )
            energies[i] = E['total']
        
        return separations, energies
    
    def coulomb_like_energy(
        self,
        k1: int,
        k2: int,
        norm1: float = 1.0,
        norm2: float = 1.0
    ) -> float:
        """
        Calculate approximate Coulomb-like energy.
        
        For well-separated particles:
        E ~ g₂ (k₁ × norm₁ + k₂ × norm₂)²
        
        Like charges (same sign k): adds up → large energy
        Opposite charges (opposite sign k): cancels → small energy
        
        Args:
            k1: Winding number of first particle.
            k2: Winding number of second particle.
            norm1: Amplitude norm of first particle.
            norm2: Amplitude norm of second particle.
            
        Returns:
            Approximate EM energy.
        """
        # Circulation is approximately i*k*norm² for each particle
        J1 = 1j * k1 * norm1**2
        J2 = 1j * k2 * norm2**2
        
        return self.g2 * np.abs(J1 + J2)**2
    
    def analyze_two_particle(
        self,
        chi1: NDArray[np.complexfloating],
        chi2: NDArray[np.complexfloating]
    ) -> Dict:
        """
        Full analysis of two-particle interaction.
        
        Args:
            chi1: First wavefunction.
            chi2: Second wavefunction.
            
        Returns:
            Dictionary with comprehensive analysis.
        """
        k1 = calculate_winding_number(chi1, self.grid)
        k2 = calculate_winding_number(chi2, self.grid)
        
        J1 = calculate_circulation(chi1, self.grid)
        J2 = calculate_circulation(chi2, self.grid)
        
        energies = calculate_total_interaction_energy(
            chi1, chi2, self.grid, self.g1, self.g2
        )
        
        eta, delta = calculate_envelope_asymmetry(chi1, chi2, self.grid)
        
        norm1 = self.grid.norm(chi1)
        norm2 = self.grid.norm(chi2)
        
        return {
            'winding_numbers': (k1, k2),
            'circulations': (J1, J2),
            'total_circulation': J1 + J2,
            'charges': (self.charge_from_winding(round(k1)), 
                       self.charge_from_winding(round(k2))),
            'energies': energies,
            'force_type': self.force_type(chi1, chi2),
            'envelope_asymmetry': (eta, delta),
            'norms': (norm1, norm2),
        }

