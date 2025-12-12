"""
Spatial Wavefunction for SFM Particles.

This module provides explicit spatial wavefunctions φ_n(r) with radial node structure
using 3D HARMONIC OSCILLATOR states.

The number of radial nodes determines the generation/spatial mode:

  n=1: Ground state, no radial nodes (electron, u/d quarks)
  n=2: First excited state, 1 radial node (muon, c/s quarks)
  n=3: Second excited state, 2 radial nodes (tau, b/t quarks)

CRITICAL PHYSICS (SFM):
Unlike hydrogen-like wavefunctions that spread out for higher n, harmonic oscillator
states maintain the SAME spatial extent (Gaussian envelope) but have MORE oscillations
for higher n. This creates:

  - Higher n → more oscillations → larger gradients
  - Larger gradients → stronger spatial-subspace coupling
  - Stronger coupling → larger subspace amplitude A
  - Larger A → greater mass → greater spacetime curvature
  - Greater curvature → reduced spatial radius (self-consistent!)

This is the mechanism for the lepton/quark mass hierarchy in SFM.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.special import genlaguerre


@dataclass
class SpatialGrid:
    """Radial grid for spatial wavefunctions."""
    r: NDArray[np.floating]  # Radial points
    dr: float                 # Grid spacing
    N: int                    # Number of points
    
    @classmethod
    def create(cls, r_max: float = 10.0, N: int = 256) -> 'SpatialGrid':
        """Create a radial grid from 0 to r_max."""
        # Avoid r=0 singularity
        r = np.linspace(1e-6, r_max, N)
        dr = r[1] - r[0]
        return cls(r=r, dr=dr, N=N)


class SpatialWavefunction:
    """
    Radial wavefunction φ_n(r) using 3D HARMONIC OSCILLATOR states.
    
    For s-wave (l=0) 3D harmonic oscillator:
    
      φ_n(r) = N × L_{n-1}^{1/2}(r²/a₀²) × exp(-r²/(2a₀²))
    
    Where L_k^α(x) are generalized Laguerre polynomials.
    
    Key property: ALL modes share the same Gaussian envelope, but higher n
    has more oscillations INSIDE this envelope, creating larger gradients.
    
    This is the correct physics for SFM mass hierarchy!
    """
    
    def __init__(self, n: int, a0: float = 1.0):
        """
        Initialize spatial wavefunction.
        
        Args:
            n: Principal quantum number (n=1,2,3 for e/μ/τ or gen 1/2/3)
            a0: Characteristic length scale (harmonic oscillator length)
        """
        self.n = n
        self.a0 = a0
    
    def __call__(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate φ_n(r) at radial points."""
        return self.evaluate(r)
    
    def evaluate(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute radial wavefunction φ_n(r) using 3D harmonic oscillator states.
        
        For s-wave (l=0):
          n=1: φ₁ ∝ exp(-r²/(2a₀²))                    - Gaussian, no nodes
          n=2: φ₂ ∝ (3/2 - r²/a₀²) exp(-r²/(2a₀²))    - 1 node
          n=3: φ₃ ∝ polynomial(r²) exp(-r²/(2a₀²))    - 2 nodes
        
        All share the SAME Gaussian envelope, different polynomial prefactors.
        """
        # Dimensionless coordinate
        x = (r / self.a0) ** 2  # r²/a₀²
        
        # Gaussian envelope (same for all n!)
        envelope = np.exp(-x / 2)
        
        # Generalized Laguerre polynomial L_{n-1}^{1/2}(x)
        # Using scipy's genlaguerre for numerical stability
        k = self.n - 1  # polynomial degree (n-1 nodes)
        alpha = 0.5     # for 3D harmonic oscillator s-wave
        
        # Compute Laguerre polynomial
        L = genlaguerre(k, alpha)(x)
        
        # Full wavefunction
        phi = L * envelope
        
        # Normalize: ∫|φ|² r² dr = 1/(4π)
        norm = np.sqrt(np.trapz(phi**2 * r**2, r) * 4 * np.pi)
        if norm > 1e-10:
            phi = phi / norm
        
        return phi
    
    def gradient(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute radial gradient dφ_n/dr analytically.
        
        For φ_n(r) = L_{n-1}^{1/2}(r²/a₀²) × exp(-r²/(2a₀²)):
        
        dφ/dr = exp(-r²/(2a₀²)) × [dL/dx × 2r/a₀² - L × r/a₀²]
              = exp(-r²/(2a₀²)) × (2r/a₀²) × [dL/dx - L/2]
        
        Where dL_{k}^{α}/dx = -L_{k-1}^{α+1}(x) for k ≥ 1.
        
        This is the KEY quantity - higher n means more oscillations in L,
        which means larger |dφ/dr|, creating stronger coupling!
        """
        x = (r / self.a0) ** 2  # r²/a₀²
        envelope = np.exp(-x / 2)
        
        k = self.n - 1
        alpha = 0.5
        
        # Laguerre polynomial
        L = genlaguerre(k, alpha)(x)
        
        # Derivative of Laguerre: dL_k^α/dx = -L_{k-1}^{α+1}(x) for k ≥ 1
        if k >= 1:
            dL_dx = -genlaguerre(k - 1, alpha + 1)(x)
        else:
            dL_dx = np.zeros_like(x)
        
        # Chain rule: dφ/dr = d(L·env)/dr
        # = envelope × [dL/dx × dx/dr - L × x × d(envelope)/dx × dx/dr / envelope]
        # where dx/dr = 2r/a₀²
        dx_dr = 2 * r / self.a0**2
        
        # dφ/dr = envelope × [dL/dx × dx/dr + L × (-r/a₀²)]
        #       = envelope × [dL/dx × 2r/a₀² - L × r/a₀²]
        dphi = envelope * (dL_dx * dx_dr - L * r / self.a0**2)
        
        # Apply same normalization as evaluate()
        phi = self.evaluate(r)
        phi_unnorm = L * envelope
        norm = np.sqrt(np.trapz(phi_unnorm**2 * r**2, r) * 4 * np.pi)
        if norm > 1e-10:
            dphi = dphi / norm
        
        return dphi
    
    def _laguerre(self, n: int, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute generalized Laguerre polynomial L_n^{1/2}(x)."""
        # Use scipy for numerical stability
        return genlaguerre(n, 0.5)(x)
    
    def compute_gradient_integral(self, r: NDArray[np.floating]) -> float:
        """
        Compute the spatial gradient integral ∫ (dφ/dr)² r² dr.
        
        This measures the "kinetic energy" contribution from spatial gradients.
        For harmonic oscillator states:
          - Higher n → more oscillations → larger |dφ/dr|² → larger integral
          - This is the mechanism for enhanced coupling in higher generations!
        """
        dphi = self.gradient(r)
        
        integrand = dphi**2 * r**2
        return float(np.trapz(integrand, r) * 4 * np.pi)
    
    def compute_coupling_factor(self, r: NDArray[np.floating]) -> float:
        """
        Compute √(∫ (dφ/dr)² r² dr) = RMS gradient strength.
        
        This is the spatial part of the coupling:
        E_coupling = -α × [RMS_gradient] × [subspace_winding]
        
        For HARMONIC OSCILLATOR states:
          n=1: √1.5 ≈ 1.22  (ground state, no nodes)
          n=2: √3.5 ≈ 1.87  (1 node, ~1.5× stronger)
          n=3: √5.5 ≈ 2.35  (2 nodes, ~1.9× stronger)
        
        This INCREASING coupling with n is the mechanism for mass hierarchy!
        Higher n → more oscillations → larger gradients → stronger coupling
        → larger subspace amplitude → larger mass
        """
        dphi = self.gradient(r)
        
        gradient_integral = np.trapz(dphi**2 * r**2, r) * 4 * np.pi
        return float(np.sqrt(gradient_integral))


def compute_spatial_subspace_coupling(
    phi: SpatialWavefunction,
    chi: NDArray[np.complexfloating],
    r_grid: NDArray[np.floating],
    sigma_grid: NDArray[np.floating],
    dsigma: float,
    alpha: float
) -> float:
    """
    Compute the full spatial-subspace coupling energy:
    
    E_coupling = -α × [spatial_factor] × [subspace_factor]
    
    Where:
    - spatial_factor = √(∫ (dφ/dr)² r² dr) = RMS gradient strength
      INCREASES with n for harmonic oscillator states!
    - subspace_factor = Im[∫ χ* (∂χ/∂σ) dσ] = effective winding
    
    Args:
        phi: Spatial wavefunction object
        chi: Subspace wavefunction (complex array)
        r_grid: Radial grid points
        sigma_grid: Subspace grid points
        dsigma: Subspace grid spacing
        alpha: Coupling constant
        
    Returns:
        E_coupling energy (sign determined by winding)
    """
    # Spatial factor: √(∫ (dφ/dr)² r² dr) - RMS gradient strength
    dphi_vals = phi.gradient(r_grid)
    gradient_integral = np.trapz(dphi_vals**2 * r_grid**2, r_grid) * 4 * np.pi
    spatial_factor = np.sqrt(gradient_integral)
    
    # Subspace factor: Im[∫ χ* (∂χ/∂σ) dσ] - effective winding
    # Use spectral derivative
    N = len(chi)
    k = np.fft.fftfreq(N, dsigma / (2 * np.pi)) * 2 * np.pi
    chi_hat = np.fft.fft(chi)
    dchi_hat = 1j * k * chi_hat
    dchi = np.fft.ifft(dchi_hat)
    
    # IMAGINARY part carries the winding information!
    subspace_factor = np.imag(np.sum(np.conj(chi) * dchi) * dsigma)
    
    # Full coupling
    E_coupling = -alpha * spatial_factor * subspace_factor
    
    return E_coupling


def compute_coupling_gradient_chi(
    phi: SpatialWavefunction,
    chi: NDArray[np.complexfloating],
    r_grid: NDArray[np.floating],
    sigma_grid: NDArray[np.floating],
    dsigma: float,
    alpha: float
) -> NDArray[np.complexfloating]:
    """
    Compute gradient of E_coupling with respect to χ*.
    
    E_coupling = -α × spatial_factor × Im[∫χ*(∂χ/∂σ)dσ]
    
    The gradient w.r.t. χ* involves the imaginary part operator.
    """
    # Spatial factor (RMS gradient strength)
    dphi_vals = phi.gradient(r_grid)
    gradient_integral = np.trapz(dphi_vals**2 * r_grid**2, r_grid) * 4 * np.pi
    spatial_factor = np.sqrt(gradient_integral)
    
    # Subspace derivative
    N = len(chi)
    k = np.fft.fftfreq(N, dsigma / (2 * np.pi)) * 2 * np.pi
    chi_hat = np.fft.fft(chi)
    dchi_hat = 1j * k * chi_hat
    dchi = np.fft.ifft(dchi_hat)
    
    # Gradient of Im[∫χ*(∂χ/∂σ)dσ] w.r.t. χ*
    # For f = Im[∫χ*g dσ], δf/δχ* = (1/2i)g = -i/2 × g
    grad = -alpha * spatial_factor * (-1j/2) * dchi
    
    return grad

