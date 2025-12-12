"""
Spatial Wavefunction for SFM Particles.

This module provides explicit spatial wavefunctions φ_n(r) with radial node structure.
The number of radial nodes determines the generation/spatial mode:

  n=1: Ground state, no radial nodes (electron, u/d quarks)
  n=2: First excited state, 1 radial node (muon, c/s quarks)
  n=3: Second excited state, 2 radial nodes (tau, b/t quarks)

The gradient ∇φ_n naturally scales differently for different n, which creates
the mass hierarchy when combined with the subspace wavefunction gradient ∂χ/∂σ.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from dataclasses import dataclass


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
    Radial wavefunction φ_n(r) with n-1 radial nodes.
    
    Based on hydrogen-like radial functions, but with adjustable scale.
    The key physics: higher n means more nodes, which means larger gradients,
    which creates stronger spatial-subspace coupling.
    """
    
    def __init__(self, n: int, a0: float = 1.0):
        """
        Initialize spatial wavefunction.
        
        Args:
            n: Principal quantum number (n=1,2,3 for e/μ/τ or gen 1/2/3)
            a0: Characteristic length scale (like Bohr radius)
        """
        self.n = n
        self.a0 = a0
    
    def __call__(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate φ_n(r) at radial points."""
        return self.evaluate(r)
    
    def evaluate(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute radial wavefunction φ_n(r).
        
        Using hydrogen-like radial functions (simplified, s-wave only):
        n=1: φ₁ ∝ exp(-r/a₀)
        n=2: φ₂ ∝ (1 - r/(2a₀)) exp(-r/(2a₀))
        n=3: φ₃ ∝ (1 - 2r/(3a₀) + 2r²/(27a₀²)) exp(-r/(3a₀))
        """
        rho = r / self.a0
        
        if self.n == 1:
            # Ground state: no nodes
            phi = np.exp(-rho)
        elif self.n == 2:
            # First excited: 1 node at r = 2a₀
            phi = (1 - rho/2) * np.exp(-rho/2)
        elif self.n == 3:
            # Second excited: 2 nodes
            phi = (1 - 2*rho/3 + 2*rho**2/27) * np.exp(-rho/3)
        else:
            # General case using Laguerre polynomials (simplified)
            # L_{n-1}^1(2r/(n*a0)) * exp(-r/(n*a0))
            x = 2*rho/self.n
            L = self._laguerre(self.n - 1, x)
            phi = L * np.exp(-rho/self.n)
        
        # Normalize
        norm = np.sqrt(np.trapz(phi**2 * r**2, r) * 4 * np.pi)
        if norm > 1e-10:
            phi = phi / norm
        
        return phi
    
    def gradient(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute radial gradient dφ_n/dr.
        
        This is the KEY quantity - it scales differently for different n,
        creating the mass hierarchy through the coupling integral.
        """
        rho = r / self.a0
        
        if self.n == 1:
            # d/dr[exp(-r/a₀)] = -1/a₀ exp(-r/a₀)
            dphi = -1/self.a0 * np.exp(-rho)
        elif self.n == 2:
            # d/dr[(1-r/2a₀)exp(-r/2a₀)]
            dphi = (-1/(2*self.a0) - (1-rho/2)*(-1/(2*self.a0))) * np.exp(-rho/2)
            dphi = (1/(2*self.a0)) * (rho/2 - 1 - (1 - rho/2)) * np.exp(-rho/2)
            # Simplified: derivative of (1-x/2)e^(-x/2) where x = r/a0
            # = -1/2 e^(-x/2) + (1-x/2)(-1/2)e^(-x/2) = -1/2 e^(-x/2)(1 + 1 - x/2)
            # = -1/2 e^(-x/2)(2 - x/2) = -(1 - x/4) e^(-x/2)
            dphi = -1/self.a0 * (1 - rho/4) * np.exp(-rho/2)
        elif self.n == 3:
            # Numerical derivative for n=3 (more complex)
            phi = self.evaluate(r)
            dphi = np.gradient(phi, r)
        else:
            phi = self.evaluate(r)
            dphi = np.gradient(phi, r)
        
        return dphi
    
    def _laguerre(self, n: int, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute generalized Laguerre polynomial L_n^1(x)."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 - x
        else:
            # Recurrence relation
            L0 = np.ones_like(x)
            L1 = 2 - x
            for k in range(2, n + 1):
                L2 = ((2*k + 1 - x) * L1 - (k + 1) * L0) / k
                L0, L1 = L1, L2
            return L1
    
    def compute_gradient_integral(self, r: NDArray[np.floating]) -> float:
        """
        Compute the spatial gradient integral ∫ (dφ/dr)² φ² r² dr.
        
        This measures how "active" the spatial gradient is, weighted by
        the wavefunction amplitude. Higher n → more nodes → larger integral.
        """
        phi = self.evaluate(r)
        dphi = self.gradient(r)
        
        integrand = dphi**2 * phi**2 * r**2
        return float(np.trapz(integrand, r) * 4 * np.pi)
    
    def compute_coupling_factor(self, r: NDArray[np.floating]) -> float:
        """
        Compute ∫ (dφ/dr) φ r² dr.
        
        This is the spatial part of the coupling integral:
        E_coupling = -α ∫∫ (∇φ · ∂χ/∂σ) φ χ d³x dσ
                   = -α × [∫(dφ/dr)φ r² dr] × [∫(∂χ/∂σ)χ dσ]
        """
        phi = self.evaluate(r)
        dphi = self.gradient(r)
        
        integrand = dphi * phi * r**2
        return float(np.trapz(integrand, r) * 4 * np.pi)


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
    
    E_coupling = -α ∫∫ (∇φ · ∂χ/∂σ) φ χ d³x dσ
    
    For radial φ and periodic χ, this separates:
    E_coupling = -α × [spatial_factor] × [subspace_factor]
    
    Where:
    - spatial_factor = ∫ (dφ/dr) φ r² dr
    - subspace_factor = ∫ (∂χ/∂σ) χ* dσ  (real part)
    
    Args:
        phi: Spatial wavefunction object
        chi: Subspace wavefunction (complex array)
        r_grid: Radial grid points
        sigma_grid: Subspace grid points
        dsigma: Subspace grid spacing
        alpha: Coupling constant
        
    Returns:
        E_coupling energy (negative drives amplitude up)
    """
    # Spatial factor: ∫ (dφ/dr) φ r² dr
    phi_vals = phi.evaluate(r_grid)
    dphi_vals = phi.gradient(r_grid)
    spatial_factor = np.trapz(dphi_vals * phi_vals * r_grid**2, r_grid) * 4 * np.pi
    
    # Subspace factor: ∫ χ* (∂χ/∂σ) dσ
    # Use spectral derivative
    N = len(chi)
    k = np.fft.fftfreq(N, dsigma / (2 * np.pi)) * 2 * np.pi
    chi_hat = np.fft.fft(chi)
    dchi_hat = 1j * k * chi_hat
    dchi = np.fft.ifft(dchi_hat)
    
    subspace_factor = np.real(np.sum(np.conj(chi) * dchi) * dsigma)
    
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
    
    δE_coupling/δχ* = -α × [spatial_factor] × (∂χ/∂σ)
    """
    # Spatial factor
    phi_vals = phi.evaluate(r_grid)
    dphi_vals = phi.gradient(r_grid)
    spatial_factor = np.trapz(dphi_vals * phi_vals * r_grid**2, r_grid) * 4 * np.pi
    
    # Subspace derivative
    N = len(chi)
    k = np.fft.fftfreq(N, dsigma / (2 * np.pi)) * 2 * np.pi
    chi_hat = np.fft.fft(chi)
    dchi_hat = 1j * k * chi_hat
    dchi = np.fft.ifft(dchi_hat)
    
    # Gradient
    grad = -alpha * spatial_factor * dchi
    
    return grad

