"""
SpectralGrid class for discretizing the S¹ subspace.

Provides FFT-based differentiation on the periodic domain [0, 2π).
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


class SpectralGrid:
    """
    Spectral discretization of the S¹ subspace.
    
    The subspace coordinate σ ranges from 0 to 2π with periodic
    boundary conditions: f(σ + 2π) = f(σ).
    
    Uses FFT-based spectral methods for high-accuracy differentiation.
    
    Attributes:
        N: Number of grid points.
        sigma: Array of σ values on the grid.
        dsigma: Grid spacing.
        k: Wavenumber array for spectral differentiation.
    """
    
    def __init__(self, N: int = 256):
        """
        Initialize the spectral grid.
        
        Args:
            N: Number of grid points. Should be a power of 2 for FFT efficiency.
               Typical values: 128, 256, 512.
        """
        if N < 4:
            raise ValueError(f"N must be at least 4, got {N}")
        
        self.N = N
        
        # Grid points: σ ∈ [0, 2π) with periodic BCs
        # Don't include 2π as it's the same as 0
        self.dsigma = 2 * np.pi / N
        self.sigma = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        # Wavenumbers for spectral differentiation
        # FFT convention: k = [0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1]
        self.k = np.fft.fftfreq(N, d=self.dsigma / (2 * np.pi))
        
        # Precompute Laplacian coefficients: -k²
        self._laplacian_coeff = -self.k**2
    
    def derivative(self, f: NDArray[np.complexfloating], order: int = 1) -> NDArray[np.complexfloating]:
        """
        Compute the derivative of a function using spectral methods.
        
        d^n f / dσ^n using FFT: multiply by (ik)^n in Fourier space.
        
        Args:
            f: Function values on the grid (can be complex).
            order: Order of derivative (1 for first, 2 for second, etc.)
            
        Returns:
            The derivative evaluated on the grid.
        """
        if len(f) != self.N:
            raise ValueError(f"Function length {len(f)} doesn't match grid size {self.N}")
        
        # Transform to Fourier space
        f_hat = np.fft.fft(f)
        
        # Multiply by (ik)^order
        derivative_hat = f_hat * (1j * self.k) ** order
        
        # Transform back
        return np.fft.ifft(derivative_hat)
    
    def first_derivative(self, f: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute the first derivative df/dσ.
        
        Args:
            f: Function values on the grid.
            
        Returns:
            First derivative evaluated on the grid.
        """
        return self.derivative(f, order=1)
    
    def second_derivative(self, f: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute the second derivative d²f/dσ².
        
        Uses the Laplacian coefficients for efficiency.
        
        Args:
            f: Function values on the grid.
            
        Returns:
            Second derivative evaluated on the grid.
        """
        f_hat = np.fft.fft(f)
        d2f_hat = f_hat * self._laplacian_coeff
        return np.fft.ifft(d2f_hat)
    
    def laplacian(self, f: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute the Laplacian d²f/dσ² (same as second_derivative).
        
        Args:
            f: Function values on the grid.
            
        Returns:
            Laplacian evaluated on the grid.
        """
        return self.second_derivative(f)
    
    def integrate(self, f: NDArray[np.complexfloating]) -> complex:
        """
        Integrate a function over the full domain [0, 2π].
        
        Uses trapezoidal rule (exact for periodic functions sampled
        at sufficient resolution).
        
        Args:
            f: Function values on the grid.
            
        Returns:
            The integral ∫₀^(2π) f(σ) dσ.
        """
        return np.sum(f) * self.dsigma
    
    def norm(self, f: NDArray[np.complexfloating]) -> float:
        """
        Compute the L² norm of a function.
        
        ||f||² = ∫₀^(2π) |f(σ)|² dσ
        
        Args:
            f: Function values on the grid.
            
        Returns:
            The L² norm.
        """
        return np.sqrt(np.real(self.integrate(np.abs(f)**2)))
    
    def normalize(
        self, 
        f: NDArray[np.complexfloating], 
        target_norm: float = np.sqrt(2 * np.pi)
    ) -> NDArray[np.complexfloating]:
        """
        Normalize a function to have a specified L² norm.
        
        Default normalization: ∫₀^(2π) |f|² dσ = 2π
        (so that uniform |f|=1 integrates to 2π)
        
        Args:
            f: Function values on the grid.
            target_norm: Desired L² norm. Default √(2π).
            
        Returns:
            Normalized function.
        """
        current_norm = self.norm(f)
        if current_norm < 1e-14:
            raise ValueError("Cannot normalize zero function")
        return f * (target_norm / current_norm)
    
    def inner_product(
        self, 
        f: NDArray[np.complexfloating], 
        g: NDArray[np.complexfloating]
    ) -> complex:
        """
        Compute the inner product <f|g> = ∫₀^(2π) f*(σ) g(σ) dσ.
        
        Args:
            f: First function (will be conjugated).
            g: Second function.
            
        Returns:
            The inner product.
        """
        return self.integrate(np.conj(f) * g)
    
    def circulation(self, chi: NDArray[np.complexfloating]) -> complex:
        """
        Compute the circulation integral for a wavefunction.
        
        J = ∫₀^(2π) χ*(σ) ∂χ/∂σ dσ
        
        This is proportional to the winding number k.
        
        Args:
            chi: Wavefunction values on the grid.
            
        Returns:
            The circulation integral.
        """
        dchi = self.first_derivative(chi)
        return self.integrate(np.conj(chi) * dchi)
    
    def winding_number(self, chi: NDArray[np.complexfloating]) -> float:
        """
        Extract the winding number from the circulation integral.
        
        For χ(σ) = A exp(ikσ) f(σ), the circulation gives ik times the norm.
        
        k = Im[J] / (∫|χ|² dσ)
        
        Args:
            chi: Wavefunction values on the grid.
            
        Returns:
            The (approximate) winding number.
        """
        J = self.circulation(chi)
        norm_sq = np.real(self.integrate(np.abs(chi)**2))
        if norm_sq < 1e-14:
            return 0.0
        return np.imag(J) / norm_sq
    
    def interpolate(
        self, 
        f: NDArray[np.complexfloating], 
        sigma_new: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """
        Interpolate function to new σ values using spectral methods.
        
        Uses the FFT representation to evaluate at arbitrary points.
        
        Args:
            f: Function values on the grid.
            sigma_new: New σ values where to evaluate.
            
        Returns:
            Interpolated function values.
        """
        f_hat = np.fft.fft(f) / self.N
        
        result = np.zeros(len(sigma_new), dtype=complex)
        for j, kj in enumerate(self.k):
            result += f_hat[j] * np.exp(1j * kj * sigma_new)
        
        return result
    
    def create_winding_mode(self, k: int, amplitude: float = 1.0) -> NDArray[np.complexfloating]:
        """
        Create a pure winding mode exp(i k σ).
        
        Args:
            k: Winding number (integer).
            amplitude: Overall amplitude.
            
        Returns:
            The winding mode on the grid.
        """
        return amplitude * np.exp(1j * k * self.sigma)
    
    def create_gaussian_envelope(
        self, 
        center: float = 0.0, 
        width: float = 0.5,
        periodic: bool = True
    ) -> NDArray[np.floating]:
        """
        Create a Gaussian envelope centered at a given σ.
        
        f(σ) = exp(-d²/(2w²))
        
        where d is the periodic distance to the center.
        
        Args:
            center: Center of the Gaussian (in radians).
            width: Width parameter w.
            periodic: If True, use periodic distance on circle.
            
        Returns:
            Gaussian envelope on the grid.
        """
        if periodic:
            # Periodic distance on circle
            d = self.sigma - center
            d = np.mod(d + np.pi, 2 * np.pi) - np.pi  # Wrap to [-π, π]
        else:
            d = self.sigma - center
        
        return np.exp(-d**2 / (2 * width**2))
    
    def create_localized_mode(
        self,
        k: int,
        center: float = 0.0,
        width: float = 0.5,
        amplitude: float = 1.0
    ) -> NDArray[np.complexfloating]:
        """
        Create a localized winding mode: A × exp(ikσ) × Gaussian(σ).
        
        This represents a particle localized in one of the wells
        with winding number k.
        
        Args:
            k: Winding number.
            center: Center of localization (well position).
            width: Width of Gaussian envelope.
            amplitude: Overall amplitude.
            
        Returns:
            Localized winding mode on the grid.
        """
        envelope = self.create_gaussian_envelope(center, width)
        winding = np.exp(1j * k * self.sigma)
        return amplitude * envelope * winding
    
    def __repr__(self) -> str:
        return f"SpectralGrid(N={self.N}, dsigma={self.dsigma:.6f})"

