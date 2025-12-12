"""
Universal Energy Minimizer for SFM Particles.

CRITICAL FIX: E_coupling is now computed from ACTUAL WAVEFUNCTIONS:
  E_coupling = -α ∫∫ (∇φ · ∂χ/∂σ) φ χ d³x dσ

This separates into:
  E_coupling = -α × [spatial_factor] × [subspace_factor]

Where:
- spatial_factor = ∫ (dφ/dr) φ r² dr  (from radial wavefunction with n nodes)
- subspace_factor = Re[∫ χ* (∂χ/∂σ) dσ]  (from subspace wavefunction)

The mass hierarchy EMERGES from the wavefunction structures:
- Different n → different radial nodes → different spatial_factor
- Different χ structure (1/2/3 peaks) → different subspace_factor
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple, Optional

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.spatial_wavefunction import SpatialWavefunction, SpatialGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


@dataclass
class EnergyBreakdown:
    """Complete breakdown of the energy functional."""
    E_total: float
    E_subspace: float
    E_spatial: float
    E_coupling: float
    E_curvature: float
    
    # Subspace components
    E_kinetic: float
    E_potential: float
    E_nonlinear: float
    E_circulation: float
    
    # Coupling components (for debugging)
    spatial_factor: float
    subspace_factor: float


@dataclass
class MinimizationResult:
    """Result of energy minimization."""
    chi: NDArray[np.complexfloating]
    
    # Optimized variables
    A: float
    A_squared: float
    delta_x: float
    delta_sigma: float
    
    # Spatial mode (for reference)
    n_spatial: int
    
    # Energy breakdown
    energy: EnergyBreakdown
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class WavefunctionEnergyMinimizer:
    """
    Energy minimizer using actual wavefunctions for coupling.
    
    Key physics:
    - φ_n(r): Spatial wavefunction with n-1 radial nodes
    - χ(σ): Subspace wavefunction (single or composite)
    - E_coupling emerges from ∫∫ (∇φ · ∂χ/∂σ) φ χ d³x dσ
    - No input quantum numbers needed - everything from wavefunctions!
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        alpha: float,
        beta: float,
        kappa: float,
        g1: float,
        g2: float,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        r_max: float = 10.0,
        N_r: int = 256,
    ):
        """Initialize with fundamental parameters and spatial grid."""
        self.grid = grid
        self.potential = potential
        self.V_grid = potential(grid.sigma)
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.g1 = g1
        self.g2 = g2
        self.m_eff = m_eff
        self.hbar = hbar
        
        # Create spatial grid for radial wavefunctions
        self.r_grid = SpatialGrid.create(r_max=r_max, N=N_r)
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
    
    def _compute_subspace_gradient(self, chi: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """Compute ∂χ/∂σ using spectral method."""
        return self.grid.first_derivative(chi)
    
    def _compute_subspace_factor(self, chi: NDArray[np.complexfloating]) -> float:
        """
        Compute subspace coupling factor: Im[∫ χ* (∂χ/∂σ) dσ]
        
        CRITICAL: The winding is encoded in the IMAGINARY part!
        For χ ~ exp(ikσ): χ* ∂χ/∂σ = ik|χ|², so Im[...] = k∫|χ|²dσ
        
        This captures the "winding" of the subspace wavefunction.
        For composite wavefunctions, interference is automatically included.
        """
        dchi = self._compute_subspace_gradient(chi)
        factor = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        return float(np.imag(factor))  # IMAGINARY part carries winding!
    
    def compute_energy(
        self,
        chi: NDArray[np.complexfloating],
        phi: SpatialWavefunction,
        delta_x: float,
    ) -> EnergyBreakdown:
        """
        Compute E_total from actual wavefunctions.
        
        E_coupling is computed from the INTEGRAL over both wavefunctions,
        not from input quantum numbers!
        """
        # === AMPLITUDE FROM SUBSPACE WAVEFUNCTION ===
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        
        # === SUBSPACE ENERGY FROM ACTUAL INTEGRALS ===
        
        # Kinetic: ∫(ℏ²/2m)|∇χ|² dσ
        T_chi = self.operators.apply_kinetic(chi)
        E_kinetic = np.real(self.grid.inner_product(chi, T_chi))
        
        # Potential: ∫V(σ)|χ|² dσ
        E_potential = np.real(np.sum(self.V_grid * np.abs(chi)**2) * self.grid.dsigma)
        
        # Nonlinear: (g₁/2)∫|χ|⁴ dσ
        E_nonlinear = (self.g1 / 2) * np.sum(np.abs(chi)**4) * self.grid.dsigma
        
        # Circulation: g₂|∫χ*∂χ/∂σ|²
        dchi = self._compute_subspace_gradient(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        E_circulation = self.g2 * np.abs(J)**2
        
        E_subspace = E_kinetic + E_potential + E_nonlinear + E_circulation
        
        # === SPATIAL ENERGY ===
        if A_sq > 1e-10 and delta_x > 1e-10:
            E_spatial = self.hbar**2 / (2 * self.beta * A_sq * delta_x**2)
        else:
            E_spatial = 1e10
        
        # === COUPLING ENERGY FROM ACTUAL INTEGRALS ===
        # E_coupling = -α × [spatial_factor] × [subspace_factor]
        
        # Spatial factor: ∫ (dφ/dr) φ r² dr
        r = self.r_grid.r
        phi_vals = phi.evaluate(r)
        dphi_vals = phi.gradient(r)
        spatial_factor = np.trapz(dphi_vals * phi_vals * r**2, r) * 4 * np.pi
        
        # Subspace factor: Re[∫ χ* (∂χ/∂σ) dσ]
        subspace_factor = self._compute_subspace_factor(chi)
        
        # Full coupling (emerges from wavefunctions!)
        E_coupling = -self.alpha * spatial_factor * subspace_factor
        
        # === CURVATURE ENERGY ===
        if delta_x > 1e-10:
            mass_sq = (self.beta * A_sq) ** 2
            E_curvature = self.kappa * mass_sq / delta_x
        else:
            E_curvature = 1e10
        
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
        
        return EnergyBreakdown(
            E_total=E_total,
            E_subspace=E_subspace,
            E_spatial=E_spatial,
            E_coupling=E_coupling,
            E_curvature=E_curvature,
            E_kinetic=E_kinetic,
            E_potential=E_potential,
            E_nonlinear=E_nonlinear,
            E_circulation=E_circulation,
            spatial_factor=spatial_factor,
            subspace_factor=subspace_factor,
        )
    
    def _compute_wavefunction_gradient(
        self,
        chi: NDArray[np.complexfloating],
        phi: SpatialWavefunction,
        delta_x: float,
    ) -> NDArray[np.complexfloating]:
        """Compute gradient δE/δχ* for wavefunction optimization."""
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        
        # === SUBSPACE GRADIENTS ===
        T_chi = self.operators.apply_kinetic(chi)
        V_chi = self.V_grid * chi
        NL_chi = self.g1 * np.abs(chi)**2 * chi
        
        dchi = self._compute_subspace_gradient(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        circ_grad = 2 * self.g2 * np.real(np.conj(J)) * dchi
        
        grad_subspace = T_chi + V_chi + NL_chi + circ_grad
        
        # === SPATIAL GRADIENT ===
        if A_sq > 1e-10 and delta_x > 1e-10:
            grad_spatial = -self.hbar**2 / (self.beta * A_sq**2 * delta_x**2) * chi
        else:
            grad_spatial = np.zeros_like(chi)
        
        # === COUPLING GRADIENT ===
        # E_coupling = -α × spatial_factor × subspace_factor
        # δE/δχ* = -α × spatial_factor × (∂χ/∂σ)
        r = self.r_grid.r
        phi_vals = phi.evaluate(r)
        dphi_vals = phi.gradient(r)
        spatial_factor = np.trapz(dphi_vals * phi_vals * r**2, r) * 4 * np.pi
        
        grad_coupling = -self.alpha * spatial_factor * dchi
        
        # === CURVATURE GRADIENT ===
        if delta_x > 1e-10:
            grad_curvature = 4 * self.kappa * self.beta**2 * A_sq / delta_x * chi
        else:
            grad_curvature = np.zeros_like(chi)
        
        return grad_subspace + grad_spatial + grad_coupling + grad_curvature
    
    def minimize(
        self,
        chi_initial: NDArray[np.complexfloating],
        n_spatial: int,
        a0: float = 1.0,
        max_iter: int = 20000,
        tol: float = 1e-10,
        verbose: bool = False
    ) -> MinimizationResult:
        """
        Minimize energy over χ and Δx.
        
        Args:
            chi_initial: Initial subspace wavefunction
            n_spatial: Spatial mode number (determines radial node structure)
            a0: Characteristic length scale for spatial wavefunction
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress
        """
        # Create spatial wavefunction with n radial nodes
        phi = SpatialWavefunction(n=n_spatial, a0=a0)
        
        chi = chi_initial.copy()
        delta_x = 1.0
        
        dt_chi = 0.0001
        dt_dx = 0.0001
        
        energy = self.compute_energy(chi, phi, delta_x)
        E_old = energy.E_total
        
        converged = False
        final_residual = float('inf')
        
        if verbose:
            print(f"  Minimizing with n_spatial={n_spatial}")
            print(f"  Spatial factor (from φ_{n_spatial}): {energy.spatial_factor:.4g}")
        
        for iteration in range(max_iter):
            # === OPTIMIZE χ ===
            grad_chi = self._compute_wavefunction_gradient(chi, phi, delta_x)
            chi_new = chi - dt_chi * grad_chi
            
            # === OPTIMIZE Δx ===
            A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
            if A_sq > 1e-10 and delta_x > 1e-10:
                dE_ddx = -self.hbar**2 / (self.beta * A_sq * delta_x**3)
                dE_ddx -= self.kappa * (self.beta * A_sq)**2 / (delta_x**2)
            else:
                dE_ddx = 0
            
            delta_x_new = delta_x - dt_dx * dE_ddx
            delta_x_new = max(delta_x_new, 1e-6)
            delta_x_new = min(delta_x_new, 1e6)
            
            energy_new = self.compute_energy(chi_new, phi, delta_x_new)
            E_new = energy_new.E_total
            
            # Adaptive step size
            if E_new > E_old:
                dt_chi *= 0.5
                dt_dx *= 0.5
                if dt_chi < 1e-15:
                    break
                continue
            else:
                dt_chi = min(dt_chi * 1.02, 0.001)
                dt_dx = min(dt_dx * 1.02, 0.001)
            
            chi = chi_new
            delta_x = delta_x_new
            
            dE = abs(E_new - E_old)
            final_residual = dE
            
            if verbose and iteration % 2000 == 0:
                A = np.sqrt(np.sum(np.abs(chi)**2) * self.grid.dsigma)
                print(f"  Iter {iteration}: E={E_new:.4f}, A={A:.6f}, "
                      f"E_coup={energy_new.E_coupling:.4g}")
            
            if dE < tol:
                converged = True
                energy = energy_new
                break
            
            E_old = E_new
            energy = energy_new
        
        # Final values
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(A_sq)
        
        # Compute Δσ from wavefunction
        sigma = self.grid.sigma
        chi_sq = np.abs(chi)**2
        norm = np.sum(chi_sq) * self.grid.dsigma
        if norm > 1e-10:
            mean_sigma = np.sum(sigma * chi_sq) * self.grid.dsigma / norm
            var = np.sum((sigma - mean_sigma)**2 * chi_sq) * self.grid.dsigma / norm
            delta_sigma = np.sqrt(max(var, 1e-10))
        else:
            delta_sigma = 0.5
        
        return MinimizationResult(
            chi=chi,
            A=A,
            A_squared=A_sq,
            delta_x=delta_x,
            delta_sigma=delta_sigma,
            n_spatial=n_spatial,
            energy=energy,
            converged=converged,
            iterations=iteration + 1,
            final_residual=final_residual,
        )
