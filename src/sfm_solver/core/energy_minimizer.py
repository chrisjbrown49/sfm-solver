"""
Universal Energy Minimizer for SFM Particles.

CRITICAL PHYSICS:
- Quantum numbers (n, k) are INPUTS that define which particle
- n = spatial mode number (1,2,3 for electron/muon/tau)
- k = topological winding number (1 for leptons, varies for quarks)
- E_coupling uses these quantum numbers DIRECTLY, not computed from gradients
- Wavefunction structure affects E_kinetic, E_potential, E_nonlinear through actual integrals
- Minimize over (A, Δx, Δσ) with (n, k) as fixed inputs
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple, Optional

from sfm_solver.core.grid import SpectralGrid
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
    
    # Subspace components (from actual integrals)
    E_kinetic: float      # ∫(ℏ²/2m)|∇χ|² dσ
    E_potential: float    # ∫V(σ)|χ|² dσ
    E_nonlinear: float    # (g₁/2)∫|χ|⁴ dσ
    E_circulation: float  # g₂|∫χ*∂χ/∂σ|²


@dataclass
class MinimizationResult:
    """Result of energy minimization."""
    chi: NDArray[np.complexfloating]
    
    # Optimized variables
    A: float
    A_squared: float
    delta_x: float
    delta_sigma: float
    
    # INPUT quantum numbers (not computed!)
    n_spatial: int
    k_winding: int
    
    # Energy breakdown
    energy: EnergyBreakdown
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class WavefunctionEnergyMinimizer:
    """
    Energy minimizer with quantum numbers as inputs.
    
    Key physics:
    - (n, k) are quantum numbers that DEFINE which particle
    - E_coupling = -α × f(n) × k × A uses these directly
    - Wavefunction structure affects subspace energies via integrals
    - Minimize E_total(A, Δx, Δσ) with (n, k) fixed
    """
    
    # Spatial mode enhancement power for leptons
    # From research notes: f(n) = n^p where p ≈ 8.75
    SPATIAL_MODE_POWER = 8.75
    
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
    ):
        """Initialize with fundamental parameters."""
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
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
    
    def compute_energy(
        self,
        chi: NDArray[np.complexfloating],
        delta_x: float,
        n_spatial: int,
        k_winding: int,
        is_lepton: bool = True
    ) -> EnergyBreakdown:
        """
        Compute E_total with quantum numbers (n, k) as inputs.
        
        Args:
            chi: Wavefunction on grid
            delta_x: Spatial extent parameter
            n_spatial: Spatial mode quantum number (INPUT!)
            k_winding: Topological winding quantum number (INPUT!)
            is_lepton: True for leptons, False for hadrons (different coupling)
        """
        # === AMPLITUDE FROM WAVEFUNCTION ===
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
        
        # Circulation: g₂|J|² where J = ∫χ*∂χ/∂σ dσ
        dchi = self.grid.first_derivative(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        E_circulation = self.g2 * np.abs(J)**2
        
        E_subspace = E_kinetic + E_potential + E_nonlinear + E_circulation
        
        # === SPATIAL ENERGY ===
        # E_spatial = ℏ²/(2βA²Δx²)
        if A_sq > 1e-10 and delta_x > 1e-10:
            E_spatial = self.hbar**2 / (2 * self.beta * A_sq * delta_x**2)
        else:
            E_spatial = 1e10
        
        # === COUPLING ENERGY ===
        # E_coupling = -α × f(n) × k × A
        # This uses QUANTUM NUMBERS directly, not computed values!
        if is_lepton:
            # Leptons: f(n) = n^p where p ≈ 8.75
            f_n = float(n_spatial) ** self.SPATIAL_MODE_POWER
            E_coupling = -self.alpha * f_n * k_winding * A
        else:
            # Hadrons: Different coupling physics
            # For composites, the binding comes more from nonlinear term
            # Coupling is weaker
            f_n = float(n_spatial)  # Linear in n for hadrons
            E_coupling = -self.alpha * f_n * k_winding * A
        
        # === CURVATURE ENERGY ===
        # E_curvature = κ × (βA²)² / Δx
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
        )
    
    def compute_wavefunction_gradient(
        self,
        chi: NDArray[np.complexfloating],
        delta_x: float,
        n_spatial: int,
        k_winding: int,
        is_lepton: bool = True
    ) -> NDArray[np.complexfloating]:
        """
        Compute gradient δE/δχ* for wavefunction optimization.
        """
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        
        # === SUBSPACE GRADIENTS ===
        T_chi = self.operators.apply_kinetic(chi)
        V_chi = self.V_grid * chi
        NL_chi = self.g1 * np.abs(chi)**2 * chi
        
        dchi = self.grid.first_derivative(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        circ_grad = 2 * self.g2 * np.real(np.conj(J)) * dchi
        
        grad_subspace = T_chi + V_chi + NL_chi + circ_grad
        
        # === SPATIAL GRADIENT ===
        if A_sq > 1e-10 and delta_x > 1e-10:
            grad_spatial = -self.hbar**2 / (self.beta * A_sq**2 * delta_x**2) * chi
        else:
            grad_spatial = np.zeros_like(chi)
        
        # === COUPLING GRADIENT ===
        # E_coupling = -α × f(n) × k × A
        # δE/δχ* = -α × f(n) × k / (2A) × χ
        if is_lepton:
            f_n = float(n_spatial) ** self.SPATIAL_MODE_POWER
        else:
            f_n = float(n_spatial)
        
        if A > 1e-10:
            grad_coupling = -self.alpha * f_n * k_winding / (2 * A) * chi
        else:
            grad_coupling = np.zeros_like(chi)
        
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
        k_winding: int,
        is_lepton: bool = True,
        max_iter: int = 20000,
        tol: float = 1e-10,
        verbose: bool = False
    ) -> MinimizationResult:
        """
        Minimize energy over (A, Δx, Δσ) with quantum numbers (n, k) fixed.
        
        Args:
            chi_initial: Initial wavefunction
            n_spatial: Spatial mode quantum number (INPUT - defines particle!)
            k_winding: Topological winding number (INPUT - defines particle!)
            is_lepton: True for leptons, False for hadrons
        """
        chi = chi_initial.copy()
        delta_x = 1.0
        
        dt_chi = 0.0001  # Smaller step for stability
        dt_dx = 0.0001
        
        energy = self.compute_energy(chi, delta_x, n_spatial, k_winding, is_lepton)
        E_old = energy.E_total
        
        converged = False
        final_residual = float('inf')
        
        if verbose:
            print(f"  Minimizing with n={n_spatial}, k={k_winding}")
            f_n = n_spatial ** self.SPATIAL_MODE_POWER if is_lepton else n_spatial
            print(f"  f(n) = {f_n:.2f}")
        
        for iteration in range(max_iter):
            # === OPTIMIZE WAVEFUNCTION χ ===
            grad_chi = self.compute_wavefunction_gradient(chi, delta_x, n_spatial, k_winding, is_lepton)
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
            
            energy_new = self.compute_energy(chi_new, delta_x_new, n_spatial, k_winding, is_lepton)
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
                print(f"  Iter {iteration}: E={E_new:.4f}, A={A:.6f}, Δx={delta_x:.4g}")
            
            if dE < tol:
                converged = True
                energy = energy_new
                break
            
            E_old = E_new
            energy = energy_new
        
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(A_sq)
        
        # Compute emergent Δσ from wavefunction shape
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
            k_winding=k_winding,
            energy=energy,
            converged=converged,
            iterations=iteration + 1,
            final_residual=final_residual,
        )
