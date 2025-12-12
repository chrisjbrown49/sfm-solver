"""
Universal Energy Minimizer for SFM Particles.

CRITICAL REQUIREMENTS:
======================
A. ALL predictions must EMERGE from first principles of the Single-Field Model.
   NO phenomenological parameters are permitted.

B. Uses ONLY fundamental parameters derived from first principles:
   - β: Fundamental mass scale (GeV)
   - κ = 1/β²: Curvature coupling (GeV⁻²)
   - α = C × β: Spacetime-subspace coupling (GeV)
   - g₁ = α_em × β / m_e: Nonlinear self-interaction
   - g₂ = α_em / 2: Circulation coupling
   - V₀ = 1.0 GeV: Three-well potential depth

C. Minimizes E_total(A, Δx, Δσ) over ALL THREE variables simultaneously.
   This is UNIVERSAL across all particle types (leptons, baryons, mesons).

ENERGY FUNCTIONAL (from "Research Note - A Beautiful Balance"):
===============================================================
E_total = E_subspace + E_spatial + E_coupling + E_curvature

Where:
- E_subspace = E_kinetic + E_potential + E_nonlinear + E_circulation
- E_spatial = ℏ²/(2βA²Δx²)
- E_coupling = -α × k_eff × A
- E_curvature = κ × (βA²)² / Δx
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class EnergyBreakdown:
    """Complete breakdown of the four-term energy functional."""
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


@dataclass
class MinimizationResult:
    """Result of universal energy minimization."""
    # Optimized variables
    A: float              # Equilibrium amplitude
    A_squared: float      # A² (determines mass via m = β × A²)
    delta_x: float        # Equilibrium spatial extent
    delta_sigma: float    # Equilibrium subspace width
    
    # Energy breakdown
    energy: EnergyBreakdown
    
    # Input parameter
    k_eff: float          # Effective winding (from wavefunction)
    
    # Convergence info
    converged: bool
    iterations: int
    final_residual: float


class UniversalEnergyMinimizer:
    """
    Universal energy minimizer for ALL SFM particles.
    
    Minimizes E_total(A, Δx, Δσ) over all three variables simultaneously.
    The particle-specific wavefunction determines k_eff, which is passed
    as an input parameter.
    
    This class contains NO phenomenological parameters. All physics is
    encoded in the energy functional derived from SFM first principles.
    """
    
    def __init__(
        self,
        alpha: float,      # Spacetime-subspace coupling (GeV)
        beta: float,       # Mass scale (GeV)
        kappa: float,      # Curvature coupling (GeV⁻²)
        g1: float,         # Nonlinear self-interaction
        g2: float,         # Circulation coupling
        V0: float = 1.0,   # Three-well potential depth (GeV)
        m_eff: float = 1.0,  # Effective mass in subspace
        hbar: float = 1.0,   # Reduced Planck constant (natural units)
    ):
        """
        Initialize with fundamental parameters ONLY.
        
        NO phenomenological parameters allowed!
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        self.m_eff = m_eff
        self.hbar = hbar
    
    def compute_energy(
        self,
        A: float,
        delta_x: float,
        delta_sigma: float,
        k_eff: float
    ) -> EnergyBreakdown:
        """
        Compute E_total(A, Δx, Δσ) - the universal four-term energy.
        
        This is IDENTICAL for all particle types (leptons, baryons, mesons).
        The only particle-specific input is k_eff.
        
        Args:
            A: Amplitude (determines mass via m = β × A²)
            delta_x: Spatial localization scale
            delta_sigma: Subspace wavefunction width
            k_eff: Effective winding (from particle-specific wavefunction)
            
        Returns:
            EnergyBreakdown with all energy components
        """
        A_sq = A ** 2
        
        # === SUBSPACE ENERGY ===
        
        # Kinetic: ℏ²/(2m_eff) × A²/Δσ²
        E_kin = (self.hbar**2 / (2 * self.m_eff)) * A_sq / (delta_sigma**2)
        
        # Potential: V₀ × A²
        E_pot = self.V0 * A_sq
        
        # Nonlinear: (g₁/2) × A⁴/Δσ
        E_nl = (self.g1 / 2) * A_sq**2 / delta_sigma
        
        # Circulation: g₂ × k_eff² × A²
        E_circ = self.g2 * (k_eff ** 2) * A_sq
        
        E_subspace = E_kin + E_pot + E_nl + E_circ
        
        # === SPATIAL ENERGY ===
        # E_spatial = ℏ²/(2βA²Δx²)
        if A_sq > 1e-10 and delta_x > 1e-10:
            E_spatial = self.hbar**2 / (2 * self.beta * A_sq * delta_x**2)
        else:
            E_spatial = 1e10  # Penalty for invalid values
        
        # === COUPLING ENERGY ===
        # E_coupling = -α × k_eff × A
        # This is the SIMPLE theoretical form - no phenomenological scaling!
        E_coupling = -self.alpha * k_eff * A
        
        # === CURVATURE ENERGY ===
        # E_curvature = κ × (βA²)² / Δx
        if delta_x > 1e-10:
            mass_sq = (self.beta * A_sq) ** 2
            E_curvature = self.kappa * mass_sq / delta_x
        else:
            E_curvature = 1e10  # Penalty
        
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
        
        return EnergyBreakdown(
            E_total=E_total,
            E_subspace=E_subspace,
            E_spatial=E_spatial,
            E_coupling=E_coupling,
            E_curvature=E_curvature,
            E_kinetic=E_kin,
            E_potential=E_pot,
            E_nonlinear=E_nl,
            E_circulation=E_circ,
        )
    
    def compute_gradients(
        self,
        A: float,
        delta_x: float,
        delta_sigma: float,
        k_eff: float
    ) -> Tuple[float, float, float]:
        """
        Compute analytical gradients ∂E/∂A, ∂E/∂Δx, ∂E/∂Δσ.
        
        These are derived directly from the energy functional.
        No phenomenological parameters involved!
        
        Returns:
            (dE_dA, dE_ddelta_x, dE_ddelta_sigma)
        """
        A_sq = A ** 2
        
        # === ∂E/∂A ===
        
        # E_kin = (ℏ²/2m) × A²/Δσ² → dE_kin/dA = (ℏ²/m) × A/Δσ²
        dE_kin_dA = (self.hbar**2 / self.m_eff) * A / (delta_sigma**2)
        
        # E_pot = V₀ × A² → dE_pot/dA = 2V₀A
        dE_pot_dA = 2 * self.V0 * A
        
        # E_nl = (g₁/2) × A⁴/Δσ → dE_nl/dA = 2g₁A³/Δσ
        dE_nl_dA = 2 * self.g1 * A**3 / delta_sigma
        
        # E_circ = g₂k²A² → dE_circ/dA = 2g₂k²A
        dE_circ_dA = 2 * self.g2 * (k_eff**2) * A
        
        dE_subspace_dA = dE_kin_dA + dE_pot_dA + dE_nl_dA + dE_circ_dA
        
        # E_spatial = ℏ²/(2βA²Δx²) → dE_spatial/dA = -ℏ²/(βA³Δx²)
        if A > 1e-10 and delta_x > 1e-10:
            dE_spatial_dA = -self.hbar**2 / (self.beta * A**3 * delta_x**2)
        else:
            dE_spatial_dA = 0
        
        # E_coupling = -αk_eff×A → dE_coupling/dA = -αk_eff
        dE_coupling_dA = -self.alpha * k_eff
        
        # E_curvature = κ(βA²)²/Δx = κβ²A⁴/Δx → dE_curv/dA = 4κβ²A³/Δx
        if delta_x > 1e-10:
            dE_curvature_dA = 4 * self.kappa * self.beta**2 * A**3 / delta_x
        else:
            dE_curvature_dA = 0
        
        dE_dA = dE_subspace_dA + dE_spatial_dA + dE_coupling_dA + dE_curvature_dA
        
        # === ∂E/∂Δx ===
        
        # E_spatial = ℏ²/(2βA²Δx²) → dE_spatial/dΔx = -ℏ²/(βA²Δx³)
        # E_curvature = κβ²A⁴/Δx → dE_curv/dΔx = -κβ²A⁴/Δx²
        if A_sq > 1e-10 and delta_x > 1e-10:
            dE_spatial_ddx = -self.hbar**2 / (self.beta * A_sq * delta_x**3)
            dE_curvature_ddx = -self.kappa * self.beta**2 * A_sq**2 / (delta_x**2)
        else:
            dE_spatial_ddx = 0
            dE_curvature_ddx = 0
        
        dE_ddelta_x = dE_spatial_ddx + dE_curvature_ddx
        
        # === ∂E/∂Δσ ===
        
        # E_kin = (ℏ²/2m) × A²/Δσ² → dE_kin/dΔσ = -(ℏ²/m) × A²/Δσ³
        # E_nl = (g₁/2) × A⁴/Δσ → dE_nl/dΔσ = -(g₁/2) × A⁴/Δσ²
        if delta_sigma > 1e-10:
            dE_kin_dds = -(self.hbar**2 / self.m_eff) * A_sq / (delta_sigma**3)
            dE_nl_dds = -(self.g1 / 2) * A_sq**2 / (delta_sigma**2)
        else:
            dE_kin_dds = 0
            dE_nl_dds = 0
        
        dE_ddelta_sigma = dE_kin_dds + dE_nl_dds
        
        return (dE_dA, dE_ddelta_x, dE_ddelta_sigma)
    
    def minimize(
        self,
        k_eff: float,
        initial_A: float = 0.5,
        initial_dx: float = 1.0,
        initial_ds: float = 0.5,
        max_iter: int = 20000,
        tol: float = 1e-10,
        verbose: bool = False
    ) -> MinimizationResult:
        """
        Find equilibrium (A*, Δx*, Δσ*) by gradient descent.
        
        Minimizes E_total over all three variables simultaneously.
        
        Args:
            k_eff: Effective winding (from particle-specific wavefunction)
            initial_A: Starting amplitude
            initial_dx: Starting spatial extent
            initial_ds: Starting subspace width
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            MinimizationResult with equilibrium values and energy breakdown
        """
        # Initialize variables
        A = initial_A
        delta_x = initial_dx
        delta_sigma = initial_ds
        
        # Step sizes
        dt_A = 0.001
        dt_dx = 0.001
        dt_ds = 0.001
        
        # Initial energy
        energy = self.compute_energy(A, delta_x, delta_sigma, k_eff)
        E_old = energy.E_total
        
        converged = False
        final_residual = float('inf')
        
        for iteration in range(max_iter):
            # Compute gradients
            dE_dA, dE_ddx, dE_dds = self.compute_gradients(A, delta_x, delta_sigma, k_eff)
            
            # Gradient descent
            A_new = A - dt_A * dE_dA
            delta_x_new = delta_x - dt_dx * dE_ddx
            delta_sigma_new = delta_sigma - dt_ds * dE_dds
            
            # Enforce physical bounds
            A_new = max(A_new, 1e-6)
            delta_x_new = max(delta_x_new, 1e-6)
            delta_x_new = min(delta_x_new, 1e6)
            delta_sigma_new = max(delta_sigma_new, 0.01)
            delta_sigma_new = min(delta_sigma_new, 3.0)
            
            # Compute new energy
            energy_new = self.compute_energy(A_new, delta_x_new, delta_sigma_new, k_eff)
            E_new = energy_new.E_total
            
            # Adaptive step size
            if E_new > E_old:
                dt_A *= 0.5
                dt_dx *= 0.5
                dt_ds *= 0.5
                if dt_A < 1e-15:
                    break
                continue
            else:
                dt_A = min(dt_A * 1.02, 0.01)
                dt_dx = min(dt_dx * 1.02, 0.01)
                dt_ds = min(dt_ds * 1.02, 0.01)
            
            # Convergence check
            dE = abs(E_new - E_old)
            final_residual = dE
            
            if verbose and iteration % 2000 == 0:
                print(f"  Iter {iteration}: E={E_new:.6f}, A={A_new:.6f}, "
                      f"Δx={delta_x_new:.6g}, Δσ={delta_sigma_new:.4f}")
            
            if dE < tol:
                converged = True
                A, delta_x, delta_sigma = A_new, delta_x_new, delta_sigma_new
                energy = energy_new
                break
            
            A, delta_x, delta_sigma = A_new, delta_x_new, delta_sigma_new
            E_old = E_new
            energy = energy_new
        
        return MinimizationResult(
            A=A,
            A_squared=A**2,
            delta_x=delta_x,
            delta_sigma=delta_sigma,
            energy=energy,
            k_eff=k_eff,
            converged=converged,
            iterations=iteration + 1,
            final_residual=final_residual,
        )


def create_minimizer_from_constants() -> UniversalEnergyMinimizer:
    """
    Create minimizer using pure first-principles parameters.
    
    Uses the derived formulas:
    - κ = 1/β²
    - α = 0.5 × β
    - g₁ = α_em × β / m_e
    """
    from sfm_solver.core.sfm_global import SFM_CONSTANTS
    
    beta = SFM_CONSTANTS.beta_physical
    alpha_em = 1.0 / 137.036
    electron_mass_gev = 0.000511
    
    # Pure first-principles derivations
    kappa = 1.0 / (beta ** 2)
    alpha = 0.5 * beta
    g1 = alpha_em * beta / electron_mass_gev
    g2 = SFM_CONSTANTS.g2_alpha
    
    return UniversalEnergyMinimizer(
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        g1=g1,
        g2=g2,
        V0=1.0,
    )

