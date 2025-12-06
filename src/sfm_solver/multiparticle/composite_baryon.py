"""
Composite Baryon Solver for SFM.

CORRECT PHYSICS:
- Single composite wavefunction, NOT three separate quarks
- Amplitude A² = ∫|χ|² is FREE (determines mass m = βA²)
- NO normalization - amplitude emerges from energy minimization  
- Color neutrality Σe^(iφᵢ) = 0 emerges from geometry/energy minimization
- Three-well potential localizes "quark" peaks

Energy functional to minimize:
    E[χ] = ∫[ℏ²/(2m)|∇χ|² + V(σ)|χ|² + (g₁/2)|χ|⁴] dσ

Gradient for minimization:
    δE/δχ* = -ℏ²/(2m)∇²χ + V(σ)χ + g₁|χ|²χ

The amplitude equilibrium comes from:
    E = A² × (kinetic + potential) + A⁴ × (nonlinear)
    dE/dA² = 0  →  determines A²

Reference: docs/Research Note - Origin of Strong Force.html
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


@dataclass
class CompositeBaryonState:
    """Result of composite baryon solver."""
    chi_baryon: NDArray[np.complexfloating]
    
    # Amplitude (NOT normalized to 1!)
    amplitude_squared: float  # A² = ∫|χ|² dσ - determines mass
    
    # Energy components
    energy_total: float
    energy_kinetic: float
    energy_potential: float
    energy_nonlinear: float
    energy_coupling: float  # -α×k×A term that stabilizes amplitude
    
    # Color structure
    phases: Tuple[float, float, float]
    phase_differences: Tuple[float, float]
    color_sum_magnitude: float
    is_color_neutral: bool
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float


class CompositeBaryonSolver:
    """
    Solver for baryon as a single composite wavefunction.
    
    KEY: Does NOT normalize! Amplitude emerges from energy minimization.
    
    The equilibrium amplitude comes from balancing:
    - Potential energy (favors larger amplitude in wells)
    - Nonlinear energy (penalizes large amplitude for g₁ > 0)
    - Kinetic energy (penalizes sharp gradients)
    """
    
    WELL_POSITIONS = [0.0, 2*np.pi/3, 4*np.pi/3]
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g1: float = 0.1,
        g2: float = 0.1,    # Circulation coupling (EM-like)
        alpha: float = 1.0,  # Subspace-spacetime coupling
        k: int = 3,
        m_eff: float = 1.0,
        hbar: float = 1.0,
    ):
        self.grid = grid
        self.potential = potential
        self.g1 = g1
        self.g2 = g2        # Circulation coupling  
        self.alpha = alpha  # Coupling that stabilizes amplitude
        self.k = k
        self.m_eff = m_eff
        self.hbar = hbar
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
        
    def _initialize_baryon(self, initial_amplitude: float = 1.0) -> NDArray[np.complexfloating]:
        """
        Initialize baryon wavefunction with three peaks at wells.
        
        Each peak has:
        - Gaussian envelope localized at well
        - Winding factor e^(ikσ) with k=3
        - Color phase: 0, 2π/3, 4π/3 for color neutrality
        
        The initial_amplitude sets the overall scale.
        """
        sigma = self.grid.sigma
        N = len(sigma)
        
        chi = np.zeros(N, dtype=complex)
        width = 0.5  # Localization width
        
        for i, well_pos in enumerate(self.WELL_POSITIONS):
            # Color phase for neutrality
            color_phase = i * 2 * np.pi / 3
            
            # Gaussian at well (handle periodicity)
            dist = np.angle(np.exp(1j * (sigma - well_pos)))
            envelope = np.exp(-0.5 * (dist / width)**2)
            
            # Full phase: winding + color
            phase = self.k * sigma + color_phase
            
            chi += envelope * np.exp(1j * phase)
        
        # Scale to desired initial amplitude (NOT normalizing!)
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        chi *= np.sqrt(initial_amplitude / current_amp_sq)
        
        return chi
    
    def _compute_energy(self, chi: NDArray[np.complexfloating]) -> Tuple[float, float, float, float, float]:
        """
        Compute total energy: E = E_kinetic + E_potential + E_nonlinear + E_coupling
        
        CRITICAL: E_coupling scales as -A (linear in amplitude), which stabilizes
        the system against collapse to zero amplitude.
        
        NO normalization constraints!
        """
        # Amplitude
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(A_sq)
        
        # Kinetic: ∫(ℏ²/2m)|∇χ|² dσ
        T_chi = self.operators.apply_kinetic(chi)
        E_kin = np.real(self.grid.inner_product(chi, T_chi))
        
        # Potential: ∫V(σ)|χ|² dσ
        E_pot = np.real(np.sum(self._V_grid * np.abs(chi)**2) * self.grid.dsigma)
        
        # Nonlinear: (g₁/2)∫|χ|⁴ dσ
        E_nl = (self.g1 / 2) * np.sum(np.abs(chi)**4) * self.grid.dsigma
        
        # Circulation (EM-like): g₂|J|² where J = ∫χ*∂χ/∂σ dσ ≈ ik × A²
        # For three quarks with k=3, J ≈ 3ik × A²
        dchi = self.grid.first_derivative(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        E_circ = self.g2 * np.abs(J)**2
        
        # Coupling energy: -α × k × A (NEGATIVE - provides stability!)
        # For baryons, this represents the subspace-spacetime coupling
        # that creates the bound state
        E_coupling = -self.alpha * self.k * A
        
        E_total = E_kin + E_pot + E_nl + E_circ + E_coupling
        return E_total, E_kin, E_pot, E_nl, E_coupling
    
    def _compute_gradient(self, chi: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute energy gradient: δE/δχ* 
        
        Includes all terms:
        - Kinetic: δE_kin/δχ* = T χ
        - Potential: δE_pot/δχ* = V χ  
        - Nonlinear: δE_nl/δχ* = g₁|χ|²χ
        - Circulation: δE_circ/δχ* = g₂ × 2Re[J*∂/∂σ]χ (more complex)
        - Coupling: δE_coup/δχ* = -α×k/(2A) × χ
        """
        # Kinetic + Potential + Nonlinear (standard terms)
        T_chi = self.operators.apply_kinetic(chi)
        V_chi = self._V_grid * chi
        NL_chi = self.g1 * np.abs(chi)**2 * chi
        
        # Circulation gradient (g₂|J|² where J = ∫χ*∂χ/∂σ)
        # δ|J|²/δχ* = 2Re[J* × δJ/δχ*] = 2Re[J* × ∂χ/∂σ]
        dchi = self.grid.first_derivative(chi)
        J = np.sum(np.conj(chi) * dchi) * self.grid.dsigma
        circ_grad = 2 * self.g2 * np.real(np.conj(J)) * dchi
        
        # Coupling gradient: δ(-α×k×A)/δχ* = -α×k/(2A) × χ
        A_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        A = np.sqrt(max(A_sq, 1e-10))
        coupling_grad = -self.alpha * self.k / (2 * A) * chi
        
        return T_chi + V_chi + NL_chi + circ_grad + coupling_grad
    
    def _extract_color_phases(self, chi: NDArray[np.complexfloating]) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
        """Extract phases at well positions."""
        sigma = self.grid.sigma
        phases = []
        
        for well_pos in self.WELL_POSITIONS:
            idx = np.argmin(np.abs(sigma - well_pos))
            # Remove winding to get color phase
            raw_phase = np.angle(chi[idx])
            color_phase = (raw_phase - self.k * well_pos) % (2 * np.pi)
            phases.append(float(color_phase))
        
        # Phase differences (should be ~2π/3)
        diff1 = (phases[1] - phases[0]) % (2 * np.pi)
        diff2 = (phases[2] - phases[1]) % (2 * np.pi)
        
        # Color sum: Σe^(iφᵢ) should be ~0
        color_sum = sum(np.exp(1j * phi) for phi in phases)
        
        return tuple(phases), (diff1, diff2), abs(color_sum)
    
    def solve(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        dt: float = 0.001,
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> CompositeBaryonState:
        """
        Minimize total energy using gradient descent.
        
        KEY: NO normalization! Amplitude is free to evolve.
        
        Uses gradient descent: χ → χ - dt × δE/δχ*
        """
        chi = self._initialize_baryon(initial_amplitude)
        
        if verbose:
            A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
            phases, diffs, color_mag = self._extract_color_phases(chi)
            print(f"Initial: A²={A2:.4f}, color_sum={color_mag:.4f}")
            print(f"  phases: {[f'{p:.3f}' for p in phases]}")
        
        E_old, _, _, _, _ = self._compute_energy(chi)
        converged = False
        
        for iteration in range(max_iter):
            # Gradient descent (NO normalization!)
            gradient = self._compute_gradient(chi)
            chi_new = chi - dt * gradient
            
            # Compute new energy
            E_new, E_kin, E_pot, E_nl, E_coup = self._compute_energy(chi_new)
            
            # Check for energy increase (reduce step size if needed)
            if E_new > E_old:
                dt *= 0.5
                if dt < 1e-10:
                    if verbose:
                        print(f"Step size too small at iteration {iteration}")
                    break
                continue
            
            # Update
            residual = np.sqrt(np.sum(np.abs(chi_new - chi)**2) * self.grid.dsigma)
            chi = chi_new
            dE = E_old - E_new
            E_old = E_new
            
            if verbose and iteration % 100 == 0:
                A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
                _, _, color_mag = self._extract_color_phases(chi)
                print(f"Iter {iteration}: E={E_new:.4f}, A²={A2:.4f}, "
                      f"color={color_mag:.4f}, dE={dE:.2e}")
            
            if dE < tol and residual < tol:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # Final results
        E_total, E_kin, E_pot, E_nl, E_coup = self._compute_energy(chi)
        A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
        phases, diffs, color_mag = self._extract_color_phases(chi)
        
        return CompositeBaryonState(
            chi_baryon=chi,
            amplitude_squared=A2,
            energy_total=E_total,
            energy_kinetic=E_kin,
            energy_potential=E_pot,
            energy_nonlinear=E_nl,
            energy_coupling=E_coup,
            phases=phases,
            phase_differences=diffs,
            color_sum_magnitude=color_mag,
            is_color_neutral=color_mag < 0.1,
            converged=converged,
            iterations=iteration + 1,
            final_residual=residual
        )


def solve_composite_baryon(
    grid: SpectralGrid,
    potential: ThreeWellPotential,
    g1: float = 0.1,
    **kwargs
) -> CompositeBaryonState:
    """Convenience function."""
    solver = CompositeBaryonSolver(grid, potential, g1)
    return solver.solve(**kwargs)
