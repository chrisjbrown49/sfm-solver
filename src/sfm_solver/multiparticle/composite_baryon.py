"""
Composite Baryon Solver for SFM.

CORRECT PHYSICS:
- Single composite wavefunction, NOT three separate quarks
- Amplitude A² = ∫|χ|² is FREE (determines mass m = βA²)
- NO normalization - amplitude emerges from energy minimization  
- Color neutrality Σe^(iφᵢ) = 0 emerges from geometry/energy minimization
- Three-well potential localizes "quark" peaks
- QUARK TYPES (u/d) create different interference patterns → different masses

Energy functional to minimize:
    E[χ] = ∫[ℏ²/(2m)|∇χ|² + V(σ)|χ|² + (g₁/2)|χ|⁴] dσ + E_coulomb + E_coupling

The proton (uud) and neutron (udd) have:
- Different charge configurations → different Coulomb energies
- Different interference patterns in |χ(σ)|²
- Different integrated amplitudes A²_p vs A²_n
- Different masses: m = βA²

Reference: docs/Research Note - Origin of Strong Force.html
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators

# Quark charges (in units of e)
QUARK_CHARGES = {
    'u': 2/3,   # up quark: +2/3
    'd': -1/3,  # down quark: -1/3
}

# Standard baryon configurations
PROTON_QUARKS = ['u', 'u', 'd']   # uud → charge +1
NEUTRON_QUARKS = ['u', 'd', 'd']  # udd → charge 0


@dataclass
class CompositeBaryonState:
    """Result of composite baryon solver."""
    chi_baryon: NDArray[np.complexfloating]
    
    # Quark configuration
    quark_types: Tuple[str, str, str]  # e.g., ('u', 'u', 'd') for proton
    
    # Amplitude (NOT normalized to 1!)
    amplitude_squared: float  # A² = ∫|χ|² dσ - determines mass
    
    # Energy components
    energy_total: float
    energy_kinetic: float
    energy_potential: float
    energy_nonlinear: float
    energy_coupling: float  # -α×k×A term that stabilizes amplitude
    energy_coulomb: float   # Charge-dependent Coulomb energy
    
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
    - Coulomb energy (charge-dependent, differs for proton vs neutron)
    
    PROTON (uud) vs NEUTRON (udd):
    - Different quark charges → different Coulomb energies
    - Different interference patterns → different A²
    - Different masses: m = β×A² (use SFM_CONSTANTS.beta for consistency)
    
    NOTE ON β:
    The baryon solver computes the amplitude A² through energy minimization.
    To convert to mass, use: m = SFM_CONSTANTS.beta × A²
    This ensures consistency with the global β from the Beautiful Equation.
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
        coulomb_strength: float = 0.06,  # Coulomb coupling (tuned to n-p mass difference)
    ):
        self.grid = grid
        self.potential = potential
        self.g1 = g1
        self.g2 = g2        # Circulation coupling  
        self.alpha = alpha  # Coupling that stabilizes amplitude
        self.k = k
        self.m_eff = m_eff
        self.hbar = hbar
        self.coulomb_strength = coulomb_strength  # EM coupling
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
        
        # Current quark configuration (set during solve)
        self._quark_charges: List[float] = [QUARK_CHARGES['u']] * 3
        
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
    
    def _compute_coulomb_energy(self, chi: NDArray[np.complexfloating]) -> float:
        """
        Compute Coulomb energy from quark charges.
        
        E_coulomb = λ × Σ_{i<j} Q_i × Q_j × A_i × A_j
        
        where A_i is the amplitude in well i.
        
        This creates the proton-neutron mass difference:
        - Proton (uud): (+2/3)(+2/3) + (+2/3)(-1/3) + (+2/3)(-1/3) = 4/9 - 2/9 - 2/9 = 0
        - Neutron (udd): (+2/3)(-1/3) + (+2/3)(-1/3) + (-1/3)(-1/3) = -2/9 - 2/9 + 1/9 = -1/3
        
        Neutron has MORE NEGATIVE Coulomb energy (less repulsion),
        which leads to HIGHER amplitude at equilibrium → higher mass.
        """
        sigma = self.grid.sigma
        well_width = np.pi / 3
        
        # Compute amplitude in each well
        well_amplitudes = []
        for well_pos in self.WELL_POSITIONS:
            # Distance from well (periodic)
            diff = np.angle(np.exp(1j * (sigma - well_pos)))
            mask = np.abs(diff) < well_width
            A_i = np.sqrt(np.sum(np.abs(chi[mask])**2) * self.grid.dsigma) if np.any(mask) else 0.0
            well_amplitudes.append(A_i)
        
        # Coulomb sum: Σ_{i<j} Q_i × Q_j × A_i × A_j
        E_coulomb = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                E_coulomb += self._quark_charges[i] * self._quark_charges[j] * well_amplitudes[i] * well_amplitudes[j]
        
        return self.coulomb_strength * E_coulomb
    
    def _compute_energy(self, chi: NDArray[np.complexfloating]) -> Tuple[float, float, float, float, float, float]:
        """
        Compute total energy: E = E_kinetic + E_potential + E_nonlinear + E_coupling + E_coulomb
        
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
        
        # Coulomb energy (charge-dependent)
        E_coulomb = self._compute_coulomb_energy(chi)
        
        E_total = E_kin + E_pot + E_nl + E_circ + E_coupling + E_coulomb
        return E_total, E_kin, E_pot, E_nl, E_coupling, E_coulomb
    
    def _compute_coulomb_gradient(self, chi: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute gradient of Coulomb energy.
        
        δE_coulomb/δχ* = λ × Σ_{i<j} Q_i × Q_j × δ(A_i × A_j)/δχ*
        
        This is approximated by applying charge-weighted modulation.
        """
        sigma = self.grid.sigma
        well_width = np.pi / 3
        
        # Compute charge-weighted effective potential at each point
        # Approximate: gradient is proportional to χ weighted by local charge environment
        coulomb_weight = np.zeros_like(sigma)
        
        # For each point, compute its contribution to Coulomb energy
        for i, well_pos in enumerate(self.WELL_POSITIONS):
            diff = np.angle(np.exp(1j * (sigma - well_pos)))
            envelope_i = np.exp(-0.5 * (diff / (well_width / 2))**2)
            
            # Sum over pairs involving this well
            for j in range(3):
                if j != i:
                    # Other well envelope
                    other_pos = self.WELL_POSITIONS[j]
                    diff_j = np.angle(np.exp(1j * (sigma - other_pos)))
                    envelope_j = np.exp(-0.5 * (diff_j / (well_width / 2))**2)
                    
                    coulomb_weight += self._quark_charges[i] * self._quark_charges[j] * envelope_i * envelope_j
        
        return self.coulomb_strength * coulomb_weight * chi
    
    def _compute_gradient(self, chi: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """
        Compute energy gradient: δE/δχ* 
        
        Includes all terms:
        - Kinetic: δE_kin/δχ* = T χ
        - Potential: δE_pot/δχ* = V χ  
        - Nonlinear: δE_nl/δχ* = g₁|χ|²χ
        - Circulation: δE_circ/δχ* = g₂ × 2Re[J*∂/∂σ]χ (more complex)
        - Coupling: δE_coup/δχ* = -α×k/(2A) × χ
        - Coulomb: δE_coul/δχ* (charge-dependent)
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
        
        # Coulomb gradient
        coulomb_grad = self._compute_coulomb_gradient(chi)
        
        return T_chi + V_chi + NL_chi + circ_grad + coupling_grad + coulomb_grad
    
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
        quark_types: List[str] = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
        dt: float = 0.001,
        initial_amplitude: float = 1.0,
        verbose: bool = False
    ) -> CompositeBaryonState:
        """
        Minimize total energy using gradient descent.
        
        KEY: NO normalization! Amplitude is free to evolve.
        
        Args:
            quark_types: List of quark types for each well, e.g.:
                         ['u', 'u', 'd'] for proton (uud)
                         ['u', 'd', 'd'] for neutron (udd)
                         Defaults to proton configuration.
        
        Uses gradient descent: χ → χ - dt × δE/δχ*
        """
        # Set quark configuration
        if quark_types is None:
            quark_types = PROTON_QUARKS  # Default to proton
        
        if len(quark_types) != 3:
            raise ValueError(f"quark_types must have exactly 3 elements, got {len(quark_types)}")
        
        # Store quark charges for energy calculations
        self._quark_charges = [QUARK_CHARGES[q] for q in quark_types]
        
        chi = self._initialize_baryon(initial_amplitude)
        
        if verbose:
            A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
            phases, diffs, color_mag = self._extract_color_phases(chi)
            print(f"Baryon type: {''.join(quark_types)} (charges: {self._quark_charges})")
            print(f"Initial: A²={A2:.4f}, color_sum={color_mag:.4f}")
            print(f"  phases: {[f'{p:.3f}' for p in phases]}")
        
        E_old, _, _, _, _, _ = self._compute_energy(chi)
        converged = False
        residual = float('inf')
        
        for iteration in range(max_iter):
            # Gradient descent (NO normalization!)
            gradient = self._compute_gradient(chi)
            chi_new = chi - dt * gradient
            
            # Compute new energy
            E_new, E_kin, E_pot, E_nl, E_coup, E_coul = self._compute_energy(chi_new)
            
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
                      f"color={color_mag:.4f}, E_coul={E_coul:.4e}, dE={dE:.2e}")
            
            if dE < tol and residual < tol:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # Final results
        E_total, E_kin, E_pot, E_nl, E_coup, E_coul = self._compute_energy(chi)
        A2 = np.sum(np.abs(chi)**2) * self.grid.dsigma
        phases, diffs, color_mag = self._extract_color_phases(chi)
        
        return CompositeBaryonState(
            chi_baryon=chi,
            quark_types=tuple(quark_types),
            amplitude_squared=A2,
            energy_total=E_total,
            energy_kinetic=E_kin,
            energy_potential=E_pot,
            energy_nonlinear=E_nl,
            energy_coupling=E_coup,
            energy_coulomb=E_coul,
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
    quark_types: List[str] = None,
    **kwargs
) -> CompositeBaryonState:
    """Convenience function."""
    solver = CompositeBaryonSolver(grid, potential, g1)
    return solver.solve(quark_types=quark_types, **kwargs)


def solve_proton(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for proton (uud) configuration."""
    solver = CompositeBaryonSolver(grid, potential, **kwargs)
    return solver.solve(quark_types=PROTON_QUARKS)


def solve_neutron(grid: SpectralGrid, potential: ThreeWellPotential, **kwargs) -> CompositeBaryonState:
    """Solve for neutron (udd) configuration."""
    solver = CompositeBaryonSolver(grid, potential, **kwargs)
    return solver.solve(quark_types=NEUTRON_QUARKS)
