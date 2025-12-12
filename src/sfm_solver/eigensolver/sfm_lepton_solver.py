"""
SFM Lepton Solver - Pure First-Principles Implementation.

CRITICAL REQUIREMENTS (from Tier1_Lepton_Solver_Fix_Plan.md):
============================================================
A. ALL predictions must EMERGE from the first principles of the Single-Field
   Model theoretical framework. NO phenomenological parameters are permitted.

B. The solver uses ONLY the fundamental parameters already derived:
   - β: Fundamental mass scale (optimized for all particles)
   - κ = 1/β²: Curvature coupling
   - α = C × β: Spacetime-subspace coupling
   - g₁ = α_em × β / m_e: Nonlinear self-interaction
   - g₂ = α_em / 2: Circulation coupling
   - V₀ = 1.0 GeV: Three-well potential depth

These parameters are shared across ALL solvers (lepton, baryon, meson).
No solver-specific parameters are allowed.

ENERGY FUNCTIONAL (from "Research Note - A Beautiful Balance"):
===============================================================
E_total(A, Δx, Δσ) = E_subspace + E_spatial + E_coupling + E_curvature

We minimize over ALL THREE variables:
- A: Wavefunction amplitude (from χ)
- Δx: Spatial localization scale
- Δσ: Subspace wavefunction width

The mass hierarchy m_μ/m_e ≈ 207 and m_τ/m_e ≈ 3477 must EMERGE from
solving this energy minimization, not be imposed.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.sfm_global import SFM_CONSTANTS
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


# Lepton winding number - k=1 for all charged leptons
LEPTON_WINDING = 1

# Lepton spatial modes
LEPTON_SPATIAL_MODE = {
    'electron': 1,  # n = 1
    'muon': 2,      # n = 2
    'tau': 3,       # n = 3
}


@dataclass
class SFMLeptonState:
    """
    Result of SFM lepton solver.
    
    Contains the equilibrium values of all three optimization variables
    (A, Δx, Δσ) and the energy breakdown.
    """
    # Wavefunction and particle identity
    chi: NDArray[np.complexfloating]
    particle: str
    
    # The THREE optimization variables
    amplitude: float           # A = sqrt(A²)
    amplitude_squared: float   # A² - determines mass via m = β×A²
    delta_x: float            # Spatial localization (OPTIMIZED, not derived!)
    delta_sigma: float        # Subspace width (OPTIMIZED!)
    
    # Complete four-term energy breakdown
    energy_total: float
    energy_subspace: float
    energy_spatial: float
    energy_coupling: float
    energy_curvature: float
    
    # Subspace energy components
    energy_kinetic: float
    energy_potential: float
    energy_nonlinear: float
    energy_circulation: float
    
    # Winding structure
    k: int = LEPTON_WINDING
    k_eff: float = 1.0
    
    # Spatial mode
    n_spatial: int = 1
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 0.0


class SFMLeptonSolver:
    """
    Pure first-principles lepton solver.
    
    Minimizes E_total(A, Δx, Δσ) over ALL THREE variables simultaneously.
    
    NO PHENOMENOLOGICAL PARAMETERS - mass ratios must EMERGE from physics!
    """
    
    LEPTON_K = 1
    
    def __init__(
        self,
        grid: Optional[SpectralGrid] = None,
        potential: Optional[ThreeWellPotential] = None,
        g1: Optional[float] = None,
        g2: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        kappa: Optional[float] = None,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        c: float = 1.0,
        use_physical: Optional[bool] = None,
    ):
        """Initialize with first-principles parameters only."""
        if grid is None:
            grid = SpectralGrid(N=128)
        if potential is None:
            potential = ThreeWellPotential(V0=1.0, V1=0.1)
        
        self.grid = grid
        self.potential = potential
        
        if use_physical is None:
            use_physical = SFM_CONSTANTS.use_physical
        self.use_physical = use_physical
        
        self.g1 = g1 if g1 is not None else SFM_CONSTANTS.g1
        self.g2 = g2 if g2 is not None else SFM_CONSTANTS.g2_alpha
        
        if use_physical:
            self.alpha = alpha if alpha is not None else SFM_CONSTANTS.alpha_coupling_base
            self.beta = beta if beta is not None else SFM_CONSTANTS.beta_physical
            self.kappa = kappa if kappa is not None else SFM_CONSTANTS.kappa_physical
        else:
            self.alpha = alpha if alpha is not None else 2.5
            self.beta = beta if beta is not None else 1.0
            self.kappa = kappa if kappa is not None else 0.10
        
        self.m_eff = m_eff
        self.hbar = hbar
        self.c = c
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
    
    def _create_wavefunction(
        self,
        A: float,
        delta_sigma: float,
        well_pos: float = 0.0
    ) -> NDArray[np.complexfloating]:
        """
        Create parameterized wavefunction with specified amplitude and width.
        
        χ(σ) = A × exp(-(σ-σ₀)²/(2Δσ²)) × exp(ikσ)
        
        Args:
            A: Amplitude (determines mass via m = β×A²)
            delta_sigma: Width in subspace
            well_pos: Center position in subspace
            
        Returns:
            Wavefunction array
        """
        sigma = self.grid.sigma
        
        # Gaussian envelope with specified width
        dist = np.angle(np.exp(1j * (sigma - well_pos)))
        envelope = np.exp(-0.5 * (dist / delta_sigma)**2)
        
        # Winding factor e^(ikσ)
        winding = np.exp(1j * self.LEPTON_K * sigma)
        
        chi = envelope * winding
        
        # Normalize to get amplitude A
        current_amp_sq = np.sum(np.abs(chi)**2) * self.grid.dsigma
        if current_amp_sq > 1e-10:
            chi *= A / np.sqrt(current_amp_sq)
        
        return chi
    
    def _compute_k_eff(self, chi: NDArray[np.complexfloating]) -> float:
        """Compute effective winding from wavefunction gradient."""
        dchi = self.grid.first_derivative(chi)
        numerator = np.sum(np.abs(dchi)**2) * self.grid.dsigma
        denominator = np.sum(np.abs(chi)**2) * self.grid.dsigma
        
        if denominator < 1e-10:
            return 1.0
        return float(np.sqrt(numerator / denominator))
    
    def _compute_circulation(self, chi: NDArray[np.complexfloating]) -> complex:
        """Compute circulation: J = ∫ χ* (dχ/dσ) dσ"""
        dchi = self.grid.first_derivative(chi)
        return np.sum(np.conj(chi) * dchi) * self.grid.dsigma
    
    def _compute_energy(
        self,
        A: float,
        delta_x: float,
        delta_sigma: float,
        n_spatial: int
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Compute E_total(A, Δx, Δσ) - the full four-term energy.
        
        All three variables are independent inputs - NO slaving of Δx to A!
        
        Args:
            A: Amplitude
            delta_x: Spatial localization (independent variable!)
            delta_sigma: Subspace width (independent variable!)
            n_spatial: Spatial mode number (1, 2, or 3)
            
        Returns:
            (E_total, E_subspace, E_spatial, E_coupling, E_curvature,
             E_kin, E_pot, E_nl, E_circ, k_eff)
        """
        # Create wavefunction with specified A and Δσ
        chi = self._create_wavefunction(A, delta_sigma)
        
        A_sq = A ** 2
        k_eff = self._compute_k_eff(chi)
        
        # === SUBSPACE ENERGY ===
        
        # Kinetic: ℏ²/(2m_eff) × A²/Δσ²
        # For Gaussian: ⟨∇²⟩ ~ 1/Δσ²
        E_kin = (self.hbar**2 / (2 * self.m_eff)) * A_sq / (delta_sigma**2)
        
        # Potential: V₀ × A² (average potential energy)
        V0 = 1.0  # Three-well depth
        E_pot = V0 * A_sq
        
        # Nonlinear: (g₁/2) × A⁴ / Δσ
        # For |χ|⁴: integral ~ A⁴/Δσ for Gaussian
        E_nl = (self.g1 / 2) * A_sq**2 / delta_sigma
        
        # Circulation: g₂ × k² × A²
        E_circ = self.g2 * (self.LEPTON_K ** 2) * A_sq
        
        E_subspace = E_kin + E_pot + E_nl + E_circ
        
        # === SPATIAL ENERGY ===
        # E_spatial = ℏ²/(2βA²Δx²)
        if A_sq > 1e-10 and delta_x > 1e-10:
            E_spatial = self.hbar**2 / (2 * self.beta * A_sq * delta_x**2)
        else:
            E_spatial = 1e10  # Large penalty for invalid values
        
        # === COUPLING ENERGY ===
        # E_coupling = -α × n × k × A
        E_coupling = -self.alpha * n_spatial * self.LEPTON_K * A
        
        # === CURVATURE ENERGY ===
        # E_curvature = κ × (βA²)² / Δx
        if delta_x > 1e-10:
            mass_sq = (self.beta * A_sq) ** 2
            E_curvature = self.kappa * mass_sq / delta_x
        else:
            E_curvature = 1e10  # Large penalty
        
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
        
        return (E_total, E_subspace, E_spatial, E_coupling, E_curvature,
                E_kin, E_pot, E_nl, E_circ, k_eff)
    
    def _compute_gradients(
        self,
        A: float,
        delta_x: float,
        delta_sigma: float,
        n_spatial: int
    ) -> Tuple[float, float, float]:
        """
        Compute gradients ∂E/∂A, ∂E/∂Δx, ∂E/∂Δσ analytically.
        
        Returns:
            (dE_dA, dE_ddelta_x, dE_ddelta_sigma)
        """
        A_sq = A ** 2
        
        # === ∂E_subspace/∂A ===
        # E_kin = (ℏ²/2m) × A²/Δσ² → dE_kin/dA = (ℏ²/m) × A/Δσ²
        dE_kin_dA = (self.hbar**2 / self.m_eff) * A / (delta_sigma**2)
        
        # E_pot = V₀ × A² → dE_pot/dA = 2V₀A
        V0 = 1.0
        dE_pot_dA = 2 * V0 * A
        
        # E_nl = (g₁/2) × A⁴/Δσ → dE_nl/dA = 2g₁A³/Δσ
        dE_nl_dA = 2 * self.g1 * A**3 / delta_sigma
        
        # E_circ = g₂k²A² → dE_circ/dA = 2g₂k²A
        dE_circ_dA = 2 * self.g2 * (self.LEPTON_K**2) * A
        
        dE_subspace_dA = dE_kin_dA + dE_pot_dA + dE_nl_dA + dE_circ_dA
        
        # === ∂E_spatial/∂A ===
        # E_spatial = ℏ²/(2βA²Δx²) → dE_spatial/dA = -ℏ²/(βA³Δx²)
        if A > 1e-10 and delta_x > 1e-10:
            dE_spatial_dA = -self.hbar**2 / (self.beta * A**3 * delta_x**2)
        else:
            dE_spatial_dA = 0
        
        # === ∂E_coupling/∂A ===
        # E_coupling = -αnkA → dE_coupling/dA = -αnk
        dE_coupling_dA = -self.alpha * n_spatial * self.LEPTON_K
        
        # === ∂E_curvature/∂A ===
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
    
    def solve_lepton(
        self,
        particle: str = 'electron',
        max_iter: int = 10000,
        tol: float = 1e-10,
        verbose: bool = False
    ) -> SFMLeptonState:
        """
        Solve for a lepton by minimizing E_total(A, Δx, Δσ).
        
        All three variables are optimized simultaneously!
        
        Args:
            particle: 'electron', 'muon', or 'tau'
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            SFMLeptonState with equilibrium values.
        """
        n_spatial = LEPTON_SPATIAL_MODE.get(particle, 1)
        
        # Initial values for the THREE optimization variables
        A = 0.5           # Initial amplitude
        delta_x = 1.0     # Initial spatial scale
        delta_sigma = 0.5 # Initial subspace width
        
        # Step sizes for each variable
        dt_A = 0.001
        dt_dx = 0.001
        dt_ds = 0.001
        
        if verbose:
            print("=" * 60)
            print(f"SFM LEPTON SOLVER: {particle.upper()}")
            print(f"  Spatial mode n = {n_spatial}")
            print(f"  Optimizing E_total(A, Δx, Δσ) over ALL THREE variables")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}, κ={self.kappa:.6g}")
            print("=" * 60)
        
        # Initial energy
        result = self._compute_energy(A, delta_x, delta_sigma, n_spatial)
        E_old = result[0]
        
        converged = False
        final_residual = float('inf')
        
        for iteration in range(max_iter):
            # Compute gradients
            dE_dA, dE_ddx, dE_dds = self._compute_gradients(A, delta_x, delta_sigma, n_spatial)
            
            # Gradient descent on all three variables
            A_new = A - dt_A * dE_dA
            delta_x_new = delta_x - dt_dx * dE_ddx
            delta_sigma_new = delta_sigma - dt_ds * dE_dds
            
            # Enforce physical bounds
            A_new = max(A_new, 1e-6)
            delta_x_new = max(delta_x_new, 1e-6)
            delta_x_new = min(delta_x_new, 1e6)
            delta_sigma_new = max(delta_sigma_new, 0.01)
            delta_sigma_new = min(delta_sigma_new, 3.0)  # Can't be wider than the period
            
            # Compute new energy
            result = self._compute_energy(A_new, delta_x_new, delta_sigma_new, n_spatial)
            E_new = result[0]
            
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
            
            if verbose and iteration % 1000 == 0:
                print(f"  Iter {iteration}: E={E_new:.6f}, A={A_new:.6f}, "
                      f"Δx={delta_x_new:.6g}, Δσ={delta_sigma_new:.4f}, dE={dE:.2e}")
            
            if dE < tol:
                converged = True
                A, delta_x, delta_sigma = A_new, delta_x_new, delta_sigma_new
                break
            
            A, delta_x, delta_sigma = A_new, delta_x_new, delta_sigma_new
            E_old = E_new
        
        # Final energy computation
        result = self._compute_energy(A, delta_x, delta_sigma, n_spatial)
        (E_total, E_subspace, E_spatial, E_coupling, E_curvature,
         E_kin, E_pot, E_nl, E_circ, k_eff) = result
        
        A_sq = A ** 2
        
        # Create wavefunction for output
        chi = self._create_wavefunction(A, delta_sigma)
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"RESULTS: {particle.upper()}")
            print(f"  Amplitude A = {A:.6f}")
            print(f"  Amplitude A² = {A_sq:.6f}")
            print(f"  Δx = {delta_x:.6g} (OPTIMIZED)")
            print(f"  Δσ = {delta_sigma:.6f} (OPTIMIZED)")
            print(f"  E_total = {E_total:.6f}")
            print(f"    E_subspace = {E_subspace:.6f}")
            print(f"      E_kinetic = {E_kin:.6f}")
            print(f"      E_potential = {E_pot:.6f}")
            print(f"      E_nonlinear = {E_nl:.6f}")
            print(f"      E_circulation = {E_circ:.6f}")
            print(f"    E_spatial = {E_spatial:.6f}")
            print(f"    E_coupling = {E_coupling:.6f}")
            print(f"    E_curvature = {E_curvature:.6f}")
            print(f"  Converged: {converged} ({iteration+1} iterations)")
            print("=" * 60)
        
        return SFMLeptonState(
            chi=chi,
            particle=particle,
            amplitude=A,
            amplitude_squared=float(A_sq),
            delta_x=float(delta_x),
            delta_sigma=float(delta_sigma),
            energy_total=float(E_total),
            energy_subspace=float(E_subspace),
            energy_spatial=float(E_spatial),
            energy_coupling=float(E_coupling),
            energy_curvature=float(E_curvature),
            energy_kinetic=float(E_kin),
            energy_potential=float(E_pot),
            energy_nonlinear=float(E_nl),
            energy_circulation=float(E_circ),
            k=self.LEPTON_K,
            k_eff=float(k_eff),
            n_spatial=n_spatial,
            converged=converged,
            iterations=iteration + 1,
            final_residual=float(final_residual),
        )
    
    def solve_electron(self, **kwargs) -> SFMLeptonState:
        """Solve for electron (n=1)."""
        return self.solve_lepton(particle='electron', **kwargs)
    
    def solve_muon(self, **kwargs) -> SFMLeptonState:
        """Solve for muon (n=2)."""
        return self.solve_lepton(particle='muon', **kwargs)
    
    def solve_tau(self, **kwargs) -> SFMLeptonState:
        """Solve for tau (n=3)."""
        return self.solve_lepton(particle='tau', **kwargs)
    
    def solve_lepton_spectrum(
        self,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, SFMLeptonState]:
        """
        Solve for all three charged leptons.
        
        Returns:
            Dictionary with 'electron', 'muon', 'tau' states.
        """
        results = {}
        particles = ['electron', 'muon', 'tau']
        
        if verbose:
            print("=" * 60)
            print("SFM LEPTON SPECTRUM SOLVER")
            print("  Minimizing E_total(A, Δx, Δσ) over ALL THREE variables")
            print("  NO phenomenological parameters!")
            print("=" * 60)
        
        for particle in particles:
            if verbose:
                print(f"\n--- Solving for {particle.upper()} ---")
            state = self.solve_lepton(particle=particle, verbose=verbose, **kwargs)
            results[particle] = state
        
        # Display mass ratios
        if verbose:
            e_A2 = results['electron'].amplitude_squared
            mu_A2 = results['muon'].amplitude_squared
            tau_A2 = results['tau'].amplitude_squared
            
            print("\n" + "=" * 60)
            print("EMERGENT MASS RATIOS:")
            print(f"  A²_e = {e_A2:.6f}")
            print(f"  A²_μ = {mu_A2:.6f}")
            print(f"  A²_τ = {tau_A2:.6f}")
            
            if e_A2 > 1e-10:
                print(f"\n  m_μ/m_e = {mu_A2/e_A2:.4f} (target: 206.768)")
                print(f"  m_τ/m_e = {tau_A2/e_A2:.4f} (target: 3477.23)")
                if mu_A2 > 1e-10:
                    print(f"  m_τ/m_μ = {tau_A2/mu_A2:.4f} (target: 16.817)")
            print("=" * 60)
        
        return results
    
    def compute_mass_ratios(
        self,
        results: Dict[str, SFMLeptonState]
    ) -> Dict[str, float]:
        """Compute mass ratios from solver results."""
        e_A2 = results['electron'].amplitude_squared
        mu_A2 = results['muon'].amplitude_squared
        tau_A2 = results['tau'].amplitude_squared
        
        return {
            'mu_e': mu_A2 / e_A2 if e_A2 > 1e-10 else 0.0,
            'tau_e': tau_A2 / e_A2 if e_A2 > 1e-10 else 0.0,
            'tau_mu': tau_A2 / mu_A2 if mu_A2 > 1e-10 else 0.0,
        }


def solve_lepton_masses(verbose: bool = True) -> Dict[str, float]:
    """
    Convenience function to solve for lepton mass ratios.
    """
    solver = SFMLeptonSolver()
    results = solver.solve_lepton_spectrum(verbose=verbose)
    ratios = solver.compute_mass_ratios(results)
    
    return {
        'A2_e': results['electron'].amplitude_squared,
        'A2_mu': results['muon'].amplitude_squared,
        'A2_tau': results['tau'].amplitude_squared,
        'm_mu/m_e': ratios['mu_e'],
        'm_tau/m_e': ratios['tau_e'],
        'm_tau/m_mu': ratios['tau_mu'],
    }
