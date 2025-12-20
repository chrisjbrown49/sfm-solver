"""
DEPRECATED: Legacy Universal Energy Minimizer for SFM Particles.

=============================================================================
WARNING: This module has been superseded by the new two-stage architecture:
    - Stage 1: DimensionlessShapeSolver (shape_solver.py)
    - Stage 2: UniversalEnergyMinimizer (energy_minimizer.py)
    - Combined: UnifiedSFMSolver (unified_solver.py)

This legacy implementation is kept for reference only.
Please use the new unified solver for all new work.
=============================================================================

This module minimizes E_total(A, Δx, Δσ) over the three free scale parameters
for ANY wavefunction structure from the NonSeparableWavefunctionSolver.

Key principle: The wavefunction STRUCTURE (from the non-separable solver)
determines what kind of particle it is. The SCALE parameters (A, Δx, Δσ)
determine how big/massive it is.

Energy functional (UNIVERSAL for all particles):
    E_total = E_subspace + E_spatial + E_coupling + E_curvature

Where:
    E_subspace = E_kinetic + E_potential + E_nonlinear + E_circulation
    E_kinetic   = (ℏ²/2m) × (A²/Δσ²) × I_kin
    E_potential = V₀ × A² × I_pot
    E_nonlinear = (g₁/2) × (A⁴/Δσ) × I_NL
    E_circulation = g₂ × |J|² × A⁴
    E_spatial   = ℏ²/(2βA²Δx²)
    E_coupling  = -α × spatial_factor(n, Δx) × subspace_factor × A
    E_curvature = κ × (βA²)² / Δx

CRITICAL PHYSICS:
    The spatial wavefunction φ_n(r; Δx) has:
    - n-1 radial nodes (determined by quantum number n)
    - Scale parameter = Δx/√(2n+1) (determined by optimization variable Δx)
    
    The spatial_factor is computed dynamically as Δx changes during optimization.

This is Phase 1 of the refactor - the universal minimizer.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.special import genlaguerre

from sfm_solver.core.nonseparable_wavefunction_solver import WavefunctionStructure


@dataclass
class EnergyMinimizationResult:
    """Result from universal energy minimization."""
    
    # Optimized scale parameters
    A: float
    A_squared: float
    delta_x: float
    delta_sigma: float
    
    # Predicted mass
    mass_gev: float
    
    # Energy breakdown
    E_total: float
    E_kinetic: float
    E_potential: float
    E_nonlinear: float
    E_circulation: float
    E_spatial: float
    E_coupling: float
    E_curvature: float
    
    # Input structure info
    n_target: int
    k_winding: int
    particle_type: str
    
    # Convergence
    converged: bool
    iterations: int
    
    # Physics checks
    compton_check: float  # Δx × β × A² (should be ~1 if Compton emerges)


class UniversalEnergyMinimizer:
    """
    Universal energy minimizer for all SFM particles.
    
    Given a converged wavefunction structure {χ_{nlm}(σ)} from the
    NonSeparableWavefunctionSolver, find the optimal (A, Δx, Δσ) that
    minimizes the total energy.
    
    The SAME code works for leptons, mesons, and baryons!
    
    The wavefunction structure encodes particle-specific physics.
    This minimizer applies UNIVERSAL energy balance.
    """
    
    def __init__(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        g1: float,
        g2: float,
        V0: float = 1.0,
        m_eff: float = 1.0,
        hbar: float = 1.0,
    ):
        """
        Initialize with universal SFM parameters.
        
        Args:
            alpha: Spatial-subspace coupling strength (GeV)
            beta: Mass coupling constant (GeV)
            kappa: Curvature coupling (GeV⁻²)
            g1: Nonlinear self-interaction strength
            g2: Circulation coupling
            V0: Three-well potential depth (GeV)
            m_eff: Effective mass in subspace
            hbar: Reduced Planck constant (natural units = 1)
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        self.m_eff = m_eff
        self.hbar = hbar
        
        # Radial grid for spatial wavefunction integrals
        self._r_grid = np.linspace(0.01, 20.0, 500)
    
    def _compute_spatial_factor(self, n: int, delta_x: float) -> float:
        """
        Compute the spatial coupling factor for harmonic oscillator wavefunction.
        
        CRITICAL PHYSICS:
        - n determines the number of radial nodes (n-1 nodes for s-wave)
        - delta_x determines the spatial scale: a = delta_x / sqrt(2n+1)
        
        The spatial wavefunction is:
            φ_n(r; Δx) = N × L_{n-1}^{1/2}(r²/a²) × exp(-r²/(2a²))
        
        where a = Δx/√(2n+1).
        
        The CORRECT spatial coupling factor (from Math Formulation Part A) is:
            ∫ |∇φ|² d³x = 4π ∫ (dφ/dr)² r² dr
        
        This is the SQUARED GRADIENT, which:
        - Is non-zero for all states (unlike ∫φ×dφ/dr which is always zero!)
        - Scales with n (more nodes → larger gradients)
        - Scales with 1/a² = (2n+1)/Δx² (tighter confinement → stronger gradients)
        
        Args:
            n: Spatial quantum number (1 for electron, 2 for muon, 3 for tau)
            delta_x: Spatial extent (optimization variable)
            
        Returns:
            Spatial coupling factor (always positive, scales as 1/Δx²)
        """
        if delta_x < 1e-10:
            return 1e10  # Return large value for small delta_x
        
        # Scale parameter: a = Δx / √(2n+1)
        # This ensures the characteristic size is Δx, adjusted for the number of nodes
        a = delta_x / np.sqrt(2 * n + 1)
        
        if a < 1e-10:
            return 1e10
        
        r = self._r_grid
        x = (r / a) ** 2
        
        # Laguerre polynomial parameters for s-wave (l=0) harmonic oscillator
        # φ_n has n-1 radial nodes, corresponding to L_{n-1}^{1/2}(x)
        k = n - 1  # Polynomial degree
        alpha_lag = 0.5  # For 3D harmonic oscillator s-wave
        
        # Compute Laguerre polynomial and its derivative
        if k >= 0:
            L = genlaguerre(k, alpha_lag)(x)
            if k >= 1:
                dL_dx = -genlaguerre(k - 1, alpha_lag + 1)(x)
            else:
                dL_dx = np.zeros_like(x)
        else:
            L = np.ones_like(x)
            dL_dx = np.zeros_like(x)
        
        # Radial wavefunction (unnormalized)
        # φ = L(x) × exp(-x/2)  where x = (r/a)²
        exp_factor = np.exp(-x / 2)
        phi = L * exp_factor
        
        # Derivative: dφ/dr = dφ/dx × dx/dr = dφ/dx × (2r/a²)
        # dφ/dx = dL/dx × exp(-x/2) + L × (-1/2) × exp(-x/2)
        #       = exp(-x/2) × [dL/dx - L/2]
        dphi_dx = exp_factor * (dL_dx - L / 2)
        dphi_dr = dphi_dx * (2 * r / a**2)
        
        # Normalize: ∫ 4π φ² r² dr = 1
        norm_sq = 4 * np.pi * np.trapz(phi**2 * r**2, r)
        if norm_sq > 1e-20:
            phi = phi / np.sqrt(norm_sq)
            dphi_dr = dphi_dr / np.sqrt(norm_sq)
        
        # CORRECT spatial coupling factor: ∫ |∇φ|² d³x = 4π ∫ (dφ/dr)² r² dr
        # This is the SQUARED GRADIENT which is always positive and non-zero!
        # Scales as 1/a² = (2n+1)/Δx²
        spatial_factor = 4 * np.pi * np.trapz(dphi_dr**2 * r**2, r)
        
        return float(spatial_factor)
    
    def _compute_subspace_factor(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        sigma_grid: NDArray,
    ) -> float:
        """
        Compute the subspace coupling factor from the entangled wavefunction.
        
        subspace_factor = Im[∫χ*(∂χ/∂σ)dσ]
        
        This is computed from the ACTUAL entangled structure once,
        then combined with spatial_factor during optimization.
        """
        dsigma = sigma_grid[1] - sigma_grid[0]
        N = len(sigma_grid)
        
        # First derivative matrix
        D1 = np.zeros((N, N), dtype=complex)
        for i in range(N):
            D1[i, (i+1) % N] = 1.0
            D1[i, (i-1) % N] = -1.0
        D1 = D1 / (2 * dsigma)
        
        # Sum all chi components for total wavefunction
        chi_total = np.zeros(N, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        
        # Compute ∂χ/∂σ
        dchi_total = D1 @ chi_total
        
        # Subspace factor = Im[∫χ*(∂χ/∂σ)dσ]
        # The imaginary part carries the winding information
        integral = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        subspace_factor = np.imag(integral)
        
        return float(subspace_factor)
    
    def _compute_shape_integrals(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        sigma_grid: NDArray,
        V_sigma: NDArray,
    ) -> Dict[str, float]:
        """
        Compute shape-dependent integrals from the entangled wavefunction.
        
        These are computed ONCE from the unit-normalized structure,
        then scaled by (A, Δσ) during minimization.
        """
        dsigma = sigma_grid[1] - sigma_grid[0]
        
        # First derivative matrix
        N = len(sigma_grid)
        D1 = np.zeros((N, N), dtype=complex)
        for i in range(N):
            D1[i, (i+1) % N] = 1.0
            D1[i, (i-1) % N] = -1.0
        D1 = D1 / (2 * dsigma)
        
        # Sum all chi components for total wavefunction
        chi_total = np.zeros(N, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        
        # === Shape integrals (for unit-normalized structure) ===
        
        # I_kinetic = Σ_{nlm} ∫|∂χ_{nlm}/∂σ|² dσ
        I_kinetic = 0.0
        for chi in chi_components.values():
            dchi = D1 @ chi
            I_kinetic += np.real(np.sum(np.abs(dchi)**2)) * dsigma
        
        # I_potential = Σ_{nlm} ∫V(σ)|χ_{nlm}|² dσ (normalized by V0)
        I_potential = 0.0
        V_normalized = V_sigma / self.V0 if self.V0 > 0 else V_sigma
        for chi in chi_components.values():
            I_potential += np.real(np.sum(V_normalized * np.abs(chi)**2)) * dsigma
        
        # I_nonlinear = ∫|χ_total|⁴ dσ
        I_nonlinear = np.sum(np.abs(chi_total)**4) * dsigma
        
        # J = ∫χ_total*(∂χ_total/∂σ)dσ (for circulation)
        dchi_total = D1 @ chi_total
        J = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        
        return {
            'I_kinetic': float(I_kinetic),
            'I_potential': float(I_potential),
            'I_nonlinear': float(I_nonlinear),
            'J': J,  # Complex
        }
    
    def _compute_coupling_energy_from_structure(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        spatial_coupling: NDArray,
        sigma_grid: NDArray,
        state_keys: list,
    ) -> float:
        """
        Compute the coupling integral factor from the entangled wavefunction.
        
        E_coupling_factor = Σ_{i,j} |R_{ij}| × Im[∫χ_i*(∂χ_j/∂σ)dσ]
        
        This is computed from the ACTUAL entangled structure!
        The full coupling energy is: E_coupling = -α × factor × A
        
        NOTE: state_keys are (n,l,m) tuples but spatial_coupling uses indices.
        We need to build an index mapping based on the state ordering.
        """
        dsigma = sigma_grid[1] - sigma_grid[0]
        N = len(sigma_grid)
        
        # First derivative matrix
        D1 = np.zeros((N, N), dtype=complex)
        for i in range(N):
            D1[i, (i+1) % N] = 1.0
            D1[i, (i-1) % N] = -1.0
        D1 = D1 / (2 * dsigma)
        
        coupling_factor = 0.0
        
        # Build index mapping: state_keys index -> spatial_coupling index
        # The spatial_coupling matrix is ordered by the basis's spatial_states
        # We assume state_keys matches this ordering (they come from the same solver)
        
        for i, key_i in enumerate(state_keys):
            for j, key_j in enumerate(state_keys):
                if i == j:
                    continue
                
                # Check if we have non-trivial chi for both states
                chi_i = chi_components.get(key_i)
                chi_j = chi_components.get(key_j)
                
                if chi_i is None or chi_j is None:
                    continue
                
                # Check if chi has significant amplitude
                norm_i = np.sum(np.abs(chi_i)**2) * dsigma
                norm_j = np.sum(np.abs(chi_j)**2) * dsigma
                
                if norm_i < 1e-10 or norm_j < 1e-10:
                    continue
                
                R_ij = spatial_coupling[i, j]
                if abs(R_ij) < 1e-10:
                    continue
                
                dchi_j = D1 @ chi_j
                
                # IMAGINARY part carries the winding!
                # Use R_ij WITH ITS SIGN (not abs) to get correct E_coupling sign
                integral = np.sum(np.conj(chi_i) * dchi_j) * dsigma
                coupling_factor += np.real(R_ij) * np.imag(integral)
        
        return float(np.real(coupling_factor))
    
    def _compute_coupling_energy_from_structure_v2(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        spatial_coupling: NDArray,
        sigma_grid: NDArray,
        state_index_map: Dict[Tuple[int, int, int], int],
    ) -> float:
        """
        Compute coupling factor using correct index mapping.
        
        E_coupling_factor = Σ_{i,j} |R_{ij}| × Im[∫χ_i*(∂χ_j/∂σ)dσ]
        
        Uses state_index_map to correctly map (n,l,m) keys to spatial_coupling indices.
        """
        dsigma = sigma_grid[1] - sigma_grid[0]
        N = len(sigma_grid)
        
        # First derivative matrix
        D1 = np.zeros((N, N), dtype=complex)
        for i in range(N):
            D1[i, (i+1) % N] = 1.0
            D1[i, (i-1) % N] = -1.0
        D1 = D1 / (2 * dsigma)
        
        coupling_factor = 0.0
        
        # Iterate over all pairs of states in chi_components
        keys_with_amplitude = []
        for key, chi in chi_components.items():
            norm = np.sum(np.abs(chi)**2) * dsigma
            if norm > 1e-10:
                keys_with_amplitude.append(key)
        
        for key_i in keys_with_amplitude:
            idx_i = state_index_map.get(key_i)
            if idx_i is None:
                continue
                
            for key_j in keys_with_amplitude:
                if key_i == key_j:
                    continue
                    
                idx_j = state_index_map.get(key_j)
                if idx_j is None:
                    continue
                
                # Get coupling matrix element using CORRECT indices
                # Use R_ij WITH SIGN (not |R_ij|) to get correct coupling sign
                R_ij = spatial_coupling[idx_i, idx_j]
                if abs(R_ij) < 1e-10:
                    continue
                
                chi_i = chi_components[key_i]
                chi_j = chi_components[key_j]
                
                dchi_j = D1 @ chi_j
                
                # IMAGINARY part carries the winding
                # Use R_ij WITH ITS SIGN (not abs) to get correct E_coupling sign
                integral = np.sum(np.conj(chi_i) * dchi_j) * dsigma
                coupling_factor += np.real(R_ij) * np.imag(integral)
        
        return float(np.real(coupling_factor))
    
    def compute_energy(
        self,
        A: float,
        delta_x: float,
        delta_sigma: float,
        shape_integrals: Dict[str, float],
        n: int,
        subspace_factor: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total energy for given (A, Δx, Δσ) and precomputed shape integrals.
        
        CRITICAL: The spatial_factor is computed DYNAMICALLY from (n, delta_x)!
        This is the key to having the spatial scale emerge from optimization.
        
        Args:
            A: Amplitude (optimization variable)
            delta_x: Spatial extent (optimization variable)
            delta_sigma: Subspace width scale (optimization variable)
            shape_integrals: Dict from _compute_shape_integrals
            n: Spatial quantum number (FIXED input - determines node structure)
            subspace_factor: From _compute_subspace_factor (computed once from chi)
            
        Returns:
            E_total, breakdown dict
        """
        A_sq = A ** 2
        
        # Unpack shape integrals
        I_kin = shape_integrals['I_kinetic']
        I_pot = shape_integrals['I_potential']
        I_NL = shape_integrals['I_nonlinear']
        J = shape_integrals['J']
        
        # === E_KINETIC: (ℏ²/2m) × (A²/Δσ²) × I_kin ===
        if delta_sigma > 1e-10:
            E_kinetic = (self.hbar**2 / (2 * self.m_eff)) * (A_sq / delta_sigma**2) * I_kin
        else:
            E_kinetic = 1e10
        
        # === E_POTENTIAL: V₀ × A² × I_pot ===
        E_potential = self.V0 * A_sq * I_pot
        
        # === E_NONLINEAR: (g₁/2) × (A⁴/Δσ) × I_NL ===
        if delta_sigma > 1e-10:
            E_nonlinear = (self.g1 / 2) * (A_sq**2 / delta_sigma) * I_NL
        else:
            E_nonlinear = 1e10
        
        # === E_CIRCULATION: g₂ × |J|² × A⁴ ===
        E_circulation = self.g2 * np.abs(J)**2 * A_sq**2
        
        # === E_SPATIAL: ℏ²/(2βA²Δx²) ===
        if A_sq > 1e-10 and delta_x > 1e-10:
            E_spatial = self.hbar**2 / (2 * self.beta * A_sq * delta_x**2)
        else:
            E_spatial = 1e10
        
        # === E_COUPLING: -α × spatial_factor(n, Δx) × subspace_factor × A ===
        # 
        # CRITICAL: spatial_factor is computed DYNAMICALLY from (n, delta_x)!
        # This allows the spatial scale to emerge from optimization.
        #
        # - spatial_factor captures how spatial wavefunction structure affects coupling
        # - subspace_factor captures how subspace wavefunction structure affects coupling
        # - Together they give the total coupling strength
        #
        spatial_factor = self._compute_spatial_factor(n, delta_x)
        coupling_factor = spatial_factor * subspace_factor
        E_coupling = -self.alpha * coupling_factor * A
        
        # === E_CURVATURE: κ × (βA²)² / Δx ===
        if delta_x > 1e-10:
            E_curvature = self.kappa * (self.beta * A_sq)**2 / delta_x
        else:
            E_curvature = 1e10
        
        E_total = E_kinetic + E_potential + E_nonlinear + E_circulation + E_spatial + E_coupling + E_curvature
        
        breakdown = {
            'E_total': E_total,
            'E_kinetic': E_kinetic,
            'E_potential': E_potential,
            'E_nonlinear': E_nonlinear,
            'E_circulation': E_circulation,
            'E_spatial': E_spatial,
            'E_coupling': E_coupling,
            'E_curvature': E_curvature,
            'spatial_factor': spatial_factor,
            'subspace_factor': subspace_factor,
            'coupling_factor': coupling_factor,
        }
        
        return E_total, breakdown
    
    def compute_gradients(
        self,
        A: float,
        delta_x: float,
        delta_sigma: float,
        shape_integrals: Dict[str, float],
        n: int,
        subspace_factor: float,
    ) -> Tuple[float, float, float]:
        """
        Compute gradients ∂E/∂A, ∂E/∂Δx, ∂E/∂Δσ.
        
        CRITICAL: The spatial_factor depends on Δx, so ∂E_coupling/∂Δx is non-zero!
        
        Returns:
            grad_A, grad_dx, grad_ds
        """
        A_sq = A ** 2
        
        I_kin = shape_integrals['I_kinetic']
        I_pot = shape_integrals['I_potential']
        I_NL = shape_integrals['I_nonlinear']
        J = shape_integrals['J']
        
        # Compute spatial_factor and its derivative w.r.t. delta_x (numerical)
        spatial_factor = self._compute_spatial_factor(n, delta_x)
        eps = delta_x * 0.001 if delta_x > 1e-6 else 1e-6
        spatial_factor_plus = self._compute_spatial_factor(n, delta_x + eps)
        d_spatial_d_dx = (spatial_factor_plus - spatial_factor) / eps
        
        coupling_factor = spatial_factor * subspace_factor
        
        # === ∂E/∂A ===
        grad_A = 0.0
        
        # ∂E_kinetic/∂A
        if delta_sigma > 1e-10:
            grad_A += 2 * A * (self.hbar**2 / (2 * self.m_eff)) * I_kin / delta_sigma**2
        
        # ∂E_potential/∂A
        grad_A += 2 * A * self.V0 * I_pot
        
        # ∂E_nonlinear/∂A
        if delta_sigma > 1e-10:
            grad_A += 4 * A**3 * (self.g1 / 2) * I_NL / delta_sigma
        
        # ∂E_circulation/∂A
        grad_A += 4 * A**3 * self.g2 * np.abs(J)**2
        
        # ∂E_spatial/∂A
        if A > 1e-10 and delta_x > 1e-10:
            grad_A += -self.hbar**2 / (self.beta * A**3 * delta_x**2)
        
        # ∂E_coupling/∂A = -α × coupling_factor
        grad_A += -self.alpha * coupling_factor
        
        # ∂E_curvature/∂A
        if delta_x > 1e-10:
            grad_A += 4 * self.kappa * self.beta**2 * A**3 / delta_x
        
        # === ∂E/∂Δx ===
        grad_dx = 0.0
        
        # ∂E_spatial/∂Δx
        if A_sq > 1e-10 and delta_x > 1e-10:
            grad_dx += -self.hbar**2 / (self.beta * A_sq * delta_x**3)
        
        # ∂E_coupling/∂Δx = -α × (∂spatial_factor/∂Δx) × subspace_factor × A
        grad_dx += -self.alpha * d_spatial_d_dx * subspace_factor * A
        
        # ∂E_curvature/∂Δx
        if delta_x > 1e-10:
            grad_dx += -self.kappa * (self.beta * A_sq)**2 / delta_x**2
        
        # === ∂E/∂Δσ ===
        grad_ds = 0.0
        
        # ∂E_kinetic/∂Δσ
        if delta_sigma > 1e-10:
            E_kin = (self.hbar**2 / (2 * self.m_eff)) * (A_sq / delta_sigma**2) * I_kin
            grad_ds += -2 * E_kin / delta_sigma
        
        # ∂E_nonlinear/∂Δσ
        if delta_sigma > 1e-10:
            E_NL = (self.g1 / 2) * (A_sq**2 / delta_sigma) * I_NL
            grad_ds += -E_NL / delta_sigma
        
        return float(grad_A), float(grad_dx), float(grad_ds)
    
    def minimize(
        self,
        structure: WavefunctionStructure,
        sigma_grid: NDArray,
        V_sigma: NDArray,
        spatial_coupling: NDArray,
        state_index_map: Dict[Tuple[int, int, int], int],
        initial_A: float = 0.1,
        initial_delta_x: float = 1.0,
        initial_delta_sigma: float = 0.5,
        max_iter: int = 10000,
        tol: float = 1e-10,
        verbose: bool = False,
    ) -> EnergyMinimizationResult:
        """
        Minimize E_total over (A, Δx, Δσ) for the given wavefunction structure.
        
        This is the UNIVERSAL minimizer - same code for all particle types!
        
        Args:
            structure: WavefunctionStructure from NonSeparableWavefunctionSolver
            sigma_grid: Subspace coordinate grid
            V_sigma: Three-well potential on grid
            spatial_coupling: Coupling matrix between states
            state_index_map: Dict mapping (n,l,m) -> index in spatial_coupling
            initial_A, initial_delta_x, initial_delta_sigma: Starting values
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            EnergyMinimizationResult with optimal A, Δx, Δσ and predicted mass
        """
        # Get the quantum number n from the structure
        n = structure.n_target
        
        if verbose:
            print(f"Minimizing energy for {structure.particle_type} "
                  f"(n={n}, k={structure.k_winding})")
        
        # Precompute shape integrals (computed ONCE)
        shape_integrals = self._compute_shape_integrals(
            structure.chi_components, sigma_grid, V_sigma
        )
        
        # Precompute SUBSPACE factor (computed ONCE from chi structure)
        # The SPATIAL factor is computed dynamically as delta_x changes!
        subspace_factor = self._compute_subspace_factor(
            structure.chi_components, sigma_grid
        )
        
        if verbose:
            print(f"  Shape integrals: I_kin={shape_integrals['I_kinetic']:.4f}, "
                  f"I_pot={shape_integrals['I_potential']:.4f}, "
                  f"I_NL={shape_integrals['I_nonlinear']:.4f}")
            print(f"  Subspace factor: {subspace_factor:.6f}")
            print(f"  n (spatial mode): {n}")
        
        # Initialize
        A = initial_A
        delta_x = initial_delta_x
        delta_sigma = initial_delta_sigma
        
        # Learning rates
        lr_A = 0.001
        lr_dx = 0.01
        lr_ds = 0.001
        
        # Compute initial energy
        # Note: spatial_factor is computed DYNAMICALLY inside compute_energy!
        energy, breakdown = self.compute_energy(
            A, delta_x, delta_sigma, shape_integrals, n, subspace_factor
        )
        
        best_energy = energy
        best_params = (A, delta_x, delta_sigma)
        best_breakdown = breakdown
        
        for iteration in range(max_iter):
            # Compute gradients
            grad_A, grad_dx, grad_ds = self.compute_gradients(
                A, delta_x, delta_sigma, shape_integrals, n, subspace_factor
            )
            
            # Gradient descent
            A_new = A - lr_A * grad_A
            delta_x_new = delta_x - lr_dx * grad_dx
            delta_sigma_new = delta_sigma - lr_ds * grad_ds
            
            # Keep positive
            A_new = max(A_new, 1e-6)
            delta_x_new = max(delta_x_new, 1e-6)
            delta_sigma_new = max(delta_sigma_new, 1e-6)
            
            # Compute new energy (spatial_factor computed dynamically inside!)
            new_energy, new_breakdown = self.compute_energy(
                A_new, delta_x_new, delta_sigma_new, shape_integrals, n, subspace_factor
            )
            
            # Accept if improved
            if new_energy < energy:
                energy_change = abs(new_energy - energy)
                A, delta_x, delta_sigma = A_new, delta_x_new, delta_sigma_new
                energy = new_energy
                breakdown = new_breakdown
                
                if energy < best_energy:
                    best_energy = energy
                    best_params = (A, delta_x, delta_sigma)
                    best_breakdown = new_breakdown
                
                # Increase learning rates
                lr_A = min(lr_A * 1.05, 0.1)
                lr_dx = min(lr_dx * 1.05, 1.0)
                lr_ds = min(lr_ds * 1.05, 0.1)
            else:
                # Reduce learning rates
                lr_A *= 0.5
                lr_dx *= 0.5
                lr_ds *= 0.5
                energy_change = 0
            
            # Convergence check
            if lr_A < 1e-12 and lr_dx < 1e-12 and lr_ds < 1e-12:
                if verbose:
                    print(f"  Converged (lr collapsed) at iteration {iteration}")
                break
            
            if energy_change > 0 and energy_change < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
            
            # Logging
            if verbose and iteration % 1000 == 0:
                mass = self.beta * A**2
                compton = delta_x * self.beta * A**2
                print(f"  Iter {iteration}: E={energy:.6f}, A={A:.4f}, "
                      f"dx={delta_x:.4f}, ds={delta_sigma:.4f}, "
                      f"mass={mass:.6f} GeV, Compton={compton:.4f}")
        
        # Use best result
        A, delta_x, delta_sigma = best_params
        breakdown = best_breakdown
        
        # Final results
        mass = self.beta * A**2
        compton_check = delta_x * self.beta * A**2
        
        if verbose:
            spatial_factor = self._compute_spatial_factor(n, delta_x)
            print(f"  Final: A={A:.4f}, dx={delta_x:.4f}, ds={delta_sigma:.4f}")
            print(f"  Mass: {mass:.6f} GeV")
            print(f"  Spatial factor (final): {spatial_factor:.6f}")
            print(f"  Compton check: dx*beta*A^2 = {compton_check:.4f}")
        
        return EnergyMinimizationResult(
            A=A,
            A_squared=A**2,
            delta_x=delta_x,
            delta_sigma=delta_sigma,
            mass_gev=mass,
            E_total=breakdown['E_total'],
            E_kinetic=breakdown['E_kinetic'],
            E_potential=breakdown['E_potential'],
            E_nonlinear=breakdown['E_nonlinear'],
            E_circulation=breakdown['E_circulation'],
            E_spatial=breakdown['E_spatial'],
            E_coupling=breakdown['E_coupling'],
            E_curvature=breakdown['E_curvature'],
            n_target=structure.n_target,
            k_winding=structure.k_winding,
            particle_type=structure.particle_type,
            converged=iteration < max_iter - 1,
            iterations=iteration + 1,
            compton_check=compton_check,
        )


def test_universal_minimizer():
    """Test the universal energy minimizer with wavefunction solver."""
    print("=" * 60)
    print("TESTING UNIVERSAL ENERGY MINIMIZER")
    print("=" * 60)
    
    from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
    
    # Parameters
    alpha = 20.0
    beta = 100.0
    kappa = 0.0001
    g1 = 5000.0
    g2 = 0.004
    V0 = 1.0
    
    # Create wavefunction solver
    wf_solver = NonSeparableWavefunctionSolver(
        alpha=alpha, beta=beta, kappa=kappa,
        g1=g1, g2=g2, V0=V0,
        n_max=5, l_max=2, N_sigma=48,
    )
    
    # Create energy minimizer (SAME parameters)
    minimizer = UniversalEnergyMinimizer(
        alpha=alpha, beta=beta, kappa=kappa,
        g1=g1, g2=g2, V0=V0,
    )
    
    print(f"\nParameters: alpha={alpha}, beta={beta}, kappa={kappa}, g1={g1}")
    
    # Test on leptons
    print("\n--- LEPTON MASSES ---")
    exp_masses = {'Electron': 0.000511, 'Muon': 0.1057, 'Tau': 1.777}
    
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        print(f"\n{name} (n={n}):")
        
        # Stage 1: Get wavefunction structure
        structure = wf_solver.solve_lepton(n_target=n, k_winding=1, verbose=False)
        
        # Stage 2: Minimize energy over (A, Δx, Δσ)
        result = minimizer.minimize(
            structure=structure,
            sigma_grid=wf_solver.get_sigma_grid(),
            V_sigma=wf_solver.get_V_sigma(),
            spatial_coupling=wf_solver.get_spatial_coupling_matrix(),
            state_index_map=wf_solver.get_state_index_map(),
            verbose=True,
        )
        
        exp = exp_masses[name]
        ratio = result.mass_gev / exp if exp > 0 else float('inf')
        print(f"  Predicted: {result.mass_gev:.6e} GeV, Experimental: {exp:.6e} GeV, Ratio: {ratio:.2f}")
    
    print("\n" + "=" * 60)
    print("UNIVERSAL MINIMIZER TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_universal_minimizer()

