"""
Meson Solver for SFM.

Solves the quark-antiquark bound state problem for mesons (pion, kaon, etc.).

Key physics:
- Mesons consist of quark (k=+3) + antiquark (k=-3)
- Net winding = 0 (neutral in the topological sense)
- Geometrically protected against annihilation
- Metastable due to opposite circulations

Mathematical formulation:
    [-ℏ²/(2m_σR²) ∂²/∂σ² + V(σ)]χ_q + g₁|χ_q + χ_q̄|²χ_q = μ_q χ_q
    [-ℏ²/(2m_σR²) ∂²/∂σ² + V(σ)]χ_q̄ + g₁|χ_q + χ_q̄|²χ_q̄ = μ_q̄ χ_q̄

The quark-antiquark system has reduced nonlinear energy compared to
quark-quark because the circulations partially cancel.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.eigensolver.linear import LinearEigensolver


@dataclass
class MesonState:
    """Result of meson solver."""
    # Wavefunctions
    chi_quark: NDArray[np.complexfloating]
    chi_antiquark: NDArray[np.complexfloating]
    
    # Energies
    energy_total: float
    energy_binding: float
    energy_quark: float
    energy_antiquark: float
    
    # Winding structure
    winding_quark: float
    winding_antiquark: float
    net_winding: float
    
    # Amplitudes
    amplitude_quark: float
    amplitude_antiquark: float
    amplitude_squared_total: float
    
    # Stability
    is_metastable: bool
    circulation_cancellation: float  # How well circulations cancel
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float
    
    # Metadata
    quark_content: str = "ud̄"  # Default pion


class MesonSolver:
    """
    Solver for quark-antiquark bound states (mesons).
    
    Key features:
    - Quark (k=+3) + antiquark (k=-3) system
    - Net winding = 0
    - Geometric protection against annihilation
    - Metastable configuration
    """
    
    # Well positions for quark and antiquark
    # In a meson, they can be in same well (more tightly bound)
    # or opposite wells (less tightly bound)
    
    # Meson configurations
    MESON_CONFIGS = {
        'pion_plus': {'content': 'ud̄', 'quark': 'u', 'antiquark': 'd'},
        'pion_zero': {'content': 'uū/dd̄', 'quark': 'u', 'antiquark': 'u'},
        'pion_minus': {'content': 'dū', 'quark': 'd', 'antiquark': 'u'},
        'kaon_plus': {'content': 'us̄', 'quark': 'u', 'antiquark': 's'},
        'kaon_zero': {'content': 'ds̄', 'quark': 'd', 'antiquark': 's'},
        'eta': {'content': 'uū+dd̄', 'quark': 'u', 'antiquark': 'u'},
    }
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g1: float = 0.1,
        k_quark: int = 3,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        R: float = 1.0
    ):
        """
        Initialize meson solver.
        
        Args:
            grid: SpectralGrid instance
            potential: Three-well potential
            g1: Nonlinear coupling constant
            k_quark: Winding number for quark (default 3)
            m_eff: Effective mass
            hbar: Reduced Planck constant
            R: Subspace radius
        """
        self.grid = grid
        self.potential = potential
        self.g1 = g1
        self.k_quark = k_quark
        self.k_antiquark = -k_quark  # Opposite winding
        self.m_eff = m_eff
        self.hbar = hbar
        self.R = R
        
        # Create operators
        self.operators = SpectralOperators(grid, m_eff, hbar)
        self._V_grid = potential(grid.sigma)
        
        # Linear solver for reference
        self.linear_solver = LinearEigensolver(grid, potential, m_eff, hbar)
    
    def _initialize_quark(
        self,
        well_idx: int,
        k: int
    ) -> NDArray[np.complexfloating]:
        """Initialize a quark/antiquark in a well with given winding."""
        well_positions = [0.0, 2*np.pi/3, 4*np.pi/3]
        center = well_positions[well_idx]
        
        envelope = self.grid.create_gaussian_envelope(
            center=center,
            width=0.5,
            periodic=True
        )
        
        winding = np.exp(1j * k * self.grid.sigma)
        chi = envelope * winding
        
        return self.grid.normalize(chi)
    
    def _compute_effective_potential(
        self,
        chi_q: NDArray,
        chi_qbar: NDArray
    ) -> NDArray:
        """Compute V_eff = V + g₁|χ_q + χ_q̄|²."""
        chi_total = chi_q + chi_qbar
        density = np.abs(chi_total)**2
        return self._V_grid + self.g1 * density
    
    def _solve_single_step(
        self,
        V_eff: NDArray,
        k: int,
        target_well: int
    ) -> Tuple[float, NDArray]:
        """Solve for one particle in effective potential."""
        H = self.operators.build_hamiltonian_matrix(V_eff)
        H = 0.5 * (H + np.conj(H.T))
        
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Find state localized in target well
        well_positions = [0.0, 2*np.pi/3, 4*np.pi/3]
        well_center = well_positions[target_well]
        
        well_weight = self.grid.create_gaussian_envelope(
            center=well_center,
            width=0.5,
            periodic=True
        )
        
        best_idx = 0
        best_overlap = -1
        
        for i in range(min(10, len(eigenvalues))):
            psi = eigenvectors[:, i]
            density = np.abs(psi)**2
            overlap = np.real(self.grid.integrate(density * well_weight))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        
        E = eigenvalues[best_idx]
        chi = eigenvectors[:, best_idx]
        
        # Apply correct winding
        envelope = np.abs(chi)
        chi = envelope * np.exp(1j * k * self.grid.sigma)
        
        return E, self.grid.normalize(chi)
    
    def _compute_total_energy(
        self,
        chi_q: NDArray,
        chi_qbar: NDArray
    ) -> float:
        """Compute total meson energy."""
        total = 0.0
        
        for chi in [chi_q, chi_qbar]:
            T = self.operators.kinetic_expectation(chi)
            V = self.operators.potential_expectation(chi, self._V_grid)
            total += T + V
        
        # Nonlinear interaction
        chi_total = chi_q + chi_qbar
        density = np.abs(chi_total)**2
        E_nl = (self.g1 / 2) * np.real(self.grid.integrate(density**2))
        
        return total + E_nl
    
    def _compute_circulation_cancellation(
        self,
        chi_q: NDArray,
        chi_qbar: NDArray
    ) -> float:
        """
        Compute how well circulations cancel.
        
        For perfect quark-antiquark pair, J_q + J_qbar ≈ 0.
        Returns a normalized measure: 0 = perfect cancellation, 1 = no cancellation.
        """
        J_q = self.grid.circulation(chi_q)
        J_qbar = self.grid.circulation(chi_qbar)
        
        total = np.abs(J_q + J_qbar)
        individual = np.abs(J_q) + np.abs(J_qbar)
        
        if individual < 1e-10:
            return 0.0
        
        return total / individual
    
    def solve(
        self,
        meson_config: str = 'pion_plus',
        quark_well: int = 0,
        antiquark_well: int = 0,
        tol: float = 1e-8,
        max_iter: int = 300,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> MesonState:
        """
        Solve the quark-antiquark system.
        
        Args:
            meson_config: Type of meson
            quark_well: Well index for quark (0, 1, or 2)
            antiquark_well: Well index for antiquark
            tol: Convergence tolerance
            max_iter: Maximum iterations
            mixing: Linear mixing parameter
            verbose: Print progress
            
        Returns:
            MesonState with complete solution
        """
        if verbose:
            print("=" * 60)
            print(f"MESON SOLVER: {meson_config.upper()}")
            print(f"  Quark k = +{self.k_quark}, Antiquark k = {self.k_antiquark}")
            print(f"  g₁ = {self.g1}")
            print("=" * 60)
        
        # Initialize quark and antiquark
        chi_q = self._initialize_quark(quark_well, self.k_quark)
        chi_qbar = self._initialize_quark(antiquark_well, self.k_antiquark)
        
        E_old = self._compute_total_energy(chi_q, chi_qbar)
        
        converged = False
        residual = float('inf')
        
        for iteration in range(max_iter):
            chi_q_old = chi_q.copy()
            chi_qbar_old = chi_qbar.copy()
            
            # Effective potential
            V_eff = self._compute_effective_potential(chi_q, chi_qbar)
            
            # Solve for each
            E_q, chi_q_new = self._solve_single_step(V_eff, self.k_quark, quark_well)
            E_qbar, chi_qbar_new = self._solve_single_step(
                V_eff, self.k_antiquark, antiquark_well
            )
            
            # Mixing
            chi_q = (1 - mixing) * chi_q_old + mixing * chi_q_new
            chi_qbar = (1 - mixing) * chi_qbar_old + mixing * chi_qbar_new
            
            chi_q = self.grid.normalize(chi_q)
            chi_qbar = self.grid.normalize(chi_qbar)
            
            # Convergence check
            E_new = self._compute_total_energy(chi_q, chi_qbar)
            dE = abs(E_new - E_old)
            
            residual = max(
                self.grid.norm(chi_q - chi_q_old),
                self.grid.norm(chi_qbar - chi_qbar_old)
            )
            
            if verbose and iteration % 20 == 0:
                print(f"  Iter {iteration}: E = {E_new:.8f}, dE = {dE:.2e}")
            
            if dE < tol and residual < tol * 10:
                converged = True
                if verbose:
                    print(f"\n  Converged at iteration {iteration}")
                break
            
            E_old = E_new
        
        # Extract winding numbers
        winding_q = self.grid.winding_number(chi_q)
        winding_qbar = self.grid.winding_number(chi_qbar)
        
        # Compute amplitudes
        A_q = np.sqrt(np.real(self.grid.integrate(np.abs(chi_q)**2)))
        A_qbar = np.sqrt(np.real(self.grid.integrate(np.abs(chi_qbar)**2)))
        
        # Circulation cancellation
        cancellation = self._compute_circulation_cancellation(chi_q, chi_qbar)
        
        # Binding energy (vs separated q + qbar)
        E_single_q = self.linear_solver.ground_state(k=self.k_quark)[0]
        E_single_qbar = self.linear_solver.ground_state(k=abs(self.k_antiquark))[0]
        E_binding = E_new - (E_single_q + E_single_qbar)
        
        # Metastability check: circulations should largely cancel
        is_metastable = cancellation < 0.5
        
        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS:")
            print(f"  Total energy: {E_new:.6f}")
            print(f"  Binding energy: {E_binding:.6f}")
            print(f"  Net winding: {winding_q + winding_qbar:.4f}")
            print(f"  Circulation cancellation: {cancellation:.4f}")
            print(f"  Metastable: {is_metastable}")
            print("=" * 60)
        
        config = self.MESON_CONFIGS.get(meson_config, {})
        
        return MesonState(
            chi_quark=chi_q,
            chi_antiquark=chi_qbar,
            energy_total=E_new,
            energy_binding=E_binding,
            energy_quark=E_q,
            energy_antiquark=E_qbar,
            winding_quark=winding_q,
            winding_antiquark=winding_qbar,
            net_winding=winding_q + winding_qbar,
            amplitude_quark=A_q,
            amplitude_antiquark=A_qbar,
            amplitude_squared_total=A_q**2 + A_qbar**2,
            is_metastable=is_metastable,
            circulation_cancellation=cancellation,
            converged=converged,
            iterations=iteration + 1,
            final_residual=residual,
            quark_content=config.get('content', 'qq̄')
        )
    
    def solve_pion(self, **kwargs) -> MesonState:
        """Convenience method to solve for pion (ud̄)."""
        return self.solve(meson_config='pion_plus', **kwargs)
    
    def calculate_mass(
        self,
        state: MesonState,
        beta: float
    ) -> float:
        """
        Calculate meson mass from amplitudes.
        
        m_meson = β × (A_q² + A_q̄²)
        """
        return beta * state.amplitude_squared_total


def solve_meson_system(
    V0: float = 1.0,
    V1: float = 0.1,
    g1: float = 0.1,
    grid_N: int = 256,
    **kwargs
) -> MesonState:
    """
    Convenience function to solve for a meson.
    
    Args:
        V0: Primary well depth
        V1: Secondary modulation
        g1: Nonlinear coupling
        grid_N: Number of grid points
        **kwargs: Passed to MesonSolver.solve()
        
    Returns:
        MesonState with solution
    """
    grid = SpectralGrid(N=grid_N)
    potential = ThreeWellPotential(V0=V0, V1=V1)
    solver = MesonSolver(grid, potential, g1=g1)
    
    return solver.solve(**kwargs)

