"""
Linear eigenvalue solver for the SFT Schrödinger equation.

Solves the linear eigenvalue problem:
    H χ = E χ
where H = T + V is the Hamiltonian with kinetic and potential terms.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Tuple, Optional, List

from sft_solver.core.grid import SpectralGrid
from sft_solver.potentials.three_well import ThreeWellPotential
from sft_solver.eigensolver.spectral import SpectralOperators


class LinearEigensolver:
    """
    Linear eigenvalue solver for the single-particle problem.
    
    Solves for the lowest eigenvalues and eigenfunctions of:
        (-ℏ²/2m d²/dσ² + V(σ)) χ(σ) = E χ(σ)
    
    with periodic boundary conditions on [0, 2π).
    
    Attributes:
        grid: SpectralGrid instance.
        potential: Potential function.
        operators: SpectralOperators for FFT-based calculations.
    """
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        m_eff: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize the linear eigensolver.
        
        Args:
            grid: SpectralGrid defining the discretization.
            potential: Potential function (e.g., ThreeWellPotential).
            m_eff: Effective mass parameter.
            hbar: Reduced Planck constant (1.0 in natural units).
        """
        self.grid = grid
        self.potential = potential
        self.m_eff = m_eff
        self.hbar = hbar
        
        self.operators = SpectralOperators(grid, m_eff, hbar)
        
        # Precompute potential on grid
        self._V_grid = potential(grid.sigma)
    
    def solve(
        self,
        n_states: int = 5,
        k: Optional[int] = None,
        tol: float = 1e-10
    ) -> Tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
        """
        Solve for the lowest n_states eigenvalues and eigenfunctions.
        
        Args:
            n_states: Number of states to compute.
            k: If specified, project onto winding number sector.
            tol: Convergence tolerance for eigensolver.
            
        Returns:
            Tuple of (energies, wavefunctions) where:
                - energies: Array of n_states eigenvalues in ascending order
                - wavefunctions: Array of shape (n_states, N) with eigenfunctions
        """
        # Build Hamiltonian
        H = self.operators.build_hamiltonian_matrix(self._V_grid)
        
        # Ensure Hermitian (numerical precision)
        H = 0.5 * (H + np.conj(H.T))
        
        # Use scipy's eigensolver
        if n_states >= self.grid.N - 1:
            # Full diagonalization for small systems
            energies, wavefunctions = np.linalg.eigh(H)
            idx = np.argsort(energies)[:n_states]
            energies = energies[idx]
            wavefunctions = wavefunctions[:, idx].T
        else:
            # Sparse eigensolver for larger systems
            H_sparse = sparse.csr_matrix(H)
            energies, wavefunctions = eigsh(
                H_sparse, 
                k=n_states, 
                which='SA',  # Smallest algebraic
                tol=tol
            )
            # Sort by energy
            idx = np.argsort(energies)
            energies = energies[idx]
            wavefunctions = wavefunctions[:, idx].T
        
        # Normalize wavefunctions
        for i in range(len(energies)):
            wavefunctions[i] = self.grid.normalize(wavefunctions[i])
        
        # If k specified, filter by winding number
        if k is not None:
            energies, wavefunctions = self._filter_by_winding(
                energies, wavefunctions, k, n_states
            )
        
        return np.real(energies), wavefunctions
    
    def solve_with_winding(
        self,
        k: int,
        n_states: int = 3,
        tol: float = 1e-10
    ) -> Tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
        """
        Solve for states with a specific winding number.
        
        This uses a modified approach that enforces the winding
        structure χ(σ) = e^(ikσ) u(σ) where u is periodic.
        
        Args:
            k: Winding number (integer).
            n_states: Number of states to compute.
            tol: Convergence tolerance.
            
        Returns:
            Tuple of (energies, wavefunctions).
        """
        # Build modified Hamiltonian for winding k sector
        # Transform: χ(σ) = e^(ikσ) u(σ)
        # The equation becomes:
        # (-ℏ²/2m (d/dσ + ik)² + V) u = E u
        # = (-ℏ²/2m (d²u/dσ² + 2ik du/dσ - k² u) + V u) = E u
        
        H_k = self._build_winding_hamiltonian(k)
        
        # Ensure Hermitian
        H_k = 0.5 * (H_k + np.conj(H_k.T))
        
        # Solve eigenvalue problem
        if n_states >= self.grid.N - 1:
            energies, u_funcs = np.linalg.eigh(H_k)
            idx = np.argsort(energies)[:n_states]
            energies = energies[idx]
            u_funcs = u_funcs[:, idx].T
        else:
            H_k_sparse = sparse.csr_matrix(H_k)
            energies, u_funcs = eigsh(
                H_k_sparse,
                k=n_states,
                which='SA',
                tol=tol
            )
            idx = np.argsort(energies)
            energies = energies[idx]
            u_funcs = u_funcs[:, idx].T
        
        # Convert back to χ = e^(ikσ) u
        phase = np.exp(1j * k * self.grid.sigma)
        wavefunctions = np.zeros_like(u_funcs, dtype=complex)
        for i in range(len(energies)):
            chi = phase * u_funcs[i]
            wavefunctions[i] = self.grid.normalize(chi)
        
        return np.real(energies), wavefunctions
    
    def _build_winding_hamiltonian(self, k: int) -> NDArray[np.complexfloating]:
        """
        Build Hamiltonian in the winding-k sector.
        
        For χ = e^(ikσ) u, the transformed Hamiltonian is:
        H_k = -ℏ²/(2m) [d²/dσ² + 2ik d/dσ - k²] + V
        """
        N = self.grid.N
        
        # Second derivative matrix (from FFT)
        d2 = np.zeros((N, N), dtype=complex)
        for j in range(N):
            e_j = np.zeros(N)
            e_j[j] = 1.0
            d2[:, j] = self.grid.second_derivative(e_j)
        
        # First derivative matrix (for 2ik d/dσ term)
        d1 = np.zeros((N, N), dtype=complex)
        for j in range(N):
            e_j = np.zeros(N)
            e_j[j] = 1.0
            d1[:, j] = self.grid.first_derivative(e_j)
        
        # Build H_k
        coeff = self.hbar**2 / (2 * self.m_eff)
        H_k = -coeff * d2 - coeff * 2j * k * d1 + coeff * k**2 * np.eye(N)
        H_k += np.diag(self._V_grid)
        
        return H_k
    
    def _filter_by_winding(
        self,
        energies: NDArray,
        wavefunctions: NDArray,
        k: int,
        n_wanted: int
    ) -> Tuple[NDArray, NDArray]:
        """
        Filter eigenstates by winding number.
        
        Keeps states whose winding number is close to k.
        """
        filtered_E = []
        filtered_psi = []
        
        for i, psi in enumerate(wavefunctions):
            winding = self.grid.winding_number(psi)
            if abs(winding - k) < 0.5:
                filtered_E.append(energies[i])
                filtered_psi.append(psi)
            if len(filtered_E) >= n_wanted:
                break
        
        return np.array(filtered_E), np.array(filtered_psi)
    
    def ground_state(
        self, 
        k: int = 1
    ) -> Tuple[float, NDArray[np.complexfloating]]:
        """
        Find the ground state with given winding number.
        
        Args:
            k: Winding number.
            
        Returns:
            Tuple of (energy, wavefunction).
        """
        energies, wavefunctions = self.solve_with_winding(k, n_states=1)
        return float(energies[0]), wavefunctions[0]
    
    def compute_amplitude_squared(
        self, 
        wavefunction: NDArray[np.complexfloating]
    ) -> float:
        """
        Compute the integrated amplitude squared A²_χ.
        
        A²_χ = ∫₀^(2π) |χ(σ)|² dσ
        
        This determines the mass via m = β A².
        
        Args:
            wavefunction: Wavefunction on the grid.
            
        Returns:
            Integrated amplitude squared.
        """
        return np.real(self.grid.integrate(np.abs(wavefunction)**2))
    
    def compute_mass(
        self, 
        wavefunction: NDArray[np.complexfloating],
        beta: float
    ) -> float:
        """
        Compute the particle mass from the wavefunction.
        
        m = β A²_χ
        
        Args:
            wavefunction: Wavefunction on the grid.
            beta: Mass coupling constant (GeV).
            
        Returns:
            Mass in GeV.
        """
        A_sq = self.compute_amplitude_squared(wavefunction)
        return beta * A_sq
    
    def verify_eigenstate(
        self,
        energy: float,
        wavefunction: NDArray[np.complexfloating],
        tol: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Verify that a state is an eigenstate of H.
        
        Checks that H|ψ⟩ = E|ψ⟩ by computing the residual.
        
        Args:
            energy: Claimed eigenvalue.
            wavefunction: Claimed eigenfunction.
            tol: Tolerance for residual.
            
        Returns:
            Tuple of (is_eigenstate, residual_norm).
        """
        H_psi = self.operators.apply_hamiltonian(wavefunction, self._V_grid)
        residual = H_psi - energy * wavefunction
        residual_norm = self.grid.norm(residual)
        
        return residual_norm < tol, float(residual_norm)
    
    def energy_components(
        self, 
        wavefunction: NDArray[np.complexfloating]
    ) -> dict:
        """
        Break down the energy into kinetic and potential parts.
        
        Args:
            wavefunction: Normalized wavefunction.
            
        Returns:
            Dictionary with 'kinetic', 'potential', and 'total' energies.
        """
        T = self.operators.kinetic_expectation(wavefunction)
        V = self.operators.potential_expectation(wavefunction, self._V_grid)
        
        return {
            'kinetic': T,
            'potential': V,
            'total': T + V
        }

