"""
Spectral methods for eigenvalue problems on S¹.

Provides FFT-based operators and utilities for constructing
the Hamiltonian in spectral representation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from typing import Tuple, Optional

from sfm_solver.core.grid import SpectralGrid


class SpectralOperators:
    """
    Spectral operators for the Hamiltonian on S¹.
    
    Provides efficient construction of kinetic energy and
    derivative operators using FFT methods.
    
    Attributes:
        grid: The SpectralGrid instance.
        kinetic_fourier: Kinetic energy in Fourier basis.
    """
    
    def __init__(self, grid: SpectralGrid, m_eff: float = 1.0, hbar: float = 1.0):
        """
        Initialize spectral operators.
        
        Args:
            grid: SpectralGrid instance defining the discretization.
            m_eff: Effective mass parameter (default 1.0 in natural units).
            hbar: Reduced Planck constant (default 1.0 in natural units).
        """
        self.grid = grid
        self.m_eff = m_eff
        self.hbar = hbar
        
        # Kinetic energy coefficients in Fourier space
        # T = -ℏ²/(2m) d²/dσ²
        # In Fourier space: T_k = ℏ²k²/(2m)
        self.kinetic_fourier = (hbar**2 / (2 * m_eff)) * grid.k**2
    
    def kinetic_energy_diagonal(self) -> NDArray[np.floating]:
        """
        Return the kinetic energy as diagonal in Fourier space.
        
        This is the eigenvalue of T for each Fourier mode.
        """
        return self.kinetic_fourier
    
    def apply_kinetic(
        self, 
        psi: NDArray[np.complexfloating]
    ) -> NDArray[np.complexfloating]:
        """
        Apply the kinetic energy operator to a wavefunction.
        
        T ψ = -ℏ²/(2m) d²ψ/dσ²
        
        Uses FFT for efficiency.
        
        Args:
            psi: Wavefunction on the grid.
            
        Returns:
            T ψ evaluated on the grid.
        """
        psi_hat = np.fft.fft(psi)
        T_psi_hat = self.kinetic_fourier * psi_hat
        return np.fft.ifft(T_psi_hat)
    
    def apply_hamiltonian(
        self,
        psi: NDArray[np.complexfloating],
        V: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """
        Apply the Hamiltonian H = T + V to a wavefunction.
        
        Args:
            psi: Wavefunction on the grid.
            V: Potential energy on the grid.
            
        Returns:
            H ψ evaluated on the grid.
        """
        T_psi = self.apply_kinetic(psi)
        return T_psi + V * psi
    
    def build_hamiltonian_matrix(
        self, 
        V: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """
        Build the full Hamiltonian matrix in real space.
        
        H = T + V where T is the kinetic energy and V is diagonal.
        
        The kinetic energy matrix is constructed from FFT:
        T_{ij} = (1/N) Σ_k exp(ik(σ_i - σ_j)) × ℏ²k²/(2m)
        
        Args:
            V: Potential energy on the grid.
            
        Returns:
            Full Hamiltonian matrix (N × N).
        """
        N = self.grid.N
        
        # Kinetic energy matrix using FFT
        # T = F^† Λ F where Λ is diagonal with kinetic eigenvalues
        F = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)  # Normalized FFT matrix
        F_dag = np.conj(F.T)
        Lambda = np.diag(self.kinetic_fourier)
        T = F_dag @ Lambda @ F
        
        # Potential energy is diagonal
        V_matrix = np.diag(V)
        
        return T + V_matrix
    
    def build_sparse_hamiltonian(
        self, 
        V: NDArray[np.floating]
    ) -> sparse.csr_matrix:
        """
        Build a sparse Hamiltonian matrix.
        
        Uses sparse representation for efficiency with large grids.
        The kinetic energy is applied via FFT, so we store it
        as a separate operator.
        
        For sparse eigensolvers, we return a LinearOperator instead.
        This method builds a full sparse matrix approximation.
        
        Args:
            V: Potential energy on the grid.
            
        Returns:
            Sparse Hamiltonian matrix.
        """
        H_dense = self.build_hamiltonian_matrix(V)
        return sparse.csr_matrix(H_dense)
    
    def winding_projection(self, k: int) -> NDArray[np.complexfloating]:
        """
        Create a projection operator for winding number k.
        
        This selects Fourier modes near k.
        
        Args:
            k: Target winding number.
            
        Returns:
            Diagonal projection in Fourier space.
        """
        # Allow modes within ±1 of target k
        k_grid = self.grid.k
        proj = np.zeros(self.grid.N)
        for dk in [-1, 0, 1]:
            mask = np.abs(k_grid - (k + dk)) < 0.5
            proj[mask] = 1.0
        return proj
    
    def expectation_value(
        self,
        psi: NDArray[np.complexfloating],
        V: NDArray[np.floating]
    ) -> float:
        """
        Calculate the energy expectation value <ψ|H|ψ>.
        
        Args:
            psi: Normalized wavefunction.
            V: Potential energy on the grid.
            
        Returns:
            Energy expectation value.
        """
        H_psi = self.apply_hamiltonian(psi, V)
        E = self.grid.inner_product(psi, H_psi)
        return np.real(E)
    
    def kinetic_expectation(
        self, 
        psi: NDArray[np.complexfloating]
    ) -> float:
        """
        Calculate the kinetic energy expectation value <ψ|T|ψ>.
        
        Args:
            psi: Wavefunction (should be normalized).
            
        Returns:
            Kinetic energy expectation value.
        """
        T_psi = self.apply_kinetic(psi)
        T_exp = self.grid.inner_product(psi, T_psi)
        return np.real(T_exp)
    
    def potential_expectation(
        self,
        psi: NDArray[np.complexfloating],
        V: NDArray[np.floating]
    ) -> float:
        """
        Calculate the potential energy expectation value <ψ|V|ψ>.
        
        Args:
            psi: Wavefunction (should be normalized).
            V: Potential energy on the grid.
            
        Returns:
            Potential energy expectation value.
        """
        V_exp = self.grid.integrate(np.conj(psi) * V * psi)
        return np.real(V_exp)
    
    def variance(
        self,
        psi: NDArray[np.complexfloating],
        V: NDArray[np.floating]
    ) -> float:
        """
        Calculate the energy variance <H²> - <H>².
        
        Useful for checking eigenstate quality.
        
        Args:
            psi: Wavefunction (should be normalized).
            V: Potential energy on the grid.
            
        Returns:
            Energy variance.
        """
        H_psi = self.apply_hamiltonian(psi, V)
        H2_psi = self.apply_hamiltonian(H_psi, V)
        
        E = np.real(self.grid.inner_product(psi, H_psi))
        E2 = np.real(self.grid.inner_product(psi, H2_psi))
        
        return E2 - E**2

