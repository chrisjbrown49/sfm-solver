"""
Coupled subspace-spacetime eigenvalue solver.

This module implements the full coupled eigenvalue problem that includes
the spacetime-subspace coupling term responsible for the mass hierarchy.

The full 5D Hamiltonian (for spherically symmetric, l=0 states) is:

    Ĥ = -ℏ²/(2m) ∇² - ℏ²/(2m_σR²) ∂²/∂σ² + V(σ) + g₁|ψ|² - α ∇·(∂/∂σ)

For spherical symmetry with u(r) = r·φ(r):

    Ĥ = -ℏ²/(2m) d²/dr² - ℏ²/(2m_σR²) d²/dσ² + V(σ) + g₁|ψ|² - α (d²/dr dσ)

The coupling term -α(∂²/∂r∂σ) mixes the spatial and subspace derivatives,
preventing variable separation and creating the mass hierarchy:

    - Electron (n=1, k=1): Smooth spatial structure → minimal coupling → A²_e
    - Muon (n=2, k=1): One radial node → enhanced coupling → A²_μ ≈ 207×A²_e  
    - Tau (n=3, k=1): Two radial nodes → further enhanced → A²_τ ≈ 3477×A²_e

The coupling constant α is determined by fitting to the observed m_μ/m_e ratio,
then the tau mass becomes a prediction of the theory.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse import kron, eye, diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
import warnings

from sfm_solver.spatial.radial import RadialGrid, RadialOperators
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential


@dataclass
class CoupledGrids:
    """
    Combined radial-subspace grid structure.
    
    The total wavefunction lives on a 2D grid: ψ(r_i, σ_j).
    This is stored as a 1D vector with index = i * N_σ + j.
    
    Attributes:
        radial: RadialGrid for spatial dimension.
        subspace: SpectralGrid for subspace dimension.
        shape: Tuple (N_r, N_σ) giving 2D grid shape.
        size: Total number of grid points.
    """
    radial: RadialGrid
    subspace: SpectralGrid
    
    def __post_init__(self):
        """Initialize combined grid properties."""
        self.N_r = self.radial.N
        self.N_sigma = self.subspace.N
        self.shape = (self.N_r, self.N_sigma)
        self.size = self.N_r * self.N_sigma
    
    def to_2d(self, psi_flat: NDArray) -> NDArray:
        """
        Reshape flattened wavefunction to 2D array.
        
        Args:
            psi_flat: Wavefunction as 1D array of length N_r * N_σ.
            
        Returns:
            2D array of shape (N_r, N_σ).
        """
        return psi_flat.reshape(self.shape)
    
    def to_flat(self, psi_2d: NDArray) -> NDArray:
        """
        Flatten 2D wavefunction to 1D array.
        
        Args:
            psi_2d: Wavefunction as 2D array (N_r, N_σ).
            
        Returns:
            1D array of length N_r * N_σ.
        """
        return psi_2d.flatten()
    
    def integrate(self, psi: NDArray) -> float:
        """
        Integrate |ψ|² over full (r,σ) domain.
        
        ∫∫ |ψ|² r² dr dσ (with 4π from angular integration)
        
        Args:
            psi: Wavefunction (can be 1D or 2D).
            
        Returns:
            ∫ |ψ|² d³x dσ
        """
        psi_2d = psi.reshape(self.shape) if psi.ndim == 1 else psi
        
        # First integrate over σ at each r
        sigma_integral = np.trapezoid(np.abs(psi_2d)**2, self.subspace.sigma, axis=1)
        
        # Then integrate over r with r² weight
        return 4 * np.pi * np.trapezoid(sigma_integral * self.radial.r**2, self.radial.r)
    
    def normalize(self, psi: NDArray) -> NDArray:
        """
        Normalize wavefunction over full domain.
        
        Args:
            psi: Wavefunction to normalize.
            
        Returns:
            Normalized wavefunction (same shape as input).
        """
        norm = np.sqrt(self.integrate(psi))
        if norm > 1e-15:
            return psi / norm
        return psi


class CoupledHamiltonian:
    """
    Coupled radial-subspace Hamiltonian.
    
    Builds the full Hamiltonian matrix including the coupling term:
    
    Ĥ = T_r ⊗ I_σ + I_r ⊗ T_σ + I_r ⊗ V(σ) - α D_r ⊗ D_σ
    
    where:
    - T_r = -ℏ²/(2m) d²/dr² is radial kinetic energy
    - T_σ = -ℏ²/(2m_σR²) d²/dσ² is subspace kinetic energy
    - V(σ) is the three-well potential
    - D_r, D_σ are first derivative operators
    - α is the coupling constant
    
    The tensor product structure means the matrix has size (N_r × N_σ)².
    """
    
    def __init__(
        self,
        grids: CoupledGrids,
        potential: ThreeWellPotential,
        alpha: float = 0.0,
        m_spatial: float = 1.0,
        m_subspace: float = 1.0,
        R_subspace: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize coupled Hamiltonian.
        
        Args:
            grids: CoupledGrids with radial and subspace grids.
            potential: Three-well potential for subspace.
            alpha: Spacetime-subspace coupling constant.
            m_spatial: Effective mass for spatial motion.
            m_subspace: Effective mass for subspace motion.
            R_subspace: Subspace radius.
            hbar: Reduced Planck constant.
        """
        self.grids = grids
        self.potential = potential
        self.alpha = alpha
        self.m_spatial = m_spatial
        self.m_subspace = m_subspace
        self.R_subspace = R_subspace
        self.hbar = hbar
        
        # Build component operators
        self._build_operators()
    
    def _build_operators(self):
        """Build the component operators for the Hamiltonian."""
        N_r = self.grids.N_r
        N_sigma = self.grids.N_sigma
        
        # Radial operators
        self.radial_ops = RadialOperators(
            self.grids.radial, 
            self.m_spatial, 
            self.hbar
        )
        
        # Radial kinetic: -ℏ²/(2m) d²/dr²
        T_r = self.radial_ops.kinetic_matrix()
        
        # Radial first derivative (for coupling)
        D_r = self.radial_ops.D1
        
        # Subspace operators (using FFT-based spectral method)
        # Build finite difference version for sparse matrix
        dsigma = self.grids.subspace.dsigma
        
        # Subspace second derivative (periodic)
        main_diag = -2.0 * np.ones(N_sigma) / dsigma**2
        off_diag = 1.0 * np.ones(N_sigma - 1) / dsigma**2
        
        # Periodic boundary: connect first and last points
        T_sigma_diags = [
            (off_diag, -1),
            (main_diag, 0),
            (off_diag, 1),
        ]
        T_sigma = diags(
            [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(N_sigma, N_sigma),
            format='lil'
        )
        # Add periodic connections
        T_sigma[0, -1] = 1.0 / dsigma**2
        T_sigma[-1, 0] = 1.0 / dsigma**2
        T_sigma = T_sigma.tocsr()
        
        # Multiply by -ℏ²/(2m_σR²)
        kinetic_coeff_sigma = -self.hbar**2 / (2 * self.m_subspace * self.R_subspace**2)
        T_sigma = kinetic_coeff_sigma * T_sigma
        
        # Subspace first derivative (periodic, central differences)
        D_sigma = lil_matrix((N_sigma, N_sigma))
        for i in range(N_sigma):
            D_sigma[i, (i+1) % N_sigma] = 0.5 / dsigma
            D_sigma[i, (i-1) % N_sigma] = -0.5 / dsigma
        D_sigma = D_sigma.tocsr()
        
        # Subspace potential
        V_sigma = diags(
            [self.potential(self.grids.subspace.sigma)],
            [0],
            shape=(N_sigma, N_sigma),
            format='csr'
        )
        
        # Store operators for later use
        self.T_r = T_r
        self.T_sigma = T_sigma
        self.D_r = D_r
        self.D_sigma = D_sigma
        self.V_sigma = V_sigma
        
        # Identity matrices
        self.I_r = eye(N_r, format='csr')
        self.I_sigma = eye(N_sigma, format='csr')
    
    def build_matrix(
        self, 
        V_nonlinear: Optional[NDArray] = None
    ) -> csr_matrix:
        """
        Build the full coupled Hamiltonian matrix.
        
        H = T_r ⊗ I_σ + I_r ⊗ T_σ + I_r ⊗ V_σ + V_nl - α D_r ⊗ D_σ
        
        Args:
            V_nonlinear: Nonlinear potential g₁|ψ|² on 2D grid (optional).
            
        Returns:
            Sparse Hamiltonian matrix of size (N_r × N_σ)².
        """
        # Radial kinetic ⊗ Identity
        H = kron(self.T_r, self.I_sigma, format='csr')
        
        # Identity ⊗ Subspace kinetic
        H = H + kron(self.I_r, self.T_sigma, format='csr')
        
        # Identity ⊗ Subspace potential
        H = H + kron(self.I_r, self.V_sigma, format='csr')
        
        # Coupling term: -α D_r ⊗ D_σ
        if abs(self.alpha) > 1e-15:
            H = H - self.alpha * kron(self.D_r, self.D_sigma, format='csr')
        
        # Nonlinear potential (diagonal)
        if V_nonlinear is not None:
            V_flat = V_nonlinear.flatten()
            H = H + diags([V_flat], [0], shape=H.shape, format='csr')
        
        return H
    
    def apply(
        self, 
        psi: NDArray, 
        V_nonlinear: Optional[NDArray] = None
    ) -> NDArray:
        """
        Apply Hamiltonian to wavefunction without building full matrix.
        
        This is more memory-efficient for large grids.
        
        Args:
            psi: Wavefunction (flattened).
            V_nonlinear: Nonlinear potential (flattened, optional).
            
        Returns:
            H·ψ
        """
        psi_2d = self.grids.to_2d(psi)
        result_2d = np.zeros_like(psi_2d)
        
        # Radial kinetic: Apply T_r to each σ slice
        for j in range(self.grids.N_sigma):
            result_2d[:, j] += self.T_r @ psi_2d[:, j]
        
        # Subspace kinetic + potential: Apply to each r slice
        H_sigma = self.T_sigma + self.V_sigma
        for i in range(self.grids.N_r):
            result_2d[i, :] += H_sigma @ psi_2d[i, :]
        
        # Coupling term: -α D_r ⊗ D_σ
        if abs(self.alpha) > 1e-15:
            # First apply D_σ (along σ for each r)
            D_sigma_psi = np.zeros_like(psi_2d)
            for i in range(self.grids.N_r):
                D_sigma_psi[i, :] = self.D_sigma @ psi_2d[i, :]
            
            # Then apply D_r (along r for each σ)
            for j in range(self.grids.N_sigma):
                result_2d[:, j] -= self.alpha * (self.D_r @ D_sigma_psi[:, j])
        
        # Nonlinear potential
        if V_nonlinear is not None:
            V_2d = V_nonlinear.reshape(self.grids.shape) if V_nonlinear.ndim == 1 else V_nonlinear
            result_2d += V_2d * psi_2d
        
        return self.grids.to_flat(result_2d)


@dataclass
class CoupledSolution:
    """
    Solution of the coupled eigenvalue problem.
    
    Attributes:
        energy: Total energy eigenvalue.
        psi: 2D wavefunction ψ(r,σ).
        amplitude_squared: Total A² = ∫∫|ψ|² r² dr dσ.
        n_radial: Radial quantum number (number of nodes).
        k_subspace: Subspace winding number.
        converged: Whether self-consistent iteration converged.
        iterations: Number of iterations.
        coupling_energy: Energy from coupling term.
    """
    energy: float
    psi: NDArray
    amplitude_squared: float
    n_radial: int
    k_subspace: int
    converged: bool
    iterations: int
    coupling_energy: float


class CoupledEigensolver:
    """
    Solver for the coupled radial-subspace eigenvalue problem.
    
    Solves the nonlinear eigenvalue problem:
    
        [Ĥ_r + Ĥ_σ + Ĥ_coupling + g₁|ψ|²] ψ = E ψ
    
    using self-consistent iteration:
    1. Start with initial guess (separable ansatz)
    2. Compute effective potential V_eff = V(σ) + g₁|ψ|²
    3. Solve linear eigenvalue problem
    4. Update ψ and iterate until self-consistent
    
    The coupling term -α(∂²/∂r∂σ) cannot be separated, so we must
    work with the full 2D problem.
    """
    
    def __init__(
        self,
        grids: CoupledGrids,
        potential: ThreeWellPotential,
        alpha: float = 0.0,
        g1: float = 0.0,
        m_spatial: float = 1.0,
        m_subspace: float = 1.0,
        R_subspace: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize the coupled solver.
        
        Args:
            grids: CoupledGrids with radial and subspace grids.
            potential: Three-well potential.
            alpha: Coupling constant (determines mass hierarchy).
            g1: Nonlinear coupling constant.
            m_spatial: Spatial effective mass.
            m_subspace: Subspace effective mass.
            R_subspace: Subspace radius.
            hbar: Reduced Planck constant.
        """
        self.grids = grids
        self.potential = potential
        self.alpha = alpha
        self.g1 = g1
        
        self.hamiltonian = CoupledHamiltonian(
            grids, potential, alpha,
            m_spatial, m_subspace, R_subspace, hbar
        )
    
    def create_initial_guess(
        self,
        n_radial: int = 1,
        k_subspace: int = 1,
        radial_width: float = 1.0
    ) -> NDArray:
        """
        Create separable initial guess.
        
        ψ₀(r,σ) = u_n(r) · χ_k(σ)
        
        where:
        - u_n(r) is radial mode with (n-1) nodes
        - χ_k(σ) = exp(ikσ) is subspace plane wave
        
        Args:
            n_radial: Radial quantum number (1,2,3 for e,μ,τ).
            k_subspace: Subspace winding number.
            radial_width: Width parameter for radial Gaussian.
            
        Returns:
            Initial wavefunction on 2D grid, flattened.
        """
        # Radial part: Gaussian with (n-1) nodes
        u_r = self.grids.radial.create_gaussian(radial_width, n_nodes=n_radial-1)
        
        # Subspace part: plane wave exp(ikσ)
        sigma = self.grids.subspace.sigma
        chi_sigma = np.exp(1j * k_subspace * sigma)
        # Normalize over subspace
        chi_sigma = chi_sigma / np.sqrt(2 * np.pi)
        
        # Tensor product
        psi_2d = np.outer(u_r, chi_sigma)
        
        # Normalize over full domain
        psi_flat = self.grids.to_flat(psi_2d)
        return self.grids.normalize(psi_flat)
    
    def solve(
        self,
        n_radial: int = 1,
        k_subspace: int = 1,
        max_iter: int = 100,
        tol: float = 1e-8,
        mixing: float = 0.3,
        radial_width: float = 1.0,
        verbose: bool = False
    ) -> CoupledSolution:
        """
        Solve for a specific (n,k) state.
        
        Args:
            n_radial: Radial quantum number (1 for electron, 2 for muon, 3 for tau).
            k_subspace: Subspace winding number (1 for leptons).
            max_iter: Maximum self-consistent iterations.
            tol: Convergence tolerance.
            mixing: Mixing parameter for iteration.
            radial_width: Width of initial radial Gaussian.
            verbose: Print convergence info.
            
        Returns:
            CoupledSolution with energy, wavefunction, and amplitude.
        """
        # Initial guess
        psi = self.create_initial_guess(n_radial, k_subspace, radial_width)
        
        E_old = 0.0
        converged = False
        
        for iteration in range(max_iter):
            # Compute nonlinear potential
            if abs(self.g1) > 1e-15:
                V_nonlinear = self.g1 * np.abs(psi)**2
            else:
                V_nonlinear = None
            
            # Build Hamiltonian
            H = self.hamiltonian.build_matrix(V_nonlinear)
            
            # Make Hermitian (numerical errors can break symmetry)
            H = 0.5 * (H + H.T.conj())
            
            # Solve eigenvalue problem
            try:
                # Find lowest eigenvalue
                n_eigs = min(5, self.grids.size - 2)
                energies, wavefunctions = eigsh(H, k=n_eigs, which='SA')
                
                # Sort by energy
                idx = np.argsort(energies)
                E_new = energies[idx[0]]
                psi_new = wavefunctions[:, idx[0]]
                
            except Exception as e:
                warnings.warn(f"Eigenvalue solver failed: {e}")
                break
            
            # Normalize
            psi_new = self.grids.normalize(psi_new)
            
            # Mix with previous
            psi = (1 - mixing) * psi + mixing * psi_new
            psi = self.grids.normalize(psi)
            
            # Check convergence
            dE = abs(E_new - E_old)
            
            if verbose:
                A_sq = self.grids.integrate(psi)
                print(f"Iter {iteration+1}: E = {E_new:.8f}, dE = {dE:.2e}, A² = {A_sq:.6f}")
            
            if dE < tol:
                converged = True
                break
            
            E_old = E_new
        
        # Final calculations
        psi_2d = self.grids.to_2d(psi)
        A_squared = self.grids.integrate(psi)
        
        # Count radial nodes
        psi_r_avg = np.mean(np.abs(psi_2d), axis=1)  # Average over σ
        from sfm_solver.spatial.radial import count_nodes
        n_detected = count_nodes(psi_r_avg) + 1  # +1 because n starts at 1
        
        # Compute coupling energy
        E_coupling = self._compute_coupling_energy(psi)
        
        return CoupledSolution(
            energy=E_new,
            psi=psi_2d,
            amplitude_squared=A_squared,
            n_radial=n_detected,
            k_subspace=k_subspace,
            converged=converged,
            iterations=iteration + 1,
            coupling_energy=E_coupling
        )
    
    def _compute_coupling_energy(self, psi: NDArray) -> float:
        """
        Compute the coupling energy contribution.
        
        E_coupling = -α ⟨ψ| D_r ⊗ D_σ |ψ⟩
        
        Args:
            psi: Wavefunction (flattened).
            
        Returns:
            Coupling energy.
        """
        if abs(self.alpha) < 1e-15:
            return 0.0
        
        psi_2d = self.grids.to_2d(psi)
        
        # Apply D_σ first
        D_sigma_psi = np.zeros_like(psi_2d)
        for i in range(self.grids.N_r):
            D_sigma_psi[i, :] = self.hamiltonian.D_sigma @ psi_2d[i, :]
        
        # Then apply D_r
        D_r_D_sigma_psi = np.zeros_like(psi_2d)
        for j in range(self.grids.N_sigma):
            D_r_D_sigma_psi[:, j] = self.hamiltonian.D_r @ D_sigma_psi[:, j]
        
        # Compute expectation value
        integrand = np.conj(psi_2d) * D_r_D_sigma_psi
        
        # Integrate over σ then r
        sigma_int = np.trapezoid(integrand, self.grids.subspace.sigma, axis=1)
        r_int = 4 * np.pi * np.trapezoid(sigma_int * self.grids.radial.r**2, self.grids.radial.r)
        
        return -self.alpha * np.real(r_int)
    
    def solve_spectrum(
        self,
        n_radial_values: List[int] = [1, 2, 3],
        k_subspace: int = 1,
        **kwargs
    ) -> List[CoupledSolution]:
        """
        Solve for multiple radial modes (e.g., electron, muon, tau).
        
        Args:
            n_radial_values: List of radial quantum numbers.
            k_subspace: Subspace winding number.
            **kwargs: Additional arguments passed to solve().
            
        Returns:
            List of CoupledSolution for each n.
        """
        solutions = []
        
        for n in n_radial_values:
            # Adjust radial width based on n
            width = kwargs.pop('radial_width', 1.0) * np.sqrt(n)
            
            sol = self.solve(
                n_radial=n,
                k_subspace=k_subspace,
                radial_width=width,
                **kwargs
            )
            solutions.append(sol)
        
        return solutions


def compute_mass_ratios(solutions: List[CoupledSolution]) -> Dict[str, float]:
    """
    Compute mass ratios from solved states.
    
    Mass is proportional to A², so:
        m_μ/m_e = A²_μ/A²_e
        m_τ/m_e = A²_τ/A²_e
    
    Args:
        solutions: List of CoupledSolution for different n values.
        
    Returns:
        Dictionary with mass ratios.
    """
    if len(solutions) < 2:
        return {}
    
    # Use first solution (n=1, electron) as reference
    A_e_sq = solutions[0].amplitude_squared
    
    ratios = {'A_e_squared': A_e_sq}
    
    if len(solutions) >= 2:
        A_mu_sq = solutions[1].amplitude_squared
        ratios['A_mu_squared'] = A_mu_sq
        ratios['mu_e_ratio'] = A_mu_sq / A_e_sq
    
    if len(solutions) >= 3:
        A_tau_sq = solutions[2].amplitude_squared
        ratios['A_tau_squared'] = A_tau_sq
        ratios['tau_e_ratio'] = A_tau_sq / A_e_sq
    
    return ratios

