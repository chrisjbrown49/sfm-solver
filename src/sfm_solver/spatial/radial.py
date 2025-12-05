"""
Radial spatial grid and operators for spherically symmetric solutions.

In the SFM framework, particles can be described by wavefunctions that
are products of spatial and subspace components:
    ψ(x,σ) ≈ φ(x) · χ(σ)

For spherically symmetric solutions, φ(x) = φ(r)/r where r = |x|,
reducing the 3D spatial problem to a 1D radial problem.

The coupling between spatial and subspace dimensions:
    Ĥ_coupling = -α (∂²/∂x∂σ + ∂²/∂y∂σ + ∂²/∂z∂σ)

For spherical symmetry becomes:
    Ĥ_coupling = -α (∂²/∂r∂σ) (plus angular terms that vanish for l=0)

Different spatial radial modes (n=1,2,3,...) have different nodal structures:
- n=1 (electron): No radial nodes, ground state
- n=2 (muon): One radial node
- n=3 (tau): Two radial nodes

The coupling energy depends on gradient structure, creating mass hierarchy.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse import diags, csr_matrix


@dataclass
class RadialGrid:
    """
    Radial grid for spherically symmetric wavefunctions.
    
    Uses a uniform grid in r with appropriate boundary conditions:
    - φ(0) must be finite (regularity at origin)
    - φ(r_max) → 0 (wavefunction decay)
    
    For numerical stability, we work with u(r) = r·φ(r), which satisfies
    a simpler equation with u(0) = 0.
    
    Attributes:
        N: Number of grid points.
        r_max: Maximum radius (should be large enough for wavefunction decay).
        r: Radial grid points [0, r_max].
        dr: Grid spacing.
        r_inner: Grid points excluding boundaries (for finite differences).
    """
    N: int
    r_max: float
    
    def __post_init__(self):
        """Initialize grid arrays after dataclass initialization."""
        # Uniform grid including endpoints
        self.r = np.linspace(0, self.r_max, self.N)
        self.dr = self.r[1] - self.r[0]
        
        # Inner points (excluding r=0 and r=r_max for boundary conditions)
        self.r_inner = self.r[1:-1]
        self.N_inner = len(self.r_inner)
    
    def integrate(self, f: NDArray) -> float:
        """
        Integrate function f(r) over radial coordinate with r² weight.
        
        For 3D integration of spherically symmetric function:
            ∫ f(r) d³x = 4π ∫₀^∞ f(r) r² dr
        
        Args:
            f: Function values on radial grid.
            
        Returns:
            4π ∫ f(r) r² dr
        """
        integrand = f * self.r**2
        return 4 * np.pi * np.trapezoid(integrand, self.r)
    
    def integrate_u(self, u: NDArray) -> float:
        """
        Integrate |u(r)|² where u = r·φ.
        
        For normalization: ∫|φ|² r² dr = ∫|u|² dr
        
        Args:
            u: Reduced wavefunction u = r·φ on grid.
            
        Returns:
            4π ∫ |u|² dr
        """
        return 4 * np.pi * np.trapezoid(np.abs(u)**2, self.r)
    
    def normalize_u(self, u: NDArray) -> NDArray:
        """
        Normalize reduced wavefunction u = r·φ.
        
        Normalization: 4π ∫|u|² dr = 1
        
        Args:
            u: Reduced wavefunction to normalize.
            
        Returns:
            Normalized wavefunction.
        """
        norm = np.sqrt(self.integrate_u(u))
        if norm > 1e-15:
            return u / norm
        return u
    
    def u_to_phi(self, u: NDArray) -> NDArray:
        """
        Convert reduced wavefunction u to physical wavefunction φ.
        
        φ(r) = u(r)/r for r > 0
        φ(0) = lim_{r→0} u(r)/r = u'(0) (by L'Hôpital)
        
        Args:
            u: Reduced wavefunction u = r·φ.
            
        Returns:
            Physical wavefunction φ(r).
        """
        phi = np.zeros_like(u)
        # For r > 0: φ = u/r
        nonzero = self.r > 0
        phi[nonzero] = u[nonzero] / self.r[nonzero]
        # At r = 0: use limit (derivative of u at origin)
        if self.N > 1:
            phi[0] = (u[1] - u[0]) / self.dr
        return phi
    
    def phi_to_u(self, phi: NDArray) -> NDArray:
        """
        Convert physical wavefunction φ to reduced form u.
        
        u(r) = r·φ(r)
        
        Args:
            phi: Physical wavefunction φ(r).
            
        Returns:
            Reduced wavefunction u = r·φ.
        """
        return self.r * phi
    
    def create_gaussian(self, width: float, n_nodes: int = 0) -> NDArray:
        """
        Create Gaussian-like initial guess with specified number of nodes.
        
        The spatial structure for different generations:
        - n=1 (electron): exp(-r²/2σ²), no nodes
        - n=2 (muon): r·exp(-r²/2σ²), one node at r=0
        - n=3 (tau): (r²-σ²)·exp(-r²/2σ²), two nodes
        
        Args:
            width: Characteristic width of Gaussian.
            n_nodes: Number of radial nodes (0, 1, or 2).
            
        Returns:
            Reduced wavefunction u(r) = r·φ(r) on grid.
        """
        r = self.r
        
        if n_nodes == 0:
            # Ground state: φ ~ exp(-r²/2w²), u ~ r·exp(-r²/2w²)
            phi = np.exp(-r**2 / (2 * width**2))
        elif n_nodes == 1:
            # First excitation: φ ~ r·exp(-r²/2w²), u ~ r²·exp(-r²/2w²)
            phi = r * np.exp(-r**2 / (2 * width**2))
        elif n_nodes == 2:
            # Second excitation: φ ~ (r²-w²)·exp(-r²/2w²)
            phi = (r**2 - width**2) * np.exp(-r**2 / (2 * width**2))
        else:
            # General case: use Laguerre-like polynomial
            from scipy.special import genlaguerre
            L = genlaguerre(n_nodes, 0.5)
            phi = L(r**2 / width**2) * np.exp(-r**2 / (2 * width**2))
        
        u = self.phi_to_u(phi)
        return self.normalize_u(u)


class RadialOperators:
    """
    Differential operators for radial Schrödinger equation.
    
    For the reduced wavefunction u = r·φ, the radial Schrödinger equation
    becomes:
        [-ℏ²/(2m) d²/dr² + V_eff(r)] u = E u
    
    where V_eff includes centrifugal terms for l > 0.
    
    For the spacetime-subspace coupling, we need:
        ∂u/∂r (first derivative for coupling term)
    
    Attributes:
        grid: RadialGrid instance.
        m_eff: Effective mass in spatial dimensions.
        hbar: Reduced Planck constant.
    """
    
    def __init__(
        self,
        grid: RadialGrid,
        m_eff: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize radial operators.
        
        Args:
            grid: RadialGrid defining the discretization.
            m_eff: Effective mass parameter.
            hbar: Reduced Planck constant.
        """
        self.grid = grid
        self.m_eff = m_eff
        self.hbar = hbar
        
        # Pre-compute kinetic energy coefficient
        self.kinetic_coeff = -hbar**2 / (2 * m_eff)
        
        # Build operator matrices
        self._build_operators()
    
    def _build_operators(self):
        """Build sparse matrix representations of operators."""
        N = self.grid.N
        dr = self.grid.dr
        
        # Second derivative matrix (finite differences)
        # d²u/dr² ≈ (u_{i+1} - 2u_i + u_{i-1}) / dr²
        main_diag = -2.0 * np.ones(N) / dr**2
        off_diag = 1.0 * np.ones(N - 1) / dr**2
        
        self.D2 = diags(
            [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(N, N),
            format='csr'
        )
        
        # First derivative matrix (central differences)
        # du/dr ≈ (u_{i+1} - u_{i-1}) / (2·dr)
        off_diag_p = 0.5 * np.ones(N - 1) / dr
        off_diag_m = -0.5 * np.ones(N - 1) / dr
        
        self.D1 = diags(
            [off_diag_m, np.zeros(N), off_diag_p],
            [-1, 0, 1],
            shape=(N, N),
            format='csr'
        )
        
        # Apply boundary conditions
        # u(0) = 0 (regularity at origin)
        # u(r_max) = 0 (decay at infinity)
        self._apply_boundary_conditions()
    
    def _apply_boundary_conditions(self):
        """
        Apply Dirichlet boundary conditions.
        
        For u = r·φ:
        - u(0) = 0 (since r=0)
        - u(r_max) = 0 (wavefunction decay)
        """
        N = self.grid.N
        
        # Convert to lil_matrix for efficient modification
        D2_lil = self.D2.tolil()
        D1_lil = self.D1.tolil()
        
        # Zero out first and last rows (boundary points)
        D2_lil[0, :] = 0
        D2_lil[-1, :] = 0
        D2_lil[0, 0] = 1
        D2_lil[-1, -1] = 1
        
        D1_lil[0, :] = 0
        D1_lil[-1, :] = 0
        
        self.D2 = D2_lil.tocsr()
        self.D1 = D1_lil.tocsr()
    
    def kinetic_matrix(self) -> csr_matrix:
        """
        Build kinetic energy operator matrix.
        
        T = -ℏ²/(2m) d²/dr²
        
        Returns:
            Sparse matrix for kinetic operator.
        """
        return self.kinetic_coeff * self.D2
    
    def apply_kinetic(self, u: NDArray) -> NDArray:
        """
        Apply kinetic energy operator to wavefunction.
        
        Args:
            u: Reduced wavefunction.
            
        Returns:
            T·u = -ℏ²/(2m) d²u/dr²
        """
        return self.kinetic_matrix() @ u
    
    def apply_derivative(self, u: NDArray) -> NDArray:
        """
        Apply first derivative operator.
        
        This is needed for the spacetime-subspace coupling term.
        
        Args:
            u: Reduced wavefunction.
            
        Returns:
            du/dr
        """
        return self.D1 @ u
    
    def build_hamiltonian(
        self,
        V: Optional[NDArray] = None,
        l: int = 0
    ) -> csr_matrix:
        """
        Build full radial Hamiltonian matrix.
        
        H = T + V + V_centrifugal
        
        For l=0 (s-waves), V_centrifugal = 0.
        For l>0, V_centrifugal = ℏ²l(l+1)/(2mr²).
        
        Args:
            V: Potential energy on grid (or None for free particle).
            l: Angular momentum quantum number.
            
        Returns:
            Sparse Hamiltonian matrix.
        """
        N = self.grid.N
        r = self.grid.r
        
        # Start with kinetic energy
        H = self.kinetic_matrix().copy()
        
        # Add potential energy
        if V is not None:
            H = H + diags([V], [0], shape=(N, N), format='csr')
        
        # Add centrifugal term for l > 0
        if l > 0:
            # V_cent = ℏ²l(l+1)/(2mr²)
            # Avoid division by zero at r=0
            V_cent = np.zeros(N)
            nonzero = r > 0
            V_cent[nonzero] = (
                self.hbar**2 * l * (l + 1) / 
                (2 * self.m_eff * r[nonzero]**2)
            )
            H = H + diags([V_cent], [0], shape=(N, N), format='csr')
        
        return H
    
    def kinetic_expectation(self, u: NDArray) -> float:
        """
        Compute kinetic energy expectation value.
        
        <T> = ∫ u* (-ℏ²/2m d²/dr²) u dr
        
        For normalized u (4π∫|u|²dr = 1).
        
        Args:
            u: Reduced wavefunction.
            
        Returns:
            Kinetic energy expectation value.
        """
        Tu = self.apply_kinetic(u)
        # 4π factor for spherical integration
        return 4 * np.pi * np.real(np.trapezoid(np.conj(u) * Tu, self.grid.r))
    
    def potential_expectation(self, u: NDArray, V: NDArray) -> float:
        """
        Compute potential energy expectation value.
        
        <V> = ∫ u* V u dr = ∫ V |u|² dr
        
        Args:
            u: Reduced wavefunction.
            V: Potential on radial grid.
            
        Returns:
            Potential energy expectation value.
        """
        return 4 * np.pi * np.trapezoid(V * np.abs(u)**2, self.grid.r)
    
    def gradient_coupling_matrix(self) -> csr_matrix:
        """
        Get the first derivative matrix for coupling calculations.
        
        For the spacetime-subspace coupling:
            Ĥ_coupling = -α (∂²/∂r∂σ)
        
        We need ∂/∂r to be applied in the tensor product with ∂/∂σ.
        
        Returns:
            Sparse first derivative matrix.
        """
        return self.D1.copy()


def create_harmonic_potential(
    grid: RadialGrid,
    omega: float = 1.0,
    m: float = 1.0
) -> NDArray:
    """
    Create harmonic oscillator potential.
    
    V(r) = (1/2) m ω² r²
    
    Useful for testing and as a confining potential.
    
    Args:
        grid: RadialGrid.
        omega: Angular frequency.
        m: Mass.
        
    Returns:
        Potential values on grid.
    """
    return 0.5 * m * omega**2 * grid.r**2


def create_coulomb_potential(
    grid: RadialGrid,
    Z: float = 1.0,
    a0: float = 1.0
) -> NDArray:
    """
    Create Coulomb potential.
    
    V(r) = -Z e²/(4πε₀ r) = -Z/r (in atomic units with a₀)
    
    Args:
        grid: RadialGrid.
        Z: Nuclear charge.
        a0: Bohr radius (sets units).
        
    Returns:
        Potential values on grid.
    """
    V = np.zeros(grid.N)
    nonzero = grid.r > 0
    V[nonzero] = -Z / (grid.r[nonzero] / a0)
    # Regularize at origin (use finite cutoff)
    V[0] = V[1] if grid.N > 1 else 0
    return V


def solve_radial_eigenstates(
    grid: RadialGrid,
    V: NDArray,
    n_states: int = 3,
    m_eff: float = 1.0,
    hbar: float = 1.0,
    l: int = 0
) -> Tuple[NDArray, NDArray]:
    """
    Solve for radial eigenstates.
    
    Finds the lowest n_states eigenvalues and eigenfunctions
    of the radial Schrödinger equation.
    
    Args:
        grid: RadialGrid.
        V: Potential on grid.
        n_states: Number of states to find.
        m_eff: Effective mass.
        hbar: Reduced Planck constant.
        l: Angular momentum quantum number.
        
    Returns:
        Tuple of (energies, wavefunctions).
        energies: Array of shape (n_states,).
        wavefunctions: Array of shape (N, n_states).
    """
    from scipy.sparse.linalg import eigsh
    
    ops = RadialOperators(grid, m_eff, hbar)
    H = ops.build_hamiltonian(V, l)
    
    # Use eigsh for sparse symmetric matrices
    # Request smallest eigenvalues (ground state and excited states)
    energies, wavefunctions = eigsh(
        H, 
        k=n_states, 
        which='SA',  # Smallest algebraic
        return_eigenvectors=True
    )
    
    # Sort by energy
    idx = np.argsort(energies)
    energies = energies[idx]
    wavefunctions = wavefunctions[:, idx]
    
    # Normalize each eigenstate
    for i in range(n_states):
        wavefunctions[:, i] = grid.normalize_u(wavefunctions[:, i])
    
    return energies, wavefunctions


def count_nodes(u: NDArray) -> int:
    """
    Count the number of nodes (zero crossings) in wavefunction.
    
    Useful for identifying the radial quantum number n.
    
    Args:
        u: Reduced wavefunction on grid.
        
    Returns:
        Number of interior nodes (excluding boundaries).
    """
    # Find sign changes
    signs = np.sign(np.real(u))
    # Count where sign changes (excluding endpoints)
    sign_changes = np.abs(np.diff(signs)) > 1
    return np.sum(sign_changes)

