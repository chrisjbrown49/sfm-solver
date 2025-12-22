"""
Spatial Coupling Builder for SFM.

This module builds 4D nonseparable structure from subspace shape.
Still operates at dimensionless scale - no physical parameters involved.

KEY PRINCIPLE: Still Stage 1 (dimensionless)
    - Takes normalized subspace shape from shape solver
    - Builds spatial (n,l,m) components via perturbative coupling
    - All at unit spatial scale (no Delta_x)
    - Output normalized: integral_Sigma |chi_nlm|^2 = 1

The spatial-subspace coupling creates the generation hierarchy through
mixing of different (n,l,m) components.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from scipy.special import genlaguerre
import warnings


@dataclass
class SpatialState:
    """A spatial basis state |n, l, m>."""
    n: int  # Principal quantum number (1, 2, 3, ...)
    l: int  # Angular momentum (0, 1, 2, ...)
    m: int  # Magnetic quantum number (-l, ..., +l)
    
    def __hash__(self):
        return hash((self.n, self.l, self.m))
    
    def __eq__(self, other):
        return self.n == other.n and self.l == other.l and self.m == other.m
    
    def energy(self) -> int:
        """Harmonic oscillator energy: E = 2n + l"""
        return 2*self.n + self.l


class SpatialCouplingBuilder:
    """
    Build 4D nonseparable structure from subspace shape.
    
    Still dimensionless - operates at unit spatial scale.
    Creates induced (n,l,m) components through spatial-subspace coupling.
    """
    
    def __init__(
        self,
        alpha_dimensionless: float,
        n_max: int = 5,
        l_max: int = 2,
        a0: float = 1.0,
        verbose: bool = False
    ):
        """
        Initialize spatial coupling builder.
        
        Args:
            alpha_dimensionless: Spatial-subspace coupling (alpha / V0)
            n_max: Maximum spatial quantum number
            l_max: Maximum angular momentum
            a0: Characteristic length scale for basis (unit scale)
            verbose: Print diagnostic information
        """
        self.alpha_dimless = alpha_dimensionless
        self.n_max = n_max
        self.l_max = l_max
        self.a0 = a0
        self.verbose = verbose
        
        # Build spatial basis states
        self.spatial_states: List[SpatialState] = []
        for n in range(1, n_max + 1):
            for l in range(min(n, l_max + 1)):
                for m in range(-l, l + 1):
                    self.spatial_states.append(SpatialState(n, l, m))
        
        self.N_spatial = len(self.spatial_states)
        
        # Create index mapping
        self._state_to_idx: Dict[Tuple[int, int, int], int] = {}
        for i, state in enumerate(self.spatial_states):
            self._state_to_idx[(state.n, state.l, state.m)] = i
        
        # Compute coupling matrix R_ij = <phi_i|grad^2|phi_j>
        # This is dimensionless (gradients in unit length)
        self.R_coupling = self._compute_coupling_matrix()
        
        if self.verbose:
            print("=== SpatialCouplingBuilder Initialized ===")
            print(f"  alpha_dimensionless: {alpha_dimensionless:.6f}")
            print(f"  n_max: {n_max}, l_max: {l_max}")
            print(f"  Total spatial states: {self.N_spatial}")
    
    def build_4d_structure(
        self,
        subspace_shape: NDArray,
        n_target: int = 1,
        l_target: int = 0,
        m_target: int = 0,
        Delta_x: Optional[float] = None
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Build 4D wavefunction structure with spatial coupling.
        
        Takes the subspace shape chi(sigma) and creates the full
        nonseparable structure:
            psi(r, sigma) = Sum_{n,l,m} phi_nlm(r) chi_nlm(sigma)
        
        The chi_nlm components are related by spatial-subspace coupling.
        
        Input:
            subspace_shape: Normalized chi(sigma) with integral|chi|^2 = 1
            n_target, l_target, m_target: Primary quantum numbers
            Delta_x: Optional spatial scale in fm (for scale-aware coupling)
            
        Output:
            {(n,l,m): chi_nlm(sigma)} all dimensionless, normalized
            
        The key is that this builds the RELATIVE structure of different
        (n,l,m) components, independent of overall scale.
        
        If Delta_x is provided, recomputes coupling matrix at that scale.
        """
        if self.verbose:
            print(f"\n=== Building 4D Structure (n={n_target}, l={l_target}, m={m_target}) ===")
        
        chi_components = {}
        
        # Primary state (given)
        chi_components[(n_target, l_target, m_target)] = subspace_shape.copy()
        
        # Get target state index
        target_idx = self._get_state_index(n_target, l_target, m_target)
        if target_idx < 0:
            warnings.warn(f"Target state ({n_target},{l_target},{m_target}) not in basis")
            return chi_components
        
        # Recompute coupling matrix if scale provided (outer loop)
        if Delta_x is not None:
            a_n = Delta_x / np.sqrt(2 * n_target + 1)  # Characteristic scale
            R_coupling = self._compute_coupling_matrix_at_scale(a_n)
            if self.verbose:
                print(f"  Using scale-aware coupling at Delta_x={Delta_x:.6f} fm (a_n={a_n:.6f})")
        else:
            R_coupling = self.R_coupling  # Use pre-computed unit scale
        
        # Energy of target state (dimensionless, harmonic oscillator)
        E_target = 2*n_target + l_target
        
        # Build induced components from coupling
        for i, state in enumerate(self.spatial_states):
            n, l, m = state.n, state.l, state.m
            
            # Skip the primary state
            if (n, l, m) == (n_target, l_target, m_target):
                continue
            
            # Spatial coupling matrix element
            R_ij = R_coupling[i, target_idx]
            
            if abs(R_ij) < 1e-10:
                # No coupling, component is zero
                chi_components[(n, l, m)] = np.zeros_like(subspace_shape)
                continue
            
            # Energy denominator (dimensionless)
            E_state = 2*n + l
            E_denom = E_target - E_state
            
            # Regularization for near-degenerate states
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            # Induced component (perturbative)
            # chi_nlm = -alpha * R_ij * chi_primary / E_denom
            chi_nlm = -self.alpha_dimless * R_ij * subspace_shape / E_denom
            
            chi_components[(n, l, m)] = chi_nlm
        
        # RENORMALIZE the total 4D structure
        # NOTE: This is necessary because with large alpha_dimless >> 1, the perturbative
        # approximation breaks down and induced components become large. Renormalization
        # ensures the energy minimizer receives a properly normalized structure.
        N_sigma = len(subspace_shape)
        dsigma = 2*np.pi / N_sigma
        
        # Compute total norm
        chi_total = sum(chi_components.values())
        total_norm = np.sqrt(np.sum(np.abs(chi_total)**2) * dsigma)
        
        # Renormalize all components
        if total_norm > 1e-10:
            for key in chi_components:
                chi_components[key] = chi_components[key] / total_norm
        
        if self.verbose:
            # Report total norm for diagnostic purposes
            N_sigma = len(subspace_shape)
            dsigma = 2*np.pi / N_sigma
            total_norm_sq = 0.0
            for chi in chi_components.values():
                total_norm_sq += np.sum(np.abs(chi)**2) * dsigma
            
            primary_norm_sq = np.sum(np.abs(subspace_shape)**2) * dsigma
            
            print(f"  Created {len(chi_components)} spatial components")
            print(f"  Primary component norm: {np.sqrt(primary_norm_sq):.6f}")
            print(f"  Total structure norm: {np.sqrt(total_norm_sq):.6f}")
        
        return chi_components
    
    def _get_state_index(self, n: int, l: int, m: int) -> int:
        """Get the index of state |n, l, m> in spatial basis."""
        return self._state_to_idx.get((n, l, m), -1)
    
    def _compute_coupling_matrix(self) -> NDArray:
        """
        Compute spatial coupling matrix R_ij = <phi_i|grad^2|phi_j>.
        
        For 3D harmonic oscillator wavefunctions at unit scale.
        This is dimensionless (gradients in unit length).
        
        Returns:
            R_coupling[i, j]: Coupling between states i and j
        """
        N = self.N_spatial
        R = np.zeros((N, N), dtype=float)
        
        # Radial grid for integration (unit scale)
        r = np.linspace(0.01, 10.0 * self.a0, 200)
        dr = r[1] - r[0]
        
        for i, state_i in enumerate(self.spatial_states):
            for j, state_j in enumerate(self.spatial_states):
                # Compute <phi_i|grad^2|phi_j>
                # For harmonic oscillator, this has analytical properties
                # but we'll use numerical integration for generality
                
                R_ij = self._compute_laplacian_overlap(state_i, state_j, r, dr)
                R[i, j] = R_ij
        
        return R
    
    def _compute_coupling_matrix_at_scale(self, a_n: float) -> NDArray:
        """
        Compute R_ij = <phi_i|grad^2|phi_j> at spatial scale a_n.
        
        Harmonic oscillator wavefunctions scale as phi(r/a_n)/a_n^(3/2).
        Laplacian scales as grad^2 ~ 1/a_n^2.
        
        For the Laplacian operator acting on scaled wavefunctions:
            R(a_n) = R(a0=1) / a_n^2
        
        This is the analytical scaling for harmonic oscillator states.
        
        Args:
            a_n: Characteristic spatial scale in fm
            
        Returns:
            Coupling matrix scaled appropriately
        """
        # Scale from unit coupling: R(a_n) = R(a0=1) / a_n^2
        return self.R_coupling / (a_n ** 2)
    
    def _compute_laplacian_overlap(
        self,
        state_i: SpatialState,
        state_j: SpatialState,
        r: NDArray,
        dr: float
    ) -> float:
        """
        Compute <phi_i|grad^2|phi_j> for harmonic oscillator states.
        
        The Laplacian in spherical coordinates is:
            grad^2 = (1/r^2) d/dr(r^2 d/dr) + (1/r^2) L^2
        
        For states with same l and m, the angular part contributes:
            <Y_lm|L^2|Y_lm> = l(l+1)
        """
        ni, li, mi = state_i.n, state_i.l, state_i.m
        nj, lj, mj = state_j.n, state_j.l, state_j.m
        
        # Angular momentum selection rules
        if li != lj or mi != mj:
            # For simplicity, assume no coupling between different l,m
            # (proper implementation would include angular matrix elements)
            return 0.0
        
        # Get radial wavefunctions
        R_i = self._radial_wavefunction(ni, li, r)
        R_j = self._radial_wavefunction(nj, lj, r)
        
        # Compute radial derivative of R_j
        dR_j = np.gradient(R_j, dr)
        d2R_j = np.gradient(dR_j, dr)
        
        # Radial part of Laplacian:
        # (1/r^2) d/dr(r^2 dR/dr) = d^2R/dr^2 + (2/r) dR/dr
        laplacian_R_j = d2R_j + (2.0 / (r + 1e-20)) * dR_j
        
        # Angular part contributes l(l+1)/r^2
        laplacian_R_j += (li * (li + 1) / (r**2 + 1e-20)) * R_j
        
        # Overlap integral: <R_i | laplacian | R_j>
        integrand = R_i * laplacian_R_j * r**2
        overlap = np.trapz(integrand, dx=dr)
        
        return overlap
    
    def _radial_wavefunction(self, n: int, l: int, r: NDArray) -> NDArray:
        """
        Compute radial wavefunction R_{nl}(r) for 3D harmonic oscillator.
        
        R_{nl}(r) = N_{nl} * r^l * L_{n-l-1}^{l+1/2}(r^2/a0^2) * exp(-r^2/(2*a0^2))
        
        where L_k^alpha is the generalized Laguerre polynomial.
        
        Args:
            n: Principal quantum number
            l: Angular momentum
            r: Radial coordinate array
            
        Returns:
            Radial wavefunction values
        """
        if n < 1 or l < 0 or l >= n:
            return np.zeros_like(r)
        
        # Reduced coordinate
        rho = r**2 / self.a0**2
        
        # Laguerre polynomial
        k = n - l - 1
        alpha = l + 0.5
        
        if k < 0:
            return np.zeros_like(r)
        
        L = genlaguerre(k, alpha)(rho)
        
        # Radial wavefunction
        R = (r**l) * L * np.exp(-rho / 2.0)
        
        # Normalization (approximate, for relative structure)
        norm = np.sqrt(np.trapz(R**2 * r**2, dx=(r[1]-r[0])))
        if norm > 1e-20:
            R = R / norm
        
        return R

