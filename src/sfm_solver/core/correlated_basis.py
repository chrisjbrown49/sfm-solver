"""
Correlated Basis for Non-Separable SFM Wavefunctions.

This module provides the basis functions for the full non-separable solution
where angular distortions correlate with subspace position σ.

The wavefunction ansatz is:
    ψ(r,θ,φ,σ) = Σ_{n,l,m} R_{nl}(r) Y_l^m(θ,φ) χ_{nlm}(σ)

Each angular component (n,l,m) has its OWN subspace function χ_{nlm}(σ).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import genlaguerre, sph_harm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings


@dataclass
class SpatialState:
    """A spatial basis state |n, l, m⟩."""
    n: int  # Principal quantum number (1, 2, 3, ...)
    l: int  # Angular momentum quantum number (0, 1, 2, ...)
    m: int  # Magnetic quantum number (-l, ..., +l)
    
    def __hash__(self):
        return hash((self.n, self.l, self.m))
    
    def __eq__(self, other):
        return self.n == other.n and self.l == other.l and self.m == other.m


class CorrelatedBasis:
    """
    Basis for non-separable wavefunctions with correlated angular-subspace structure.
    
    Each spatial state |n, l, m⟩ has its own subspace wavefunction χ_{nlm}(σ).
    The key physics: the coupling Hamiltonian creates correlations where
    the p-wave (l=1) component's subspace function is driven by ∂χ_{l=0}/∂σ.
    """
    
    def __init__(
        self,
        n_max: int = 5,
        l_max: int = 2,
        N_sigma: int = 64,
        a0: float = 1.0,
    ):
        """
        Initialize the correlated basis.
        
        Args:
            n_max: Maximum principal quantum number
            l_max: Maximum angular momentum quantum number
            N_sigma: Number of grid points in subspace dimension
            a0: Characteristic length scale for radial functions
        """
        self.n_max = n_max
        self.l_max = l_max
        self.N_sigma = N_sigma
        self.a0 = a0
        
        # Build list of spatial states
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
        
        # Subspace grid
        self.sigma = np.linspace(0, 2 * np.pi, N_sigma, endpoint=False)
        self.dsigma = self.sigma[1] - self.sigma[0]
        
        # Radial grid (for computing integrals)
        self.r = np.linspace(0.01, 10.0 * a0, 200)
        self.dr = self.r[1] - self.r[0]
    
    def state_index(self, n: int, l: int, m: int) -> int:
        """Get the index of state |n, l, m⟩."""
        return self._state_to_idx.get((n, l, m), -1)
    
    def radial_function(self, n: int, l: int, r: Optional[NDArray] = None) -> NDArray:
        """
        Compute the radial wavefunction R_{nl}(r) for 3D harmonic oscillator.
        
        R_{nl}(r) ∝ r^l × L_{n-l-1}^{l+1/2}(r²/a₀²) × exp(-r²/(2a₀²))
        
        where L_k^α is the generalized Laguerre polynomial.
        
        NOTE: This uses a FIXED reference scale a0 for computing the wavefunction
        STRUCTURE and coupling matrix. The actual spatial SCALE (Δx) is an
        optimization variable in the energy minimizer.
        """
        if r is None:
            r = self.r
        
        # Validate quantum numbers
        if l >= n or l < 0:
            return np.zeros_like(r)
        
        # Use FIXED reference scale for structure computation
        x = (r / self.a0) ** 2
        
        # Polynomial degree
        k = n - l - 1
        alpha = l + 0.5
        
        # Generalized Laguerre polynomial
        if k >= 0:
            L = genlaguerre(k, alpha)(x)
        else:
            L = np.ones_like(x)
        
        # Full radial function (unnormalized)
        R = (r / self.a0) ** l * L * np.exp(-x / 2)
        
        # Normalize: ∫ R²(r) r² dr = 1
        norm = np.sqrt(np.trapz(R**2 * r**2, r))
        if norm > 1e-10:
            R = R / norm
        
        return R
    
    def radial_gradient(self, n: int, l: int, r: Optional[NDArray] = None) -> NDArray:
        """Compute dR_{nl}/dr numerically."""
        if r is None:
            r = self.r
        
        R = self.radial_function(n, l, r)
        dR = np.gradient(R, r)
        return dR
    
    def angular_coupling_factor(self, l1: int, m1: int, l2: int, m2: int) -> complex:
        """
        Compute the angular coupling factor for the spatial-subspace coupling.
        
        The coupling involves ∂/∂x, ∂/∂y, ∂/∂z which in spherical coords
        connect states with Δl = ±1.
        
        Returns the sum over x, y, z components:
        Σ_q ∫ Y_{l1}^{m1}* × (x_q/r) × Y_{l2}^{m2} dΩ
        
        Selection rules: |l1 - l2| = 1, |m1 - m2| ≤ 1
        """
        if abs(l1 - l2) != 1:
            return 0.0
        
        # Use Clebsch-Gordan coefficients
        # ∫ Y_{l1}^{m1}* Y_1^q Y_{l2}^{m2} dΩ = CG coefficient × normalization
        
        # For the coupling Σ_q ∂/∂x_q, we sum over q = -1, 0, +1
        # corresponding to (x-iy)/√2, z, -(x+iy)/√2
        
        total = 0.0
        
        for q in [-1, 0, 1]:
            # Selection rule on m
            if m1 != m2 + q:
                continue
            
            # Clebsch-Gordan coefficient ⟨l1 m1 | l2 m2; 1 q⟩
            # Using simplified formula for l2 = l1 ± 1
            
            if l2 == l1 + 1:
                # l1 → l1+1 transition
                cg = self._clebsch_gordan_lplus1(l1, m1, m2, q)
            elif l2 == l1 - 1:
                # l1 → l1-1 transition
                cg = self._clebsch_gordan_lminus1(l1, m1, m2, q)
            else:
                cg = 0.0
            
            total += cg
        
        return total
    
    def _clebsch_gordan_lplus1(self, l: int, m1: int, m2: int, q: int) -> float:
        """CG coefficient for l → l+1 transition via Y_1^q."""
        # ⟨l+1, m1 | l, m2; 1, q⟩ where m1 = m2 + q
        
        l1 = l + 1
        
        if m1 != m2 + q:
            return 0.0
        
        if abs(m1) > l1 or abs(m2) > l:
            return 0.0
        
        # Normalized CG coefficients
        # These come from standard angular momentum addition
        
        prefactor = np.sqrt((2*l + 1) / (4 * np.pi * (2*l + 3)))
        
        if q == 0:  # z component
            factor = np.sqrt((l + 1 + m1) * (l + 1 - m1) / ((2*l + 1) * (2*l + 3)))
        elif q == 1:  # (x + iy)/√2 component  
            factor = -np.sqrt((l + 1 + m1) * (l + m1) / (2 * (2*l + 1) * (2*l + 3)))
        elif q == -1:  # (x - iy)/√2 component
            factor = np.sqrt((l + 1 - m1) * (l - m1 + 2) / (2 * (2*l + 1) * (2*l + 3)))
        else:
            factor = 0.0
        
        return prefactor * factor
    
    def _clebsch_gordan_lminus1(self, l: int, m1: int, m2: int, q: int) -> float:
        """CG coefficient for l → l-1 transition via Y_1^q."""
        if l == 0:
            return 0.0
        
        l1 = l - 1
        
        if m1 != m2 + q:
            return 0.0
        
        if abs(m1) > l1 or abs(m2) > l:
            return 0.0
        
        prefactor = np.sqrt((2*l + 1) / (4 * np.pi * (2*l - 1)))
        
        if q == 0:  # z component
            factor = np.sqrt((l + m2) * (l - m2) / ((2*l - 1) * (2*l + 1)))
        elif q == 1:  # (x + iy)/√2 component
            factor = np.sqrt((l - m1) * (l - m1 - 1) / (2 * (2*l - 1) * (2*l + 1)))
        elif q == -1:  # (x - iy)/√2 component
            factor = -np.sqrt((l + m1) * (l + m1 - 1) / (2 * (2*l - 1) * (2*l + 1)))
        else:
            factor = 0.0
        
        return prefactor * factor
    
    def compute_radial_coupling_integral(
        self,
        n1: int, l1: int,
        n2: int, l2: int,
    ) -> float:
        """
        Compute the radial part of the coupling matrix element.
        
        ∫ R_{n1,l1}(r) × (d/dr)[R_{n2,l2}(r)] × r² dr
        
        This is non-zero even for n1 = n2 when l1 ≠ l2.
        """
        r = self.r
        
        R1 = self.radial_function(n1, l1, r)
        dR2 = self.radial_gradient(n2, l2, r)
        
        # The coupling integral
        integrand = R1 * dR2 * r**2
        result = np.trapz(integrand, r)
        
        return result
    
    def compute_full_coupling_matrix_element(
        self,
        state1: SpatialState,
        state2: SpatialState,
    ) -> complex:
        """
        Compute the full spatial coupling matrix element:
        
        ⟨n1,l1,m1|Ĥ_coupling_spatial|n2,l2,m2⟩
        
        This is the product of radial and angular parts.
        """
        # Angular factor (selection rule Δl = ±1)
        angular = self.angular_coupling_factor(state1.l, state1.m, state2.l, state2.m)
        
        if abs(angular) < 1e-10:
            return 0.0
        
        # Radial factor
        radial = self.compute_radial_coupling_integral(
            state1.n, state1.l,
            state2.n, state2.l
        )
        
        return radial * angular
    
    # =========================================================================
    # SELF-CONSISTENT Δx ITERATION METHODS (Step 1 of missing physics fix)
    # =========================================================================
    
    def _harmonic_radial_at_scale(
        self, 
        n: int, 
        l: int, 
        r: NDArray, 
        a_scale: float
    ) -> NDArray:
        """
        Compute the radial wavefunction R_{nl}(r) for given scale parameter.
        
        R_{nl}(r) ∝ r^l × L_{n-l-1}^{l+1/2}(r²/a²) × exp(-r²/(2a²))
        
        Args:
            n: Principal quantum number
            l: Angular momentum quantum number
            r: Radial grid
            a_scale: Harmonic oscillator length scale (dynamic!)
            
        Returns:
            Normalized radial wavefunction
        """
        # Validate quantum numbers
        if l >= n or l < 0:
            return np.zeros_like(r)
        
        x = (r / a_scale) ** 2
        
        # Polynomial degree
        k = n - l - 1
        alpha = l + 0.5
        
        # Generalized Laguerre polynomial
        if k >= 0:
            L = genlaguerre(k, alpha)(x)
        else:
            L = np.ones_like(x)
        
        # Full radial function (unnormalized)
        R = (r / a_scale) ** l * L * np.exp(-x / 2)
        
        # Normalize: ∫ R²(r) r² dr = 1
        norm = np.sqrt(np.trapz(R**2 * r**2, r))
        if norm > 1e-10:
            R = R / norm
        
        return R
    
    def _harmonic_radial_derivative_at_scale(
        self, 
        n: int, 
        l: int, 
        r: NDArray, 
        a_scale: float
    ) -> NDArray:
        """
        Compute dR_{nl}/dr for given scale parameter.
        
        Uses numerical differentiation of the radial wavefunction.
        
        Args:
            n: Principal quantum number
            l: Angular momentum quantum number
            r: Radial grid
            a_scale: Harmonic oscillator length scale
            
        Returns:
            Radial derivative dR_{nl}/dr
        """
        R = self._harmonic_radial_at_scale(n, l, r, a_scale)
        dR = np.gradient(R, r)
        return dR
    
    def compute_spatial_coupling_at_scale(self, a_scale: float) -> NDArray:
        """
        Compute spatial coupling matrix R_ij for given harmonic oscillator scale.
        
        This is the KEY method for self-consistent Δx iteration!
        
        During self-consistent iteration, the scale parameter a_scale depends 
        on the particle's mass (which depends on amplitude A). This method 
        recomputes the coupling matrix for the current scale.
        
        The coupling matrix element is:
            R_{ij} = ⟨φ_i|∇|φ_j⟩ = ∫ R_i(r) × (dR_j/dr) × r² dr × angular_factor
        
        For Δl = ±1 transitions only (selection rule from ∂/∂x, ∂/∂y, ∂/∂z).
        
        Args:
            a_scale: Harmonic oscillator length scale in GeV⁻¹
            
        Returns:
            Coupling matrix R_ij of shape (N_spatial, N_spatial)
        """
        N = len(self.spatial_states)
        R = np.zeros((N, N), dtype=float)
        
        # Radial grid for integration (extend if scale is larger)
        r_max = max(20.0, 10.0 * a_scale)
        r = np.linspace(0.01, r_max, 500)
        
        for i, state_i in enumerate(self.spatial_states):
            for j, state_j in enumerate(self.spatial_states):
                # Check selection rule: Δl = ±1
                if abs(state_i.l - state_j.l) != 1:
                    continue
                
                # Angular factor (Clebsch-Gordan coefficient)
                angular = self.angular_coupling_factor(
                    state_i.l, state_i.m, 
                    state_j.l, state_j.m
                )
                
                if abs(angular) < 1e-10:
                    continue
                
                # Compute radial wavefunctions with given scale
                phi_i = self._harmonic_radial_at_scale(state_i.n, state_i.l, r, a_scale)
                dphi_j = self._harmonic_radial_derivative_at_scale(state_j.n, state_j.l, r, a_scale)
                
                # Radial integral: ∫ φ_i × (dφ_j/dr) × r² dr
                radial = np.trapz(phi_i * dphi_j * r**2, r)
                
                R[i, j] = radial * np.real(angular)
        
        return R
    
    def compute_coupling_strength_at_scale(
        self, 
        n_target: int, 
        a_scale: float
    ) -> float:
        """
        Compute effective coupling strength for target state at given scale.
        
        This is a summary metric useful for diagnosing self-consistent iteration.
        
        Returns the maximum absolute value of coupling from the target n=n_target,
        l=0, m=0 state to any other state.
        
        Args:
            n_target: Target spatial quantum number
            a_scale: Harmonic oscillator length scale
            
        Returns:
            Maximum |R_{target,j}| for j ≠ target
        """
        R = self.compute_spatial_coupling_at_scale(a_scale)
        target_idx = self.state_index(n_target, 0, 0)
        
        if target_idx < 0:
            return 0.0
        
        max_coupling = 0.0
        for j in range(len(self.spatial_states)):
            if j != target_idx:
                max_coupling = max(max_coupling, abs(R[target_idx, j]))
        
        return max_coupling


def test_basis():
    """Test the correlated basis implementation."""
    print("Testing CorrelatedBasis...")
    
    basis = CorrelatedBasis(n_max=3, l_max=2, N_sigma=64)
    
    print(f"  Number of spatial states: {basis.N_spatial}")
    print(f"  States: {[(s.n, s.l, s.m) for s in basis.spatial_states]}")
    
    # Test radial functions
    print("\n  Radial function normalization:")
    for n in [1, 2, 3]:
        for l in range(n):
            R = basis.radial_function(n, l)
            norm = np.trapz(R**2 * basis.r**2, basis.r)
            print(f"    (n={n}, l={l}): norm = {norm:.6f}")
    
    # Test angular coupling factors
    print("\n  Angular coupling factors (l=0 ↔ l=1):")
    for m in [-1, 0, 1]:
        factor = basis.angular_coupling_factor(0, 0, 1, m)
        print(f"    (0,0) → (1,{m}): {factor:.4f}")
    
    # Test full coupling matrix elements
    print("\n  Full coupling matrix elements:")
    s1 = SpatialState(1, 0, 0)
    for n2 in [1, 2]:
        for m2 in [-1, 0, 1]:
            s2 = SpatialState(n2, 1, m2)
            coupling = basis.compute_full_coupling_matrix_element(s1, s2)
            if abs(coupling) > 1e-10:
                print(f"    (1,0,0) → ({n2},1,{m2}): {coupling:.4f}")
    
    print("\n  Basis test complete!")
    return basis


if __name__ == '__main__':
    test_basis()

