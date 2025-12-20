"""
Non-Separable SFM Solver with Correlated Angular-Subspace Structure.

This solver implements the full non-separable solution where:
- The wavefunction has angular distortions correlated with subspace position σ
- Each angular component (n,l,m) has its own subspace function χ_{nlm}(σ)
- Nonlinear terms require iterative self-consistent solution

The iteration scheme:
1. Build Hamiltonian with current χ components (including nonlinear terms)
2. Solve generalized eigenvalue problem
3. Extract new χ components
4. Mix old/new for stability
5. Repeat until convergence
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import warnings

from sfm_solver.core.correlated_basis import CorrelatedBasis, SpatialState


@dataclass
class NonSeparableResult:
    """Result from the non-separable solver."""
    
    # Subspace wavefunctions for each angular component
    chi_components: Dict[Tuple[int, int, int], NDArray]
    
    # Observables
    A_squared: float
    mass_gev: float
    delta_x: float
    delta_sigma: float
    
    # Angular composition (fraction in each l)
    l_composition: Dict[int, float]
    
    # Target quantum numbers
    n_target: int
    k_winding: int
    
    # Convergence info
    converged: bool
    iterations: int
    final_residual: float
    energy: float


class NonSeparableSolver:
    """
    Iterative solver for the full non-separable SFM problem.
    
    The key physics:
    - Angular distortions correlate with subspace position
    - Coupling Hamiltonian mixes l=0 ↔ l=1 ↔ l=2
    - Nonlinear terms (g₁|χ|⁴) require iteration
    - Self-consistency between amplitude and wavefunction structure
    """
    
    def __init__(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        g1: float,
        g2: float,
        V0: float = 1.0,
        n_max: int = 5,
        l_max: int = 2,
        N_sigma: int = 64,
        a0: float = 1.0,
    ):
        """
        Initialize the solver with SFM parameters.
        
        Args:
            alpha: Spatial-subspace coupling strength (GeV)
            beta: Mass coupling constant (GeV)
            kappa: Curvature coupling (GeV⁻²)
            g1: Nonlinear self-interaction strength
            g2: Circulation coupling (for EM)
            V0: Three-well potential depth (GeV)
            n_max: Maximum spatial principal quantum number
            l_max: Maximum angular momentum
            N_sigma: Subspace grid points
            a0: Characteristic length scale
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.g1 = g1
        self.g2 = g2
        self.V0 = V0
        
        # Create basis
        self.basis = CorrelatedBasis(n_max=n_max, l_max=l_max, N_sigma=N_sigma, a0=a0)
        
        # Three-well potential on subspace grid
        self.V_sigma = V0 * (1 - np.cos(3 * self.basis.sigma))
        
        # Precompute spatial coupling matrix
        self._precompute_spatial_coupling()
    
    def _precompute_spatial_coupling(self):
        """Precompute the spatial coupling matrix elements."""
        N = self.basis.N_spatial
        self.spatial_coupling = np.zeros((N, N), dtype=complex)
        
        for i, s1 in enumerate(self.basis.spatial_states):
            for j, s2 in enumerate(self.basis.spatial_states):
                coupling = self.basis.compute_full_coupling_matrix_element(s1, s2)
                self.spatial_coupling[i, j] = coupling
    
    def _build_subspace_kinetic_matrix(self) -> NDArray:
        """Build the kinetic energy operator -d²/dσ² using spectral method."""
        N = self.basis.N_sigma
        dsigma = self.basis.dsigma
        
        # Spectral derivative operator
        k = np.fft.fftfreq(N, dsigma / (2 * np.pi)) * 2 * np.pi
        
        # Second derivative in Fourier space: -k²
        # But we need the matrix form
        # Use finite differences for simplicity (spectral would require FFT each time)
        
        # Second derivative matrix (periodic boundary)
        D2 = np.zeros((N, N))
        for i in range(N):
            D2[i, i] = -2.0
            D2[i, (i+1) % N] = 1.0
            D2[i, (i-1) % N] = 1.0
        D2 = D2 / dsigma**2
        
        return -D2 / 2  # Kinetic energy: -ℏ²/(2m) d²/dσ², with ℏ=m=1
    
    def _build_subspace_derivative_matrix(self) -> NDArray:
        """Build the first derivative operator d/dσ."""
        N = self.basis.N_sigma
        dsigma = self.basis.dsigma
        
        # Central difference (periodic boundary)
        D1 = np.zeros((N, N), dtype=complex)
        for i in range(N):
            D1[i, (i+1) % N] = 1.0
            D1[i, (i-1) % N] = -1.0
        D1 = D1 / (2 * dsigma)
        
        return D1
    
    def _compute_total_amplitude(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> float:
        """Compute total amplitude squared A² = Σ ∫|χ_{nlm}|² dσ."""
        total = 0.0
        for key, chi in chi_components.items():
            total += np.sum(np.abs(chi)**2) * self.basis.dsigma
        return total
    
    def _normalize_wavefunction(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        target_A_sq: Optional[float] = None,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """Normalize so that total amplitude squared equals target (or 1 if None)."""
        current_A_sq = self._compute_total_amplitude(chi_components)
        
        if current_A_sq < 1e-20:
            return chi_components
        
        if target_A_sq is None:
            target_A_sq = 1.0
        
        scale = np.sqrt(target_A_sq / current_A_sq)
        
        normalized = {}
        for key, chi in chi_components.items():
            normalized[key] = chi * scale
        
        return normalized
    
    def _compute_l_composition(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> Dict[int, float]:
        """Compute fraction of amplitude in each l state."""
        composition = {}
        total = 0.0
        
        for (n, l, m), chi in chi_components.items():
            weight = np.sum(np.abs(chi)**2) * self.basis.dsigma
            composition[l] = composition.get(l, 0.0) + weight
            total += weight
        
        if total > 1e-20:
            for l in composition:
                composition[l] /= total
        
        return composition
    
    def _initial_guess(
        self,
        n_target: int,
        k_winding: int,
        initial_A_sq: float = 0.01,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Create initial guess for χ components.
        
        Start with primarily s-wave (l=0) state with winding k,
        but SEED small l=1 components to bootstrap the coupling!
        
        Without l=1 seeds, the coupling gradient is zero and no l-mixing occurs.
        """
        chi_components = {}
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # Primary s-wave gets 90% of initial amplitude
        primary_fraction = 0.9
        # l=1 components get 10% to bootstrap coupling
        l1_fraction = 0.1
        
        for state in self.basis.spatial_states:
            key = (state.n, state.l, state.m)
            
            # Gaussian envelope with winding for all components
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            chi_base = envelope * np.exp(1j * k_winding * sigma)
            
            if state.n == n_target and state.l == 0 and state.m == 0:
                # Primary s-wave component
                norm_sq = np.sum(np.abs(chi_base)**2) * dsigma
                chi = chi_base * np.sqrt(initial_A_sq * primary_fraction / norm_sq)
                chi_components[key] = chi
                
            elif state.l == 1:
                # SEED l=1 components to bootstrap coupling!
                # These will evolve to their equilibrium values
                # Divide l1_fraction among all l=1 states
                n_l1_states = sum(1 for s in self.basis.spatial_states if s.l == 1)
                per_state_fraction = l1_fraction / max(n_l1_states, 1)
                
                norm_sq = np.sum(np.abs(chi_base)**2) * dsigma
                chi = chi_base * np.sqrt(initial_A_sq * per_state_fraction / norm_sq)
                
                # Add small random phase to break symmetry
                chi = chi * np.exp(1j * 0.1 * state.m)
                chi_components[key] = chi
                
            else:
                # Other components (l=0 for n≠target, l=2) start at zero
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        return chi_components
    
    def _compute_energy(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        delta_x: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the total energy functional.
        
        E = E_kinetic + E_potential + E_nonlinear + E_circulation + E_coupling + E_spatial + E_curvature
        
        CRITICAL: delta_x is a FREE variable, not computed from A²!
        The Compton relation Δx ~ 1/(βA²) should EMERGE from minimization.
        
        CONSISTENT WITH energy_minimizer.py:
        - E_kinetic: (ℏ²/2m)∫|∂χ/∂σ|² dσ
        - E_potential: ∫V(σ)|χ|² dσ
        - E_nonlinear: (g₁/2)∫|χ|⁴ dσ
        - E_circulation: g₂|∫χ*(∂χ/∂σ)|² (for EM)
        - E_spatial: ℏ²/(2βA²Δx²) (localization energy)
        - E_coupling: -α × |R_coupling| × Im[∫χ₁*(∂χ₂/∂σ)dσ]
        - E_curvature: κ × (βA²)² / Δx
        """
        dsigma = self.basis.dsigma
        D1 = self._build_subspace_derivative_matrix()
        
        E_kinetic = 0.0
        E_potential = 0.0
        E_nonlinear = 0.0
        E_circulation = 0.0
        E_coupling = 0.0
        E_spatial = 0.0
        
        # Total amplitude
        total_A_sq = self._compute_total_amplitude(chi_components)
        
        # Compute total chi for circulation term
        chi_total = np.zeros(self.basis.N_sigma, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        
        # === Subspace terms (for each component) ===
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            chi = chi_components.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            
            # Kinetic: (1/2)∫|∂χ/∂σ|² dσ (with ℏ=m=1)
            dchi = D1 @ chi
            E_kinetic += 0.5 * np.real(np.sum(np.conj(dchi) * dchi)) * dsigma
            
            # Potential: ∫V(σ)|χ|² dσ
            E_potential += np.real(np.sum(self.V_sigma * np.abs(chi)**2)) * dsigma
            
            # Nonlinear: (g₁/2)∫|χ|⁴ dσ
            E_nonlinear += (self.g1 / 2) * np.sum(np.abs(chi)**4) * dsigma
        
        # === Circulation energy: g₂|∫χ*(∂χ/∂σ)|² ===
        dchi_total = D1 @ chi_total
        J = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        E_circulation = self.g2 * np.abs(J)**2
        
        # === Spatial energy: ℏ²/(2βA²Δx²) ===
        # Δx is a FREE variable - NOT computed from A²!
        if total_A_sq > 1e-10 and delta_x > 1e-10:
            E_spatial = 1.0 / (2 * self.beta * total_A_sq * delta_x**2)
        else:
            E_spatial = 1e10
        
        # === Coupling energy (between components) ===
        # E_coupling should be NEGATIVE to drive amplitude up!
        for i, s1 in enumerate(self.basis.spatial_states):
            for j, s2 in enumerate(self.basis.spatial_states):
                if i == j:
                    continue
                
                R_coupling = self.spatial_coupling[i, j]
                if abs(R_coupling) < 1e-10:
                    continue
                
                key1 = (s1.n, s1.l, s1.m)
                key2 = (s2.n, s2.l, s2.m)
                chi1 = chi_components.get(key1, np.zeros(self.basis.N_sigma, dtype=complex))
                chi2 = chi_components.get(key2, np.zeros(self.basis.N_sigma, dtype=complex))
                
                dchi2 = D1 @ chi2
                
                # Coupling: -α × |R_coupling| × Im[∫χ₁*(dχ₂/dσ)dσ]
                coupling_integral = np.sum(np.conj(chi1) * dchi2) * dsigma
                E_coupling += -self.alpha * abs(R_coupling) * np.imag(coupling_integral)
        
        # === Curvature energy (gravitational binding): -κ × (βA²)² / Δx ===
        # CRITICAL: This is BINDING energy (negative!) - gravity lowers total energy
        # This competes with E_spatial (positive) to determine equilibrium Δx
        if delta_x > 1e-10:
            mass_sq = (self.beta * total_A_sq) ** 2
            E_curvature = -self.kappa * mass_sq / delta_x  # NEGATIVE binding!
        else:
            E_curvature = -1e10
        
        E_total = E_kinetic + E_potential + E_nonlinear + E_circulation + E_coupling + E_spatial + E_curvature
        
        breakdown = {
            'kinetic': E_kinetic,
            'potential': E_potential,
            'nonlinear': E_nonlinear,
            'circulation': E_circulation,
            'coupling': E_coupling,
            'spatial': E_spatial,
            'curvature': E_curvature,
            'delta_x': delta_x,
            'A_squared': total_A_sq,
            'total': E_total,
        }
        
        return E_total, breakdown
    
    def _compute_gradient(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray],
        delta_x: float,
    ) -> Tuple[Dict[Tuple[int, int, int], NDArray], float]:
        """
        Compute gradient of energy with respect to χ components AND Δx.
        
        Returns:
            gradients: dict of δE/δχ* for each component
            grad_delta_x: dE/dΔx for the spatial variable
        
        CONSISTENT WITH energy_minimizer.py.
        """
        dsigma = self.basis.dsigma
        D1 = self._build_subspace_derivative_matrix()
        D2 = self._build_subspace_kinetic_matrix() * (-2)  # d²/dσ²
        
        # Total amplitude
        total_A_sq = self._compute_total_amplitude(chi_components)
        
        # Compute total chi for circulation gradient
        chi_total = np.zeros(self.basis.N_sigma, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        dchi_total = D1 @ chi_total
        J = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        
        gradients = {}
        
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            chi = chi_components.get(key, np.zeros(self.basis.N_sigma, dtype=complex))
            
            # Gradient from subspace terms
            # δE_kinetic/δχ* = (1/2)(-d²χ/dσ²) = -(1/2)D2@chi
            grad = -0.5 * D2 @ chi
            
            # δE_potential/δχ* = V(σ)χ
            grad += self.V_sigma * chi
            
            # δE_nonlinear/δχ* = g₁|χ|²χ
            grad += self.g1 * np.abs(chi)**2 * chi
            
            # δE_circulation/δχ* = 2g₂Re[J*]∂χ/∂σ (contribution from this component)
            dchi = D1 @ chi
            grad += 2 * self.g2 * np.real(np.conj(J)) * dchi
            
            # δE_spatial/δχ* = -1/(βA⁴Δx²) × χ
            if total_A_sq > 1e-10 and delta_x > 1e-10:
                grad_spatial = -1.0 / (self.beta * total_A_sq**2 * delta_x**2) * chi
            else:
                grad_spatial = np.zeros_like(chi)
            grad += grad_spatial
            
            # δE_curvature/δχ* = -4κβ²A² / Δx × χ (negative because binding!)
            if delta_x > 1e-10:
                grad_curv = -4 * self.kappa * self.beta**2 * total_A_sq / delta_x * chi
            else:
                grad_curv = np.zeros_like(chi)
            grad += grad_curv
            
            gradients[key] = grad
        
        # Gradient from coupling terms
        for i, s1 in enumerate(self.basis.spatial_states):
            for j, s2 in enumerate(self.basis.spatial_states):
                if i == j:
                    continue
                
                R_coupling = self.spatial_coupling[i, j]
                if abs(R_coupling) < 1e-10:
                    continue
                
                key1 = (s1.n, s1.l, s1.m)
                key2 = (s2.n, s2.l, s2.m)
                chi2 = chi_components.get(key2, np.zeros(self.basis.N_sigma, dtype=complex))
                
                dchi2 = D1 @ chi2
                gradients[key1] = gradients[key1] - self.alpha * abs(R_coupling) * (-0.5j) * dchi2
                
                chi1 = chi_components.get(key1, np.zeros(self.basis.N_sigma, dtype=complex))
                dchi1 = D1 @ chi1
                gradients[key2] = gradients[key2] + self.alpha * abs(R_coupling) * (0.5j) * dchi1
        
        # === Gradient with respect to Δx ===
        # dE_spatial/dΔx = -1/(βA²Δx³)  (wants larger Δx - quantum pressure)
        # dE_curvature/dΔx = +κ(βA²)²/Δx²  (wants smaller Δx - gravity binding!)
        # Balance: quantum pressure vs gravitational compression
        if total_A_sq > 1e-10 and delta_x > 1e-10:
            grad_delta_x = -1.0 / (self.beta * total_A_sq * delta_x**3)  # E_spatial gradient
            grad_delta_x += self.kappa * (self.beta * total_A_sq)**2 / (delta_x**2)  # E_curvature gradient (+ because binding is negative)
        else:
            grad_delta_x = 0.0
        
        return gradients, grad_delta_x
    
    def _compute_residual(
        self,
        chi_old: Dict[Tuple[int, int, int], NDArray],
        chi_new: Dict[Tuple[int, int, int], NDArray],
    ) -> float:
        """Compute change between iterations."""
        total = 0.0
        norm = 0.0
        
        for key in chi_old:
            diff = chi_new.get(key, np.zeros_like(chi_old[key])) - chi_old[key]
            total += np.sum(np.abs(diff)**2) * self.basis.dsigma
            norm += np.sum(np.abs(chi_old[key])**2) * self.basis.dsigma
        
        if norm < 1e-20:
            return 1e10
        
        return np.sqrt(total / norm)
    
    def _compute_delta_sigma(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> float:
        """Compute subspace width from wavefunction."""
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        
        # Compute total |χ|² weighted by σ
        total_weight = 0.0
        mean_sigma = 0.0
        
        for chi in chi_components.values():
            weight = np.abs(chi)**2
            total_weight += np.sum(weight) * dsigma
            mean_sigma += np.sum(sigma * weight) * dsigma
        
        if total_weight < 1e-20:
            return 0.5
        
        mean_sigma /= total_weight
        
        # Variance
        var = 0.0
        for chi in chi_components.values():
            weight = np.abs(chi)**2
            var += np.sum((sigma - mean_sigma)**2 * weight) * dsigma
        var /= total_weight
        
        return np.sqrt(max(var, 1e-10))
    
    def solve(
        self,
        n_target: int,
        k_winding: int = 1,
        max_iter: int = 5000,
        tol: float = 1e-8,
        learning_rate: float = 0.0001,
        initial_A_sq: float = 0.001,
        initial_delta_x: float = 1.0,
        verbose: bool = False,
    ) -> NonSeparableResult:
        """
        Solve for the particle state with quantum numbers (n_target, k_winding).
        
        Uses gradient descent on the energy functional. BOTH A² and Δx are
        FREE variables that emerge from minimizing the energy.
        
        CRITICAL: The Compton relation Δx ~ 1/(βA²) should EMERGE from
        minimization, not be imposed!
        
        Args:
            n_target: Target spatial quantum number (1, 2, 3 for e, μ, τ)
            k_winding: Subspace winding number (1 for leptons)
            max_iter: Maximum iterations
            tol: Convergence tolerance on energy change
            learning_rate: Step size for gradient descent
            initial_A_sq: Initial guess for amplitude squared
            initial_delta_x: Initial guess for spatial extent (FREE variable!)
            verbose: Print progress
        
        Returns:
            NonSeparableResult with converged wavefunction and observables
        """
        if verbose:
            print(f"Solving for n={n_target}, k={k_winding}")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}, "
                  f"g₁={self.g1:.4g}, V₀={self.V0:.4g}")
        
        # Initial guess - BOTH chi and delta_x are free variables!
        chi_components = self._initial_guess(n_target, k_winding, initial_A_sq)
        delta_x = initial_delta_x  # FREE variable that will be optimized!
        
        # Initial energy
        energy, breakdown = self._compute_energy(chi_components, delta_x)
        
        converged = False
        final_residual = 1e10
        best_energy = energy
        best_chi = {k: v.copy() for k, v in chi_components.items()}
        best_delta_x = delta_x
        
        # Adaptive learning rates for wavefunction and delta_x
        lr = learning_rate
        lr_dx = learning_rate * 100  # Delta_x can have different scale
        
        for iteration in range(max_iter):
            # Compute gradients for BOTH chi and delta_x
            gradients, grad_delta_x = self._compute_gradient(chi_components, delta_x)
            
            # Gradient descent step for wavefunction
            chi_new = {}
            for key, chi in chi_components.items():
                grad = gradients.get(key, np.zeros_like(chi))
                chi_new[key] = chi - lr * grad
            
            # Gradient descent step for delta_x (keep positive!)
            delta_x_new = delta_x - lr_dx * grad_delta_x
            delta_x_new = max(delta_x_new, 0.001)  # Ensure positive
            
            # Compute new energy
            new_energy, new_breakdown = self._compute_energy(chi_new, delta_x_new)
            
            # Adaptive learning rate
            if new_energy < energy:
                # Accept step
                chi_components = chi_new
                delta_x = delta_x_new
                energy_change = abs(new_energy - energy)
                energy = new_energy
                
                # Increase learning rate slightly
                lr = min(lr * 1.01, learning_rate * 10)
                lr_dx = min(lr_dx * 1.01, learning_rate * 1000)
                
                # Track best
                if energy < best_energy:
                    best_energy = energy
                    best_chi = {k: v.copy() for k, v in chi_components.items()}
                    best_delta_x = delta_x
            else:
                # Reject step, reduce learning rate
                lr = lr * 0.5
                lr_dx = lr_dx * 0.5
                if lr < learning_rate * 1e-6:
                    if verbose:
                        print(f"  Iter {iteration}: Learning rate collapsed")
                    break
                energy_change = 0
            
            # Check for amplitude collapse
            A_sq = self._compute_total_amplitude(chi_components)
            if A_sq < 1e-30:
                if verbose:
                    print(f"  Iter {iteration}: Amplitude collapsed")
                break
            
            # Logging - show both A² and Δx evolution
            if verbose and iteration % 2000 == 0:
                l_comp = self._compute_l_composition(chi_components)
                compton_check = delta_x * self.beta * A_sq  # Should → 1 if Compton emerges
                print(f"  Iter {iteration}: E={energy:.6f}, A²={A_sq:.6e}, Δx={delta_x:.4f}, "
                      f"l=0: {l_comp.get(0,0)*100:.1f}%, lr={lr:.2e}")
                print(f"    E_coupling={new_breakdown['coupling']:.6f}, "
                      f"E_spatial={new_breakdown['spatial']:.6f}, "
                      f"Δx×βA²={compton_check:.4f} (→1 if Compton)")
            
            # Check convergence
            final_residual = energy_change
            if energy_change < tol and energy_change > 0:
                converged = True
                if verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break
        
        # Use best result
        chi_components = best_chi
        delta_x = best_delta_x
        energy, breakdown = self._compute_energy(chi_components, delta_x)
        
        # Final observables
        A_sq = self._compute_total_amplitude(chi_components)
        mass_gev = self.beta * A_sq
        l_composition = self._compute_l_composition(chi_components)
        delta_sigma = self._compute_delta_sigma(chi_components)
        
        # Check if Compton relation emerged from minimization
        compton_check = delta_x * self.beta * A_sq  # Should be ~1 if Compton emerges
        
        if verbose:
            print(f"  Final: A²={A_sq:.4e}, Δx={delta_x:.4f}, mass={mass_gev:.4e} GeV")
            print(f"  Compton check: Δx×βA² = {compton_check:.4f} (should be ~1)")
            print(f"  Energy breakdown: {breakdown}")
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
        
        return NonSeparableResult(
            chi_components=chi_components,
            A_squared=A_sq,
            mass_gev=mass_gev,
            delta_x=delta_x,
            delta_sigma=delta_sigma,
            l_composition=l_composition,
            n_target=n_target,
            k_winding=k_winding,
            converged=converged,
            iterations=iteration + 1,
            final_residual=final_residual,
            energy=energy,
        )


    def solve_perturbative(
        self,
        n_target: int,
        k_winding: int = 1,
        max_iter: int = 200,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> NonSeparableResult:
        """
        Solve using constrained perturbation theory.
        
        FIXED: Now uses IMAGINARY part of subspace integral for winding!
        
        Key physics:
        1. The PRIMARY component (n_target, l=0) amplitude is determined by
           the balance: coupling ~ g₁ × A³
        2. SECONDARY components (l=1, etc.) are induced perturbatively
        3. Effective coupling depends on resonance (energy denominators)
        4. Winding k creates Im[∫χ*(∂χ/∂σ)dσ] = k × A² (not real part!)
        """
        if verbose:
            print(f"Solving (perturbative) for n={n_target}, k={k_winding}")
            print(f"  Parameters: α={self.alpha:.4g}, β={self.beta:.4g}, g₁={self.g1:.4g}")
        
        D1 = self._build_subspace_derivative_matrix()
        dsigma = self.basis.dsigma
        sigma = self.basis.sigma
        
        target_key = (n_target, 0, 0)
        target_idx = self.basis.state_index(n_target, 0, 0)
        
        # Energy scale for target state
        E_target_0 = n_target ** 2
        
        # === STEP 1: Compute effective coupling strength for this n ===
        # Now accounting for the IMAGINARY part of the coupling integral
        
        effective_coupling = 0.0
        
        for i, state in enumerate(self.basis.spatial_states):
            if state.n == n_target and state.l == 0:
                continue  # Skip same state
            
            # Coupling matrix element from target
            R_coupling = self.spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                continue
            
            # Energy denominator
            E_state = state.n ** 2 + state.l * (state.l + 1) / 2
            E_denom = E_target_0 - E_state
            
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            if abs(E_denom) < 1e-10:
                continue
            
            # Contribution to effective coupling
            # Use MAGNITUDE of R_coupling (spatial coupling strength)
            # The sign comes from the subspace wavefunction winding
            effective_coupling += (abs(R_coupling) / abs(E_denom)) ** 2
        
        effective_coupling = np.sqrt(effective_coupling)
        
        # The subspace winding factor: for χ ~ exp(ikσ), Im[∫χ*(∂χ/∂σ)dσ] = k
        # This multiplies the effective coupling!
        winding_factor = k_winding
        effective_coupling_with_winding = effective_coupling * winding_factor
        
        if verbose:
            print(f"  Spatial coupling (effective): {effective_coupling:.6f}")
            print(f"  Winding factor: {winding_factor}")
            print(f"  Total effective coupling: {effective_coupling_with_winding:.6f}")
        
        # === STEP 2: Compute equilibrium amplitude from energy balance ===
        # E_coupling ≈ -α × eff_coupling × k × A²
        # E_nonlinear ≈ g₁/2 × A⁴
        # At equilibrium: α × eff_coupling × k × A = g₁ × A³
        # → A² = α × eff_coupling × k / g₁
        
        if effective_coupling_with_winding > 1e-10 and self.g1 > 0:
            A_sq = self.alpha * effective_coupling_with_winding / self.g1
        else:
            A_sq = 1e-10
        
        if verbose:
            print(f"  Equilibrium A² from balance: {A_sq:.6e}")
        
        # === STEP 3: Build the wavefunction at equilibrium ===
        
        # Primary component with winding k
        envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
        chi_primary = envelope * np.exp(1j * k_winding * sigma)
        
        # Normalize primary to have most of the amplitude
        primary_fraction = 0.9  # 90% in primary
        chi_primary = chi_primary * np.sqrt(A_sq * primary_fraction / 
                                            (np.sum(np.abs(chi_primary)**2) * dsigma))
        
        chi_components = {target_key: chi_primary}
        
        # Verify subspace factor (should be k × amplitude)
        dchi_primary = D1 @ chi_primary
        subspace_integral = np.sum(np.conj(chi_primary) * dchi_primary) * dsigma
        subspace_factor = np.imag(subspace_integral)
        
        if verbose:
            print(f"  Subspace factor Im[∫χ*(∂χ/∂σ)dσ]: {subspace_factor:.6f}")
        
        # Induced components from perturbation theory
        remaining_A_sq = A_sq * (1 - primary_fraction)
        induced_total = 0.0
        
        induced_components = {}
        for i, state in enumerate(self.basis.spatial_states):
            key = (state.n, state.l, state.m)
            if key == target_key:
                continue
            
            R_coupling = self.spatial_coupling[target_idx, i]
            if abs(R_coupling) < 1e-10:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
                continue
            
            E_state = state.n ** 2 + state.l * (state.l + 1) / 2
            E_denom = E_target_0 - E_state
            
            if abs(E_denom) < 0.5:
                E_denom = 0.5 * np.sign(E_denom) if E_denom != 0 else 0.5
            
            # Perturbative induced component
            # Use magnitude of R_coupling (coupling strength)
            # The gradient of χ_primary (has factor of ik) provides the phase
            induced = -self.alpha * abs(R_coupling) * dchi_primary / E_denom
            induced_components[key] = induced
            induced_total += np.sum(np.abs(induced)**2) * dsigma
        
        # Normalize induced components to remaining amplitude
        if induced_total > 1e-20:
            scale = np.sqrt(remaining_A_sq / induced_total)
            for key, induced in induced_components.items():
                chi_components[key] = induced * scale
        else:
            for key in induced_components:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        # Final amplitude check
        A_sq_final = self._compute_total_amplitude(chi_components)
        
        # Compute energy with fixed functional
        energy, breakdown = self._compute_energy(chi_components)
        mass_gev = self.beta * A_sq_final
        l_composition = self._compute_l_composition(chi_components)
        delta_sigma = self._compute_delta_sigma(chi_components)
        delta_x = 1.0 / (self.beta * A_sq_final) if A_sq_final > 1e-10 else 1.0
        
        if verbose:
            print(f"  Final A²: {A_sq_final:.6e}, mass: {mass_gev:.6e} GeV")
            print(f"  E_coupling: {breakdown.get('coupling', 0):.6e}")
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
        
        return NonSeparableResult(
            chi_components=chi_components,
            A_squared=A_sq_final,
            mass_gev=mass_gev,
            delta_x=delta_x,
            delta_sigma=delta_sigma,
            l_composition=l_composition,
            n_target=n_target,
            k_winding=k_winding,
            converged=True,  # Analytical solution
            iterations=1,
            final_residual=0.0,
            energy=energy,
        )


def test_solver():
    """Test the non-separable solver on all three leptons."""
    print("="*60)
    print("TESTING NON-SEPARABLE SFM SOLVER")
    print("="*60)
    
    # Experimental lepton masses for comparison
    ELECTRON_MASS = 0.000511  # GeV
    MUON_MASS = 0.1057       # GeV
    TAU_MASS = 1.777         # GeV
    
    # Parameters - tuned to give correct mass ratios
    # Key physics:
    # - α determines coupling strength
    # - g₁ provides nonlinear stabilization
    # - The mass hierarchy comes from energy denominators in perturbation theory
    solver = NonSeparableSolver(
        alpha=20.0,     # Coupling strength (GeV) - increased for stronger effect
        beta=100.0,     # Mass coupling (GeV)
        kappa=0.0001,   # Curvature coupling (GeV⁻²)
        g1=5000.0,      # Nonlinear coupling (provides stability)
        g2=0.004,       # Circulation coupling
        V0=1.0,         # Three-well depth (GeV)
        n_max=5,
        l_max=2,
        N_sigma=48,
        a0=1.0,         # Length scale
    )
    
    print(f"\nParameters:")
    print(f"  α = {solver.alpha} GeV (coupling strength)")
    print(f"  β = {solver.beta} GeV (mass coupling)")
    print(f"  g₁ = {solver.g1} (nonlinear)")
    print(f"  V₀ = {solver.V0} GeV (three-well)")
    
    print(f"\nBasis: {solver.basis.N_spatial} spatial states × {solver.basis.N_sigma} σ points")
    
    # Test spatial coupling matrix
    print(f"\nSpatial coupling matrix (sample):")
    for i in range(min(5, solver.basis.N_spatial)):
        for j in range(min(5, solver.basis.N_spatial)):
            if abs(solver.spatial_coupling[i,j]) > 1e-6:
                s1, s2 = solver.basis.spatial_states[i], solver.basis.spatial_states[j]
                print(f"  ({s1.n},{s1.l},{s1.m}) → ({s2.n},{s2.l},{s2.m}): {solver.spatial_coupling[i,j]:.4f}")
    
    results = {}
    
    # Test on all three leptons using GRADIENT DESCENT solver
    # BOTH A² and Δx are FREE variables that emerge from minimization!
    for n_target, name, exp_mass in [(1, 'Electron', ELECTRON_MASS), 
                                       (2, 'Muon', MUON_MASS),
                                       (3, 'Tau', TAU_MASS)]:
        print(f"\n{'='*60}")
        print(f"Solving for {name} (n={n_target})")
        print(f"  Experimental mass: {exp_mass:.6f} GeV")
        print("-"*60)
        
        # CRITICAL: Both A² and Δx are free optimization variables!
        # The Compton relation Δx ~ 1/(βA²) should EMERGE from minimization.
        result = solver.solve(
            n_target=n_target, 
            k_winding=1, 
            verbose=True,
            max_iter=50000,
            learning_rate=0.00001,
            initial_A_sq=0.001,  # Small initial amplitude
            initial_delta_x=10.0,  # Start with larger Δx
        )
        results[name] = result
        
        print(f"\n  Predicted mass: {result.mass_gev:.6e} GeV")
        print(f"  Converged: {result.converged}, iterations: {result.iterations}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Particle':<12} {'n':<4} {'Predicted (GeV)':<18} {'Experimental (GeV)':<18} {'Ratio':<10}")
    print("-"*70)
    
    exp_masses = {'Electron': ELECTRON_MASS, 'Muon': MUON_MASS, 'Tau': TAU_MASS}
    
    for name in ['Electron', 'Muon', 'Tau']:
        result = results[name]
        exp = exp_masses[name]
        ratio = result.mass_gev / exp if exp > 0 else float('inf')
        print(f"{name:<12} {result.n_target:<4} {result.mass_gev:<18.6e} {exp:<18.6f} {ratio:<10.2f}")
    
    # Mass ratios
    print(f"\n{'Ratio':<25} {'Predicted':<15} {'Experimental':<15} {'Error %':<10}")
    print("-"*65)
    
    m_e = results['Electron'].mass_gev
    m_mu = results['Muon'].mass_gev
    m_tau = results['Tau'].mass_gev
    
    ratio_mu_e = m_mu / m_e if m_e > 1e-30 else float('inf')
    ratio_tau_e = m_tau / m_e if m_e > 1e-30 else float('inf')
    
    exp_ratio_mu_e = MUON_MASS / ELECTRON_MASS
    exp_ratio_tau_e = TAU_MASS / ELECTRON_MASS
    
    error_mu_e = abs(ratio_mu_e - exp_ratio_mu_e) / exp_ratio_mu_e * 100 if exp_ratio_mu_e > 0 else float('inf')
    error_tau_e = abs(ratio_tau_e - exp_ratio_tau_e) / exp_ratio_tau_e * 100 if exp_ratio_tau_e > 0 else float('inf')
    
    print(f"{'m_μ / m_e':<25} {ratio_mu_e:<15.2f} {exp_ratio_mu_e:<15.2f} {error_mu_e:<10.1f}%")
    print(f"{'m_τ / m_e':<25} {ratio_tau_e:<15.2f} {exp_ratio_tau_e:<15.2f} {error_tau_e:<10.1f}%")
    
    # Angular composition summary
    print(f"\nAngular Composition:")
    print(f"{'Particle':<12} {'l=0 (%)':<12} {'l=1 (%)':<12} {'l=2 (%)':<12}")
    print("-"*50)
    for name in ['Electron', 'Muon', 'Tau']:
        result = results[name]
        l0 = result.l_composition.get(0, 0) * 100
        l1 = result.l_composition.get(1, 0) * 100
        l2 = result.l_composition.get(2, 0) * 100
        print(f"{name:<12} {l0:<12.1f} {l1:<12.1f} {l2:<12.1f}")
    
    return results


if __name__ == '__main__':
    test_solver()

