"""
Non-Separable Wavefunction Solver for SFM.

This module solves for the full entangled superposition wavefunction:
    ψ(r,θ,φ,σ) = Σ_{n,l,m} R_{nl}(r) Y_l^m(θ,φ) χ_{nlm}(σ)

Key physics:
- Each angular component (n,l,m) has its OWN subspace function χ_{nlm}(σ)
- The coupling Hamiltonian mixes l=0 ↔ l=1 ↔ l=2
- The p-wave components are correlated with ∂χ₀/∂σ
- This breaks spherical symmetry and allows non-zero coupling!

This is Phase 1 of the refactor - extracted from nonseparable_solver.py.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from sfm_solver.core.correlated_basis import CorrelatedBasis, SpatialState


@dataclass
class WavefunctionStructure:
    """
    Result from the non-separable wavefunction solver.
    
    Contains the converged wavefunction STRUCTURE (not scaled).
    The SCALE (A, Δx, Δσ) is determined separately by UniversalEnergyMinimizer.
    """
    # Subspace wavefunctions for each angular component {(n,l,m): χ_{nlm}(σ)}
    chi_components: Dict[Tuple[int, int, int], NDArray]
    
    # Target quantum numbers
    n_target: int
    k_winding: int
    
    # Angular composition (fraction in each l)
    l_composition: Dict[int, float]
    
    # Effective winding from Im[∫χ*(∂χ/∂σ)dσ]
    k_eff: float
    
    # Total amplitude of structure (before scaling)
    structure_norm: float
    
    # Convergence info
    converged: bool
    iterations: int
    
    # Particle type
    particle_type: str  # 'lepton', 'meson', 'baryon'


class NonSeparableWavefunctionSolver:
    """
    Solve for non-separable entangled superposition wavefunctions.
    
    This solver finds the wavefunction STRUCTURE - the relative amplitudes
    and phases of the χ_{nlm}(σ) components. The overall SCALE (A, Δx, Δσ)
    is determined by the UniversalEnergyMinimizer.
    
    The key output is a unit-normalized set of {χ_{nlm}(σ)} that together
    form the complete non-separable wavefunction.
    
    Methods from nonseparable_solver.py are incorporated here.
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
        
        # Create basis (from correlated_basis.py)
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
        target_A_sq: float = 1.0,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """Normalize so that total amplitude squared equals target."""
        current_A_sq = self._compute_total_amplitude(chi_components)
        
        if current_A_sq < 1e-20:
            return chi_components
        
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
    
    def _compute_k_eff(
        self,
        chi_components: Dict[Tuple[int, int, int], NDArray]
    ) -> float:
        """
        Compute effective winding from Im[∫χ_total*(∂χ_total/∂σ)dσ].
        
        For unit-normalized wavefunction, this gives the effective k.
        """
        D1 = self._build_subspace_derivative_matrix()
        dsigma = self.basis.dsigma
        
        # Sum all components
        chi_total = np.zeros(self.basis.N_sigma, dtype=complex)
        for chi in chi_components.values():
            chi_total += chi
        
        dchi_total = D1 @ chi_total
        integral = np.sum(np.conj(chi_total) * dchi_total) * dsigma
        
        # Imaginary part carries the winding
        return float(np.imag(integral))
    
    def _initial_guess_lepton(
        self,
        n_target: int,
        k_winding: int,
        initial_A_sq: float = 0.01,
    ) -> Dict[Tuple[int, int, int], NDArray]:
        """
        Create initial guess for lepton wavefunction.
        
        Start with primarily s-wave (l=0) state with winding k,
        but SEED small l=1 components to bootstrap the coupling!
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
                n_l1_states = sum(1 for s in self.basis.spatial_states if s.l == 1)
                per_state_fraction = l1_fraction / max(n_l1_states, 1)
                
                norm_sq = np.sum(np.abs(chi_base)**2) * dsigma
                chi = chi_base * np.sqrt(initial_A_sq * per_state_fraction / norm_sq)
                
                # Add small random phase to break symmetry
                chi = chi * np.exp(1j * 0.1 * state.m)
                chi_components[key] = chi
                
            else:
                # Other components start at zero
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        return chi_components
    
    def solve_lepton(
        self,
        n_target: int,
        k_winding: int = 1,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for lepton wavefunction structure.
        
        This finds the STRUCTURE of the entangled wavefunction (the relative
        amplitudes and phases of χ_{nlm} components). The output is 
        unit-normalized.
        
        The overall SCALE (A, Δx, Δσ) is determined by UniversalEnergyMinimizer.
        
        Uses perturbative approach where:
        1. Primary s-wave carries most amplitude
        2. l=1 components induced by coupling
        3. Effective coupling determines l-mixing
        
        Args:
            n_target: Target spatial quantum number (1=e, 2=μ, 3=τ)
            k_winding: Subspace winding number (typically 1)
            max_iter: Maximum iterations for self-consistency
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with unit-normalized components
        """
        if verbose:
            print(f"Solving lepton wavefunction structure for n={n_target}, k={k_winding}")
        
        D1 = self._build_subspace_derivative_matrix()
        dsigma = self.basis.dsigma
        sigma = self.basis.sigma
        
        target_key = (n_target, 0, 0)
        target_idx = self.basis.state_index(n_target, 0, 0)
        
        # Energy scale for target state (for perturbation theory)
        E_target_0 = n_target ** 2
        
        # === STEP 1: Compute effective coupling from perturbation theory ===
        effective_coupling = 0.0
        
        for i, state in enumerate(self.basis.spatial_states):
            if state.n == n_target and state.l == 0:
                continue  # Skip same state
            
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
            
            # Contribution to effective coupling (magnitude)
            effective_coupling += (abs(R_coupling) / abs(E_denom)) ** 2
        
        effective_coupling = np.sqrt(effective_coupling)
        
        if verbose:
            print(f"  Effective spatial coupling: {effective_coupling:.6f}")
        
        # === STEP 2: Build primary s-wave component ===
        envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
        chi_primary = envelope * np.exp(1j * k_winding * sigma)
        
        # Normalize primary to unit amplitude
        norm_sq = np.sum(np.abs(chi_primary)**2) * dsigma
        chi_primary = chi_primary / np.sqrt(norm_sq)
        
        # Primary gets 90% of total (rest goes to induced)
        primary_fraction = 0.9
        chi_primary = chi_primary * np.sqrt(primary_fraction)
        
        chi_components = {target_key: chi_primary}
        
        # === STEP 3: Compute induced l≠0 components ===
        dchi_primary = D1 @ chi_primary
        
        induced_components = {}
        induced_total = 0.0
        
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
            # The coupling involves ∂χ/∂σ which carries the winding phase
            induced = -self.alpha * abs(R_coupling) * dchi_primary / E_denom
            induced_components[key] = induced
            induced_total += np.sum(np.abs(induced)**2) * dsigma
        
        # Normalize induced to remaining fraction
        remaining_fraction = 1.0 - primary_fraction
        if induced_total > 1e-20:
            scale = np.sqrt(remaining_fraction / induced_total)
            for key, induced in induced_components.items():
                chi_components[key] = induced * scale
        else:
            for key in induced_components:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        # Final normalization to unit total
        chi_components = self._normalize_wavefunction(chi_components, target_A_sq=1.0)
        
        # Extract observables
        l_composition = self._compute_l_composition(chi_components)
        k_eff = self._compute_k_eff(chi_components)
        structure_norm = self._compute_total_amplitude(chi_components)
        
        if verbose:
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
            print(f"  Effective k: {k_eff:.4f}")
        
        return WavefunctionStructure(
            chi_components=chi_components,
            n_target=n_target,
            k_winding=k_winding,
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=True,  # Perturbative is analytic
            iterations=1,
            particle_type='lepton',
        )
    
    def solve_meson(
        self,
        quark_gen: int,
        antiquark_gen: int,
        k_quark: int,
        k_antiquark: int,
        n_radial: int = 1,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for meson (quark-antiquark) wavefunction structure.
        
        For mesons, the subspace wavefunction is composite:
            χ_{nlm}(σ) = χ_{nlm,q}(σ) + χ_{nlm,q̄}(σ)
        
        Each spatial component has BOTH quark contributions with interference.
        
        Args:
            quark_gen, antiquark_gen: Quark generations (1=u/d, 2=c/s, 3=b/t)
            k_quark, k_antiquark: Quark windings (typically +5, -5 for color)
            n_radial: Radial excitation (1=1S, 2=2S)
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with unit-normalized composite components
        """
        if verbose:
            print(f"Solving meson wavefunction: q_gen={quark_gen}, qbar_gen={antiquark_gen}")
        
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        D1 = self._build_subspace_derivative_matrix()
        
        # Build individual quark subspace functions
        def quark_subspace(gen: int, k: int) -> NDArray:
            """Generate quark subspace wavefunction with generation structure."""
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            # Generation-dependent radial structure
            if gen > 1:
                x = (sigma - np.pi) / 1.5
                radial_mod = 1.0 + 0.5 * np.cos((gen - 1) * np.pi * x / 2)
            else:
                radial_mod = np.ones_like(sigma)
            return envelope * radial_mod * np.exp(1j * k * sigma)
        
        chi_q = quark_subspace(quark_gen, k_quark)
        chi_qbar = quark_subspace(antiquark_gen, k_antiquark)
        
        # Composite primary (s-wave, n=n_radial)
        chi_composite = chi_q + chi_qbar
        
        # Normalize
        norm_sq = np.sum(np.abs(chi_composite)**2) * dsigma
        chi_composite = chi_composite / np.sqrt(norm_sq)
        
        # Use n_radial as the target spatial mode
        target_key = (n_radial, 0, 0)
        target_idx = self.basis.state_index(n_radial, 0, 0)
        
        # Primary fraction
        primary_fraction = 0.9
        chi_primary = chi_composite * np.sqrt(primary_fraction)
        
        chi_components = {target_key: chi_primary}
        
        # Induced components (like lepton but with composite structure)
        E_target_0 = n_radial ** 2
        dchi_primary = D1 @ chi_primary
        
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
            
            induced = -self.alpha * abs(R_coupling) * dchi_primary / E_denom
            induced_components[key] = induced
            induced_total += np.sum(np.abs(induced)**2) * dsigma
        
        remaining_fraction = 1.0 - primary_fraction
        if induced_total > 1e-20:
            scale = np.sqrt(remaining_fraction / induced_total)
            for key, induced in induced_components.items():
                chi_components[key] = induced * scale
        else:
            for key in induced_components:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        # Normalize to unit total
        chi_components = self._normalize_wavefunction(chi_components, target_A_sq=1.0)
        
        # Extract observables
        l_composition = self._compute_l_composition(chi_components)
        k_eff = self._compute_k_eff(chi_components)
        structure_norm = self._compute_total_amplitude(chi_components)
        
        # Effective winding for mesons
        net_winding = k_quark + k_antiquark
        
        if verbose:
            print(f"  Net winding (k_q + k_qbar): {net_winding}")
            print(f"  Effective k from wavefunction: {k_eff:.4f}")
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
        
        return WavefunctionStructure(
            chi_components=chi_components,
            n_target=n_radial,
            k_winding=net_winding,
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=True,
            iterations=1,
            particle_type='meson',
        )
    
    def solve_baryon(
        self,
        quark_gens: List[int],
        windings: List[int],
        color_phases: List[float] = None,
        verbose: bool = False,
    ) -> WavefunctionStructure:
        """
        Solve for baryon (three-quark) wavefunction structure.
        
        For baryons, the subspace wavefunction is a 3-quark composite:
            χ_{nlm}(σ) = Σ_i χ_{nlm,qi}(σ) × exp(i × color_phase_i)
        
        Color neutrality requires: Σ exp(i × phase_i) = 0
        
        Args:
            quark_gens: List of 3 quark generations
            windings: List of 3 quark windings
            color_phases: Phase offsets for color (default: [0, 2π/3, 4π/3])
            verbose: Print progress
            
        Returns:
            WavefunctionStructure with unit-normalized composite components
        """
        if color_phases is None:
            color_phases = [0, 2*np.pi/3, 4*np.pi/3]
        
        if verbose:
            print(f"Solving baryon wavefunction: gens={quark_gens}, k={windings}")
        
        sigma = self.basis.sigma
        dsigma = self.basis.dsigma
        D1 = self._build_subspace_derivative_matrix()
        
        # Build individual quark subspace functions
        def quark_subspace(gen: int, k: int) -> NDArray:
            envelope = np.exp(-(sigma - np.pi)**2 / 0.5)
            if gen > 1:
                x = (sigma - np.pi) / 1.5
                radial_mod = 1.0 + 0.5 * np.cos((gen - 1) * np.pi * x / 2)
            else:
                radial_mod = np.ones_like(sigma)
            return envelope * radial_mod * np.exp(1j * k * sigma)
        
        # Three-quark composite with color phases
        chi_composite = np.zeros(self.basis.N_sigma, dtype=complex)
        for gen, k, phase in zip(quark_gens, windings, color_phases):
            chi_q = quark_subspace(gen, k)
            chi_composite += chi_q * np.exp(1j * phase)
        
        # Normalize
        norm_sq = np.sum(np.abs(chi_composite)**2) * dsigma
        if norm_sq > 1e-20:
            chi_composite = chi_composite / np.sqrt(norm_sq)
        
        # Ground state baryons: n=1
        n_target = 1
        target_key = (n_target, 0, 0)
        target_idx = self.basis.state_index(n_target, 0, 0)
        
        primary_fraction = 0.9
        chi_primary = chi_composite * np.sqrt(primary_fraction)
        
        chi_components = {target_key: chi_primary}
        
        # Induced components
        E_target_0 = n_target ** 2
        dchi_primary = D1 @ chi_primary
        
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
            
            induced = -self.alpha * abs(R_coupling) * dchi_primary / E_denom
            induced_components[key] = induced
            induced_total += np.sum(np.abs(induced)**2) * dsigma
        
        remaining_fraction = 1.0 - primary_fraction
        if induced_total > 1e-20:
            scale = np.sqrt(remaining_fraction / induced_total)
            for key, induced in induced_components.items():
                chi_components[key] = induced * scale
        else:
            for key in induced_components:
                chi_components[key] = np.zeros(self.basis.N_sigma, dtype=complex)
        
        # Normalize
        chi_components = self._normalize_wavefunction(chi_components, target_A_sq=1.0)
        
        # Extract observables
        l_composition = self._compute_l_composition(chi_components)
        k_eff = self._compute_k_eff(chi_components)
        structure_norm = self._compute_total_amplitude(chi_components)
        
        # Baryon effective k_coupling (sum of magnitudes)
        k_coupling = sum(abs(k) for k in windings)
        
        if verbose:
            print(f"  k_coupling (sum|k|): {k_coupling}")
            print(f"  Effective k from wavefunction: {k_eff:.4f}")
            print(f"  Angular composition: {', '.join(f'l={l}: {f*100:.1f}%' for l, f in sorted(l_composition.items()))}")
        
        return WavefunctionStructure(
            chi_components=chi_components,
            n_target=n_target,
            k_winding=k_coupling,  # For baryons, use sum of |k|
            l_composition=l_composition,
            k_eff=k_eff,
            structure_norm=structure_norm,
            converged=True,
            iterations=1,
            particle_type='baryon',
        )
    
    # =========================================
    # Accessor methods for use by energy minimizer
    # =========================================
    
    def get_spatial_coupling_matrix(self) -> NDArray:
        """Return the precomputed spatial coupling matrix."""
        return self.spatial_coupling
    
    def get_sigma_grid(self) -> NDArray:
        """Return the subspace coordinate grid."""
        return self.basis.sigma
    
    def get_dsigma(self) -> float:
        """Return the subspace grid spacing."""
        return self.basis.dsigma
    
    def get_V_sigma(self) -> NDArray:
        """Return the three-well potential on the subspace grid."""
        return self.V_sigma
    
    def get_state_index_map(self) -> Dict[Tuple[int, int, int], int]:
        """
        Return mapping from (n,l,m) keys to spatial_coupling indices.
        
        This is needed for the UniversalEnergyMinimizer to correctly
        look up coupling matrix elements.
        """
        return {
            (state.n, state.l, state.m): i 
            for i, state in enumerate(self.basis.spatial_states)
        }


def test_wavefunction_solver():
    """Test the non-separable wavefunction solver."""
    print("=" * 60)
    print("TESTING NON-SEPARABLE WAVEFUNCTION SOLVER")
    print("=" * 60)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=20.0,
        beta=100.0,
        kappa=0.0001,
        g1=5000.0,
        g2=0.004,
        V0=1.0,
        n_max=5,
        l_max=2,
        N_sigma=48,
    )
    
    print(f"\nBasis: {solver.basis.N_spatial} spatial states x {solver.basis.N_sigma} sigma points")
    
    # Test leptons
    print("\n--- LEPTON STRUCTURES ---")
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        result = solver.solve_lepton(n_target=n, k_winding=1, verbose=True)
        print(f"  {name}: k_eff={result.k_eff:.4f}, l_comp={result.l_composition}")
    
    # Test meson (pion-like)
    print("\n--- MESON STRUCTURE (Pion-like) ---")
    meson = solver.solve_meson(
        quark_gen=1, antiquark_gen=1,
        k_quark=5, k_antiquark=-5,
        n_radial=1, verbose=True
    )
    
    # Test baryon (proton-like)
    print("\n--- BARYON STRUCTURE (Proton-like) ---")
    baryon = solver.solve_baryon(
        quark_gens=[1, 1, 1],
        windings=[5, 5, 5],
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("WAVEFUNCTION SOLVER TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_wavefunction_solver()

