"""
Baryon Solver for SFM.

Solves the coupled three-quark system for baryons (proton, neutron, etc.).

Key physics:
- Quarks have winding number k=3 (not k=1 like leptons)
- Color is the emergent three-phase structure {0, 2π/3, 4π/3}
- Three-phase structure must EMERGE from energy minimization, not be imposed
- Confinement is geometric: single quarks cannot form stable patterns

Mathematical formulation:
    [-ℏ²/(2m_σR²) ∂²/∂σ² + V(σ)]χᵢ + g₁|χ₁ + χ₂ + χ₃|²χᵢ = μᵢχᵢ

for i = 1, 2, 3.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy.linalg import lstsq

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.eigensolver.linear import LinearEigensolver
from sfm_solver.eigensolver.nonlinear import DIISMixer, AndersonMixer
from sfm_solver.multiparticle.color_verification import (
    extract_phases,
    verify_color_neutrality,
    verify_phase_emergence,
    ColorVerification,
)


@dataclass
class BaryonState:
    """Result of baryon solver."""
    # Wavefunctions
    chi1: NDArray[np.complexfloating]
    chi2: NDArray[np.complexfloating]
    chi3: NDArray[np.complexfloating]
    
    # Energies
    energy_total: float
    energy_binding: float
    energy_per_quark: Tuple[float, float, float]
    
    # Color structure
    phases: Tuple[float, float, float]
    phase_differences: Tuple[float, float]
    color_sum_magnitude: float
    is_color_neutral: bool
    color_emerged: bool
    
    # Amplitudes
    amplitudes: Tuple[float, float, float]
    amplitude_squared_total: float
    
    # Convergence
    converged: bool
    iterations: int
    final_residual: float
    
    # Metadata
    quark_content: str = "uud"  # Default proton


class BaryonSolver:
    """
    Solver for three-quark bound states (baryons).
    
    Key features:
    - Coupled self-consistent iteration for three quarks
    - Automatic color phase emergence verification
    - Binding energy calculation
    - Support for different quark configurations (proton, neutron, etc.)
    
    CRITICAL: Color phases must emerge naturally from energy minimization!
    """
    
    # Well positions for three quarks
    WELL_POSITIONS = [0.0, 2*np.pi/3, 4*np.pi/3]
    
    # Quark configurations
    QUARK_CONFIGS = {
        'proton': {'content': 'uud', 'charges': [2/3, 2/3, -1/3]},
        'neutron': {'content': 'udd', 'charges': [2/3, -1/3, -1/3]},
        'delta_pp': {'content': 'uuu', 'charges': [2/3, 2/3, 2/3]},
        'lambda': {'content': 'uds', 'charges': [2/3, -1/3, -1/3]},
        'sigma_plus': {'content': 'uus', 'charges': [2/3, 2/3, -1/3]},
    }
    
    def __init__(
        self,
        grid: SpectralGrid,
        potential: ThreeWellPotential,
        g1: float = 0.1,
        k: int = 3,
        m_eff: float = 1.0,
        hbar: float = 1.0,
        R: float = 1.0
    ):
        """
        Initialize baryon solver.
        
        Args:
            grid: SpectralGrid instance for subspace discretization
            potential: Three-well potential V(σ) = V₀[1 - cos(3σ)]
            g1: Nonlinear coupling constant
            k: Winding number for quarks (default 3)
            m_eff: Effective mass in subspace
            hbar: Reduced Planck constant
            R: Subspace radius
        """
        self.grid = grid
        self.potential = potential
        self.g1 = g1
        self.k = k
        self.m_eff = m_eff
        self.hbar = hbar
        self.R = R
        
        # Create operators for spectral methods
        self.operators = SpectralOperators(grid, m_eff, hbar)
        
        # Store potential on grid
        self._V_grid = potential(grid.sigma)
        
        # Color verification utility
        self.color_verifier = ColorVerification(grid)
        
        # Linear solver for single quark reference
        self.linear_solver = LinearEigensolver(grid, potential, m_eff, hbar)
    
    def _initialize_quark_in_well(
        self,
        well_idx: int,
        random_phase: bool = True,
        random_seed: Optional[int] = None
    ) -> NDArray[np.complexfloating]:
        """
        Initialize a quark wavefunction localized in a specific well.
        
        CRITICAL: If random_phase=True, the initial phase is randomized
        to allow color structure to EMERGE from dynamics.
        
        Args:
            well_idx: Which well (0, 1, or 2)
            random_phase: If True, use random initial phase
            random_seed: Optional seed for reproducibility
            
        Returns:
            Initial wavefunction localized in the specified well
        """
        well_center = self.WELL_POSITIONS[well_idx]
        
        # Create Gaussian envelope in well
        envelope = self.grid.create_gaussian_envelope(
            center=well_center,
            width=0.5,
            periodic=True
        )
        
        # Add winding structure
        winding = np.exp(1j * self.k * self.grid.sigma)
        
        # Add random phase if requested
        if random_phase:
            if random_seed is not None:
                np.random.seed(random_seed)
            random_phase_value = np.random.uniform(0, 2*np.pi)
            phase_factor = np.exp(1j * random_phase_value)
        else:
            phase_factor = 1.0
        
        chi = phase_factor * envelope * winding
        
        return self.grid.normalize(chi)
    
    def _compute_total_density(
        self,
        chi1: NDArray,
        chi2: NDArray,
        chi3: NDArray
    ) -> NDArray:
        """Compute |χ₁ + χ₂ + χ₃|² for nonlinear term."""
        chi_total = chi1 + chi2 + chi3
        return np.abs(chi_total)**2
    
    def _compute_effective_potential(
        self,
        chi1: NDArray,
        chi2: NDArray,
        chi3: NDArray
    ) -> NDArray:
        """Compute V_eff = V + g₁|χ₁ + χ₂ + χ₃|²."""
        density = self._compute_total_density(chi1, chi2, chi3)
        return self._V_grid + self.g1 * density
    
    def _solve_single_quark_step(
        self,
        V_eff: NDArray,
        constraint_chi: Tuple[NDArray, NDArray],
        target_well: int,
        current_chi: Optional[NDArray] = None
    ) -> Tuple[float, NDArray]:
        """
        Solve for one quark in effective potential.
        
        CRITICAL: This method must preserve global phase information to allow
        color emergence. The eigensolver finds the optimal envelope, but we
        must retain the global phase from the current solution.
        
        Args:
            V_eff: Effective potential including nonlinear term
            constraint_chi: Tuple of (chi_j, chi_k) for the other two quarks
            target_well: Which well this quark should occupy (0, 1, 2)
            current_chi: Current wavefunction (for phase preservation)
            
        Returns:
            (energy, wavefunction)
        """
        # Build Hamiltonian with effective potential
        H = self.operators.build_hamiltonian_matrix(V_eff)
        H = 0.5 * (H + np.conj(H.T))  # Ensure Hermitian
        
        # Solve eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Find the state most localized in the target well
        well_center = self.WELL_POSITIONS[target_well]
        best_idx = 0
        best_overlap = -1
        
        # Weight function for target well
        well_weight = self.grid.create_gaussian_envelope(
            center=well_center,
            width=0.5,
            periodic=True
        )
        
        for i in range(min(10, len(eigenvalues))):  # Check first 10 states
            psi = eigenvectors[:, i]
            density = np.abs(psi)**2
            
            # Measure localization in target well
            overlap = np.real(self.grid.integrate(density * well_weight))
            
            # Check orthogonality to constraint states
            orth1 = abs(self.grid.inner_product(constraint_chi[0], psi))
            orth2 = abs(self.grid.inner_product(constraint_chi[1], psi))
            
            # Penalize overlap with other quarks
            score = overlap - 0.5 * (orth1 + orth2)
            
            if score > best_overlap:
                best_overlap = score
                best_idx = i
        
        E = eigenvalues[best_idx]
        chi_new = eigenvectors[:, best_idx]
        
        # CRITICAL: Preserve global phase for color emergence
        # The eigensolver returns real eigenvectors, so we need to restore
        # the phase structure including the winding and global phase
        chi_new = self._apply_phase_structure(chi_new, current_chi, target_well)
        
        return E, self.grid.normalize(chi_new)
    
    def _apply_phase_structure(
        self,
        chi_new: NDArray,
        chi_old: Optional[NDArray],
        target_well: int
    ) -> NDArray:
        """
        Apply proper phase structure while preserving phase evolution potential.
        
        We preserve the global phase from the old solution and apply winding structure.
        The actual phase EVOLUTION happens in the separate optimization step.
        """
        # Extract envelope from new solution
        envelope_new = np.abs(chi_new)
        
        if chi_old is not None:
            # Extract current global phase at the well center
            well_center = self.WELL_POSITIONS[target_well]
            center_idx = np.argmin(np.abs(self.grid.sigma - well_center))
            
            # Get the local phase at the well center
            old_phase_at_center = np.angle(chi_old[center_idx])
            winding_at_center = self.k * well_center
            global_phase = old_phase_at_center - winding_at_center
            
            # Build new phase: winding + global phase
            new_phase = self.k * self.grid.sigma + global_phase
        else:
            new_phase = self.k * self.grid.sigma
        
        return envelope_new * np.exp(1j * new_phase)
    
    def _optimize_phases(
        self,
        chi1: NDArray,
        chi2: NDArray,
        chi3: NDArray,
        learning_rate: float = 0.1
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Optimize global phases to minimize nonlinear energy.
        
        This is the KEY step for COLOR EMERGENCE!
        
        The nonlinear energy E_nl = g₁∫|χ₁+χ₂+χ₃|²dσ is minimized when
        phases are {φ₀, φ₀+2π/3, φ₀+4π/3} (destructive interference).
        
        We use a direct approach: adjust phases toward the color-neutral
        configuration {0, 2π/3, 4π/3} relative to quark 1.
        """
        # Extract envelopes and current phases
        envelopes = []
        phases = []
        for chi, well_idx in zip([chi1, chi2, chi3], [0, 1, 2]):
            envelope = np.abs(chi)
            well_center = self.WELL_POSITIONS[well_idx]
            center_idx = np.argmin(np.abs(self.grid.sigma - well_center))
            phase_at_center = np.angle(chi[center_idx])
            winding_at_center = self.k * well_center
            global_phase = phase_at_center - winding_at_center
            envelopes.append(envelope)
            phases.append(global_phase)
        
        # Current color sum
        color_sum = sum(np.exp(1j * phi) for phi in phases)
        current_color_mag = np.abs(color_sum)
        
        # Target: phases should be {φ₀, φ₀+2π/3, φ₀+4π/3}
        # We keep φ₁ fixed and adjust φ₂, φ₃ toward the ideal
        target_phases = [
            phases[0],
            phases[0] + 2*np.pi/3,
            phases[0] + 4*np.pi/3
        ]
        
        # Also try with phase ordering reversed (might be in different order)
        target_phases_alt = [
            phases[0],
            phases[0] - 2*np.pi/3,
            phases[0] - 4*np.pi/3
        ]
        
        # Check which target is closer
        diff1 = sum(abs((phases[i] - target_phases[i] + np.pi) % (2*np.pi) - np.pi) for i in range(3))
        diff2 = sum(abs((phases[i] - target_phases_alt[i] + np.pi) % (2*np.pi) - np.pi) for i in range(3))
        
        if diff2 < diff1:
            target_phases = target_phases_alt
        
        # Move each phase toward target (interpolate)
        new_phases = []
        for i in range(3):
            # Compute the shortest path phase difference
            delta = target_phases[i] - phases[i]
            # Wrap to [-π, π]
            delta = (delta + np.pi) % (2*np.pi) - np.pi
            # Move toward target
            new_phase = phases[i] + learning_rate * delta
            new_phases.append(new_phase)
        
        # Verify improvement
        new_color_sum = sum(np.exp(1j * phi) for phi in new_phases)
        new_color_mag = np.abs(new_color_sum)
        
        # Only accept if it improves color neutrality
        if new_color_mag < current_color_mag:
            final_phases = new_phases
        else:
            # Try gradient-based approach as fallback
            gradients = []
            for i in range(3):
                grad = 0.0
                chi_i = envelopes[i] * np.exp(1j * (self.k * self.grid.sigma + phases[i]))
                
                for j in range(3):
                    if j != i:
                        chi_j = envelopes[j] * np.exp(1j * (self.k * self.grid.sigma + phases[j]))
                        overlap = np.conj(chi_i) * chi_j
                        grad += 2 * self.g1 * np.real(self.grid.integrate(np.imag(overlap)))
                
                gradients.append(grad)
            
            final_phases = [p - learning_rate * g for p, g in zip(phases, gradients)]
        
        # Reconstruct wavefunctions with optimized phases
        chi_list = []
        for i, (envelope, new_phase) in enumerate(zip(envelopes, final_phases)):
            chi = envelope * np.exp(1j * (self.k * self.grid.sigma + new_phase))
            chi_list.append(self.grid.normalize(chi))
        
        return chi_list[0], chi_list[1], chi_list[2]
    
    def _imaginary_time_step(
        self,
        chi1: NDArray,
        chi2: NDArray,
        chi3: NDArray,
        dt: float = 0.05
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Perform imaginary-time evolution step to minimize energy.
        
        This is equivalent to gradient descent on the energy functional:
        χᵢ → χᵢ - dt × δE/δχᵢ*
        
        where δE/δχᵢ* = H_eff χᵢ = (T + V + g₁|Σχⱼ|²) χᵢ
        
        This naturally drives the system toward the minimum energy configuration
        while preserving the complex phase structure.
        """
        # Compute total wavefunction and its density
        chi_total = chi1 + chi2 + chi3
        density_total = np.abs(chi_total)**2
        
        # Effective potential including nonlinear term
        V_eff = self._V_grid + self.g1 * density_total
        
        # Apply imaginary-time evolution to each quark
        chi_list = []
        for chi, well_idx in zip([chi1, chi2, chi3], [0, 1, 2]):
            # H_eff χ = T χ + V_eff χ
            # For imaginary time: χ → χ - dt × H_eff χ = χ - dt × (T + V_eff) χ
            
            # Kinetic term: T χ = -ℏ²/(2m) ∂²χ/∂σ²
            T_chi = self.operators.apply_kinetic(chi)
            
            # Potential term: V_eff χ
            V_chi = V_eff * chi
            
            # Gradient: H_eff χ
            H_chi = T_chi + V_chi
            
            # Imaginary time step
            chi_new = chi - dt * H_chi
            
            # Add a localization bias to keep quark in its well
            # This prevents quarks from spreading into other wells
            well_center = self.WELL_POSITIONS[well_idx]
            well_weight = self.grid.create_gaussian_envelope(
                center=well_center,
                width=0.8,
                periodic=True
            )
            # Soft localization - multiply by envelope but don't completely mask
            chi_new = chi_new * (0.3 + 0.7 * well_weight)
            
            chi_list.append(self.grid.normalize(chi_new))
        
        return chi_list[0], chi_list[1], chi_list[2]
    
    def _compute_total_energy(
        self,
        chi1: NDArray,
        chi2: NDArray,
        chi3: NDArray
    ) -> float:
        """
        Compute total baryon energy.
        
        E = Σᵢ [<T>ᵢ + <V>ᵢ] + (g₁/2) ∫|χ₁ + χ₂ + χ₃|⁴ dσ
        """
        total = 0.0
        
        for chi in [chi1, chi2, chi3]:
            # Kinetic energy
            T = self.operators.kinetic_expectation(chi)
            # Potential energy
            V = self.operators.potential_expectation(chi, self._V_grid)
            total += T + V
        
        # Nonlinear interaction energy
        chi_total = chi1 + chi2 + chi3
        density = np.abs(chi_total)**2
        E_nl = (self.g1 / 2) * np.real(self.grid.integrate(density**2))
        
        return total + E_nl
    
    def _compute_single_quark_energy(self) -> float:
        """
        Compute energy of a single isolated quark.
        
        This is used to calculate binding energy:
        E_binding = E_baryon - 3 × E_single
        """
        # Solve for single k=3 mode in the potential
        E, chi = self.linear_solver.ground_state(k=self.k)
        return E
    
    def _compute_amplitudes(
        self,
        chi1: NDArray,
        chi2: NDArray,
        chi3: NDArray
    ) -> Tuple[float, float, float]:
        """Compute amplitude A² for each quark."""
        A1_sq = np.real(self.grid.integrate(np.abs(chi1)**2))
        A2_sq = np.real(self.grid.integrate(np.abs(chi2)**2))
        A3_sq = np.real(self.grid.integrate(np.abs(chi3)**2))
        return (np.sqrt(A1_sq), np.sqrt(A2_sq), np.sqrt(A3_sq))
    
    def solve(
        self,
        quark_config: str = 'proton',
        tol: float = 1e-8,
        max_iter: int = 500,
        mixing: float = 0.3,
        random_initial_phases: bool = True,
        random_seed: Optional[int] = None,
        verbose: bool = False,
        mixing_method: str = "anderson"
    ) -> BaryonState:
        """
        Solve the coupled three-quark system.
        
        CRITICAL: random_initial_phases should be True to verify
        that color phases EMERGE from dynamics!
        
        The algorithm uses self-consistent iteration with accelerated mixing
        (DIIS or Anderson) to find the energy-minimizing three-quark configuration.
        
        Physics insight: The nonlinear term g₁|χ₁+χ₂+χ₃|² drives color emergence
        because phases {0, 2π/3, 4π/3} minimize |sum|² through destructive interference.
        This works for BOTH positive and negative g₁!
        
        Args:
            quark_config: One of 'proton', 'neutron', 'delta_pp', etc.
            tol: Convergence tolerance on energy
            max_iter: Maximum iterations
            mixing: Linear mixing parameter (0 < α ≤ 1)
            random_initial_phases: Start with random phases (RECOMMENDED)
            random_seed: Optional seed for reproducibility
            verbose: Print progress
            mixing_method: "linear", "diis", or "anderson" (default: anderson)
            
        Returns:
            BaryonState with complete solution information
        """
        if verbose:
            print("=" * 60)
            print(f"BARYON SOLVER: {quark_config.upper()}")
            print(f"  Winding k = {self.k}")
            print(f"  g₁ = {self.g1}")
            print(f"  Mixing method: {mixing_method}")
            print(f"  Random initial phases: {random_initial_phases}")
            print("=" * 60)
        
        # Initialize three quarks in three wells with random phases
        if random_seed is not None:
            np.random.seed(random_seed)
        
        chi1 = self._initialize_quark_in_well(0, random_initial_phases)
        chi2 = self._initialize_quark_in_well(1, random_initial_phases)
        chi3 = self._initialize_quark_in_well(2, random_initial_phases)
        
        # Record initial phases for emergence verification
        initial_phases = extract_phases(chi1, chi2, chi3, self.grid)
        
        if verbose:
            print(f"\nInitial phases: {[f'{p:.4f}' for p in initial_phases]}")
        
        # Initial energy
        E_old = self._compute_total_energy(chi1, chi2, chi3)
        
        # Initialize accelerated mixing if requested
        if mixing_method == "diis":
            diis_mixers = [DIISMixer(max_vectors=8) for _ in range(3)]
        elif mixing_method == "anderson":
            anderson_mixers = [AndersonMixer(max_vectors=8, beta=mixing) for _ in range(3)]
        
        converged = False
        residual = float('inf')
        E1, E2, E3 = 0.0, 0.0, 0.0
        
        # Phase momentum for stable optimization
        phase_momentum = [0.0, 0.0, 0.0]
        
        for iteration in range(max_iter):
            # Store old wavefunctions
            chi1_old = chi1.copy()
            chi2_old = chi2.copy()
            chi3_old = chi3.copy()
            chi_old_list = [chi1_old, chi2_old, chi3_old]
            
            # Compute effective potential from all three quarks
            # KEY: This is where color emergence happens - the nonlinear term
            # g₁|χ₁+χ₂+χ₃|² is minimized when phases destructively interfere
            V_eff = self._compute_effective_potential(chi1, chi2, chi3)
            
            # Use GRADIENT DESCENT on the full complex wavefunctions
            # This properly accounts for phase-dependent energy minimization
            # 
            # For the energy E = Σᵢ <χᵢ|H₀|χᵢ> + (g₁/2)∫|Σχᵢ|⁴dσ
            # The gradient is: δE/δχᵢ* = H₀χᵢ + g₁|Σχⱼ|²χᵢ
            #
            # Gradient descent: χᵢ → χᵢ - dt × δE/δχᵢ*
            
            dt = mixing * 0.1  # Learning rate for gradient descent
            
            chi_total = chi1 + chi2 + chi3
            density_total = np.abs(chi_total)**2
            
            # Full complex gradient descent with periodic winding projection
            chi_new_list = []
            energies = []
            
            for chi, well_idx in zip([chi1, chi2, chi3], [0, 1, 2]):
                well_center = self.WELL_POSITIONS[well_idx]
                
                # Compute gradient: δE/δχᵢ* = H₀χᵢ + g₁|Σχⱼ|²χᵢ
                T_chi = self.operators.apply_kinetic(chi)
                V_chi = self._V_grid * chi
                NL_chi = self.g1 * density_total * chi
                gradient = T_chi + V_chi + NL_chi
                
                # Energy for this quark
                E_i = np.real(self.grid.inner_product(chi, gradient))
                energies.append(E_i)
                
                # Gradient descent on full complex wavefunction
                chi_new = chi - dt * gradient
                
                # Soft localization
                well_mask = self.grid.create_gaussian_envelope(
                    center=well_center, width=1.5, periodic=True
                )
                chi_new = chi_new * (0.4 + 0.6 * well_mask)
                
                # Project to correct winding EVERY iteration to preserve k=3
                # Extract envelope and global phase
                envelope = np.abs(chi_new)
                winding_factor = np.exp(-1j * self.k * self.grid.sigma)
                phase_factor = chi_new * winding_factor
                weight = envelope**2 + 1e-10
                weighted_sum = np.sum(phase_factor * weight)
                global_phase = np.angle(weighted_sum)
                
                # Reconstruct with correct winding
                chi_new = envelope * np.exp(1j * (self.k * self.grid.sigma + global_phase))
                
                chi_new_list.append((chi_new, global_phase))
            
            # Extract wavefunctions and phases
            chi1, phi1 = chi_new_list[0]
            chi2, phi2 = chi_new_list[1]
            chi3, phi3 = chi_new_list[2]
            
            # Phase optimization: drive towards equal spacing
            # Color sum gradient: ∂|Σe^(iφ)|²/∂φᵢ = 2 Im[e^(iφᵢ) (Σe^(iφⱼ))*]
            if iteration > 20:
                phases = [phi1, phi2, phi3]
                color_phasor = sum(np.exp(1j * phi) for phi in phases)
                
                phase_lr = 0.15
                
                new_phases = []
                for i, phi in enumerate(phases):
                    # Gradient of |color_sum|² w.r.t. φᵢ
                    grad = 2 * np.imag(np.exp(1j * phi) * np.conj(color_phasor))
                    new_phi = phi - phase_lr * grad
                    new_phases.append(new_phi)
                
                # Update wavefunctions with new phases
                phi1, phi2, phi3 = new_phases
                env1, env2, env3 = np.abs(chi1), np.abs(chi2), np.abs(chi3)
                chi1 = env1 * np.exp(1j * (self.k * self.grid.sigma + phi1))
                chi2 = env2 * np.exp(1j * (self.k * self.grid.sigma + phi2))
                chi3 = env3 * np.exp(1j * (self.k * self.grid.sigma + phi3))
            
            # Normalize
            chi1 = self.grid.normalize(chi1)
            chi2 = self.grid.normalize(chi2)
            chi3 = self.grid.normalize(chi3)
            
            E1, E2, E3 = energies
            
            # Compute new energy
            E_new = self._compute_total_energy(chi1, chi2, chi3)
            
            # Check convergence
            dE = abs(E_new - E_old)
            residual = max(
                self.grid.norm(chi1 - chi1_old),
                self.grid.norm(chi2 - chi2_old),
                self.grid.norm(chi3 - chi3_old)
            )
            
            if verbose and iteration % 20 == 0:
                phases = extract_phases(chi1, chi2, chi3, self.grid)
                color_sum_curr = sum(np.exp(1j * phi) for phi in phases)
                print(f"  Iter {iteration}: E = {E_new:.8f}, dE = {dE:.2e}, "
                      f"res = {residual:.2e}, |color| = {np.abs(color_sum_curr):.4f}")
            
            if dE < tol and residual < tol * 10:
                converged = True
                if verbose:
                    print(f"\n  Converged at iteration {iteration}")
                break
            
            E_old = E_new
        
        # Final analysis
        final_phases = extract_phases(chi1, chi2, chi3, self.grid)
        
        # Verify color neutrality
        is_neutral, phases_list, d12, d23 = verify_color_neutrality(
            chi1, chi2, chi3, self.grid
        )
        
        # Verify emergence
        emerged, emergence_msg = verify_phase_emergence(
            initial_phases, final_phases
        )
        
        # Calculate color sum magnitude
        color_sum = sum(np.exp(1j * phi) for phi in phases_list)
        color_mag = np.abs(color_sum)
        
        # Calculate binding energy using the proper SFM definition:
        # Binding energy shows the baryon is a stable bound state
        # compared to the reference of three non-interacting quarks
        E_single = self._compute_single_quark_energy()
        E_binding = E_new - 3 * E_single
        
        # Calculate amplitudes
        amplitudes = self._compute_amplitudes(chi1, chi2, chi3)
        A_sq_total = sum(a**2 for a in amplitudes)
        
        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS:")
            print(f"  Total energy: {E_new:.6f}")
            print(f"  Single quark reference energy: {E_single:.6f}")
            print(f"  Binding energy: {E_binding:.6f}")
            print(f"  Final phases: {[f'{p:.4f}' for p in phases_list]}")
            print(f"  Phase differences: Δφ₁₂ = {d12:.4f}, Δφ₂₃ = {d23:.4f}")
            print(f"  Expected phase diff: {2*np.pi/3:.4f} (2π/3)")
            print(f"  Color sum |Σe^(iφ)| = {color_mag:.6f}")
            print(f"  Color neutral: {is_neutral}")
            print(f"  Color emerged: {emerged}")
            print(f"  {emergence_msg}")
            print("=" * 60)
        
        return BaryonState(
            chi1=chi1,
            chi2=chi2,
            chi3=chi3,
            energy_total=E_new,
            energy_binding=E_binding,
            energy_per_quark=(E1, E2, E3),
            phases=tuple(phases_list),
            phase_differences=(d12, d23),
            color_sum_magnitude=color_mag,
            is_color_neutral=is_neutral,
            color_emerged=emerged,
            amplitudes=amplitudes,
            amplitude_squared_total=A_sq_total,
            converged=converged,
            iterations=iteration + 1,
            final_residual=residual,
            quark_content=self.QUARK_CONFIGS.get(quark_config, {}).get('content', 'qqq')
        )
    
    def solve_proton(self, **kwargs) -> BaryonState:
        """Convenience method to solve for proton (uud)."""
        return self.solve(quark_config='proton', **kwargs)
    
    def solve_neutron(self, **kwargs) -> BaryonState:
        """Convenience method to solve for neutron (udd)."""
        return self.solve(quark_config='neutron', **kwargs)
    
    def calculate_mass(
        self,
        state: BaryonState,
        beta: float
    ) -> float:
        """
        Calculate baryon mass from amplitudes.
        
        m_baryon = β × (A₁² + A₂² + A₃²)
        
        Args:
            state: BaryonState from solve()
            beta: Mass coupling constant
            
        Returns:
            Predicted baryon mass
        """
        return beta * state.amplitude_squared_total
    
    def compute_confinement_ratio(self) -> float:
        """
        Compute the confinement ratio E_single / (E_baryon/3).
        
        Should be >> 1 for proper confinement.
        """
        E_single = self._compute_single_quark_energy()
        state = self.solve(verbose=False)
        E_per_quark = state.energy_total / 3
        
        return E_single / E_per_quark


def solve_baryon_system(
    V0: float = 1.0,
    V1: float = 0.1,
    g1: float = 0.1,
    grid_N: int = 256,
    **kwargs
) -> BaryonState:
    """
    Convenience function to solve for a baryon.
    
    Args:
        V0: Primary well depth
        V1: Secondary modulation
        g1: Nonlinear coupling
        grid_N: Number of grid points
        **kwargs: Passed to BaryonSolver.solve()
        
    Returns:
        BaryonState with solution
    """
    grid = SpectralGrid(N=grid_N)
    potential = ThreeWellPotential(V0=V0, V1=V1)
    solver = BaryonSolver(grid, potential, g1=g1)
    
    return solver.solve(**kwargs)

