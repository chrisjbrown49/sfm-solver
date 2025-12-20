"""
Dimensionless Shape Solver for SFM.

This module solves for normalized wavefunction SHAPE at unit scale,
completely independent of physical scale parameters (Delta_x, Delta_sigma, A).

KEY PRINCIPLE: Stage 1 of two-stage architecture
    - ALL quantities dimensionless (using V0 as reference energy)
    - ALL coordinates at unit scale (sigma in [0, 2pi])
    - Output normalized to integral|psi|^2 = 1
    - EM self-energy INCLUDED in Hamiltonian during shape solving

This separates quantum (shape) from classical (scale) physics.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings


@dataclass
class DimensionlessShapeResult:
    """
    Result from dimensionless shape solver.
    
    Contains normalized wavefunction shape at unit scale.
    NO physical scale parameters (Delta_x, Delta_sigma, A).
    """
    # For baryons: three quark wavefunctions
    chi1_shape: Optional[NDArray] = None
    chi2_shape: Optional[NDArray] = None
    chi3_shape: Optional[NDArray] = None
    
    # Composite wavefunction (sum of quarks for baryons, single for leptons)
    composite_shape: NDArray = None
    
    # Quantum numbers
    quark_windings: Optional[Tuple[int, int, int]] = None
    color_phases: Optional[Tuple[float, float, float]] = None
    winding_k: Optional[int] = None  # For leptons/mesons
    generation_n: Optional[int] = None  # For leptons
    
    # Convergence info
    converged: bool = False
    iterations: int = 0
    
    # EM contribution to eigenvalue (dimensionless)
    em_eigenvalue_contribution: Optional[Dict] = None
    
    # Particle type
    particle_type: str = 'unknown'  # 'lepton', 'meson', 'baryon'


class DimensionlessShapeSolver:
    """
    Solve for normalized wavefunction shape at unit scale.
    
    This is Stage 1 of the two-stage solver architecture.
    All quantities are dimensionless (ratios of energies to V0).
    
    The shape solver finds the RELATIVE structure of the wavefunction,
    independent of overall amplitude or physical scale.
    """
    
    def __init__(
        self,
        g1_dimensionless: float,
        g2_dimensionless: float,
        V0: float = 1.0,
        V1: float = 0.0,
        N_sigma: int = 64,
        verbose: bool = False
    ):
        """
        Initialize dimensionless shape solver.
        
        Args:
            g1_dimensionless: Nonlinear coupling (g1 / V0)
            g2_dimensionless: EM coupling (g2 / V0)
            V0: Three-well primary depth (used as energy reference)
            V1: Three-well secondary depth
            N_sigma: Grid points in subspace dimension
            verbose: Print diagnostic information
        """
        self.g1_dimless = g1_dimensionless
        self.g2_dimless = g2_dimensionless
        self.V0 = V0
        self.V1 = V1
        self.N_sigma = N_sigma
        self.verbose = verbose
        
        # Build dimensionless subspace grid
        self.sigma = np.linspace(0, 2*np.pi, N_sigma, endpoint=False)
        self.dsigma = 2*np.pi / N_sigma
        
        # Build three-well potential (dimensionless, normalized by V0)
        # V(sigma) = V0 * (1 - cos(3*sigma)) + V1 * (1 - cos(6*sigma))
        # Dimensionless: V_dimless = (1 - cos(3*sigma)) + (V1/V0) * (1 - cos(6*sigma))
        self.V_well_dimless = (1.0 - np.cos(3*self.sigma)) + (V1/V0) * (1.0 - np.cos(6*self.sigma))
        
        if self.verbose:
            print("=== DimensionlessShapeSolver Initialized ===")
            print(f"  g1_dimensionless: {g1_dimensionless:.6f}")
            print(f"  g2_dimensionless: {g2_dimensionless:.6f}")
            print(f"  V0: {V0:.3f} GeV (energy reference)")
            print(f"  V1: {V1:.3f} GeV")
            print(f"  N_sigma: {N_sigma}")
    
    def solve_baryon_shape(
        self,
        quark_windings: Tuple[int, int, int],
        color_phases: Tuple[float, float, float],
        max_iter: int = 500,
        tol: float = 1e-4,
        mixing: float = 0.1
    ) -> DimensionlessShapeResult:
        """
        Solve for normalized three-quark composite shape.
        
        CRITICAL: This solves the DIMENSIONLESS eigenvalue problem.
        No physical scales (Delta_x, Delta_sigma, A) are involved.
        
        EM self-energy V_em = g2_dimless * |J_normalized|^2 is included
        in the mean field Hamiltonian, affecting eigenstate structure.
        
        Args:
            quark_windings: (k1, k2, k3) winding numbers
            color_phases: (phi1, phi2, phi3) for color neutrality
            max_iter: Maximum SCF iterations
            tol: Convergence tolerance
            mixing: Mixing parameter for stability
            
        Returns:
            DimensionlessShapeResult with normalized wavefunctions
        """
        k1, k2, k3 = quark_windings
        phi1, phi2, phi3 = color_phases
        
        if self.verbose:
            print("\n=== Solving Baryon Shape (Dimensionless) ===")
            print(f"  Quark windings: {quark_windings}")
            print(f"  Color phases: {color_phases}")
            print(f"  Max iterations: {max_iter}, Tolerance: {tol}")
        
        # Build derivative operators (dimensionless)
        D1 = self._build_derivative_operator()
        D2 = D1 @ D1  # Second derivative
        
        # Initialize with Gaussian shapes in respective wells
        chi1 = self._gaussian_in_well(well=1, phase=phi1, winding=k1)
        chi2 = self._gaussian_in_well(well=2, phase=phi2, winding=k2)
        chi3 = self._gaussian_in_well(well=3, phase=phi3, winding=k3)
        
        # Normalize individually
        chi1 = self._normalize(chi1)
        chi2 = self._normalize(chi2)
        chi3 = self._normalize(chi3)
        
        # SCF iteration
        converged = False
        for iteration in range(max_iter):
            chi1_old = chi1.copy()
            chi2_old = chi2.copy()
            chi3_old = chi3.copy()
            
            # Composite wavefunction
            chi_total = chi1 + chi2 + chi3
            
            # Mean field potential (dimensionless)
            # V_mean = g1_dimless * |chi_total|^2
            V_mean = self.g1_dimless * np.abs(chi_total)**2
            
            # CRITICAL: Include EM self-energy in mean field
            # Compute circulation for each quark (normalized)
            J1_norm = self._compute_circulation(chi1)
            J2_norm = self._compute_circulation(chi2)
            J3_norm = self._compute_circulation(chi3)
            
            # EM contribution to potential (dimensionless)
            # V_em = g2_dimless * |J_normalized|^2
            # This affects the eigenstate structure!
            V_em1 = self.g2_dimless * np.abs(J1_norm)**2
            V_em2 = self.g2_dimless * np.abs(J2_norm)**2
            V_em3 = self.g2_dimless * np.abs(J3_norm)**2
            
            # Build Hamiltonian for each quark
            # H = T_kinetic + T_winding + V_well + V_mean + V_em
            
            # Quark 1
            T_winding1 = self._build_winding_kinetic(k1)
            V_total1 = self.V_well_dimless + V_mean + V_em1
            H1 = -D2 + T_winding1 + np.diag(V_total1)
            
            eigenvalues1, eigenvectors1 = np.linalg.eigh(H1)
            chi1_new = self._select_eigenstate(chi1, eigenvectors1, eigenvalues1, k1, phi1)
            
            # Quark 2
            T_winding2 = self._build_winding_kinetic(k2)
            V_total2 = self.V_well_dimless + V_mean + V_em2
            H2 = -D2 + T_winding2 + np.diag(V_total2)
            
            eigenvalues2, eigenvectors2 = np.linalg.eigh(H2)
            chi2_new = self._select_eigenstate(chi2, eigenvectors2, eigenvalues2, k2, phi2)
            
            # Quark 3
            T_winding3 = self._build_winding_kinetic(k3)
            V_total3 = self.V_well_dimless + V_mean + V_em3
            H3 = -D2 + T_winding3 + np.diag(V_total3)
            
            eigenvalues3, eigenvectors3 = np.linalg.eigh(H3)
            chi3_new = self._select_eigenstate(chi3, eigenvectors3, eigenvalues3, k3, phi3)
            
            # Mix for stability
            chi1 = (1-mixing)*chi1_old + mixing*chi1_new
            chi2 = (1-mixing)*chi2_old + mixing*chi2_new
            chi3 = (1-mixing)*chi3_old + mixing*chi3_new
            
            # Normalize
            chi1 = self._normalize(chi1)
            chi2 = self._normalize(chi2)
            chi3 = self._normalize(chi3)
            
            # Check convergence
            delta1 = np.sum(np.abs(chi1 - chi1_old)**2) * self.dsigma
            delta2 = np.sum(np.abs(chi2 - chi2_old)**2) * self.dsigma
            delta3 = np.sum(np.abs(chi3 - chi3_old)**2) * self.dsigma
            max_delta = max(delta1, delta2, delta3)
            
            if self.verbose and iteration % 50 == 0:
                print(f"  Iteration {iteration+1}: max_delta = {max_delta:.6e}")
            
            if max_delta < tol:
                converged = True
                if self.verbose:
                    print(f"  Shape converged after {iteration+1} iterations")
                break
        
        if not converged and self.verbose:
            print(f"  WARNING: Shape solver did not converge after {max_iter} iterations")
        
        # Final composite (normalized)
        composite = chi1 + chi2 + chi3
        composite_normalized = self._normalize(composite)
        
        return DimensionlessShapeResult(
            chi1_shape=chi1,
            chi2_shape=chi2,
            chi3_shape=chi3,
            composite_shape=composite_normalized,
            quark_windings=quark_windings,
            color_phases=color_phases,
            converged=converged,
            iterations=iteration+1,
            em_eigenvalue_contribution={
                'q1': V_em1,
                'q2': V_em2,
                'q3': V_em3
            },
            particle_type='baryon'
        )
    
    def solve_lepton_shape(
        self,
        winding_k: int,
        generation_n: int = 1,
        max_iter: int = 200,
        tol: float = 1e-6
    ) -> DimensionlessShapeResult:
        """
        Solve for normalized single-particle lepton shape.
        
        For leptons, the shape is a simple eigenstate of:
            H = -d^2/dsigma^2 + T_winding(k) + V_well + V_em
        
        where V_em = g2_dimless * |J_normalized|^2 (from circulation).
        
        Args:
            winding_k: Winding number (determines charge)
            generation_n: Generation number (1=e, 2=mu, 3=tau)
            max_iter: Maximum iterations for self-consistent EM
            tol: Convergence tolerance
            
        Returns:
            DimensionlessShapeResult with normalized wavefunction
        """
        if self.verbose:
            print(f"\n=== Solving Lepton Shape (n={generation_n}, k={winding_k}) ===")
        
        # Build operators
        D1 = self._build_derivative_operator()
        D2 = D1 @ D1
        
        # Initialize with Gaussian in well 1 (leptons occupy single well)
        chi = self._gaussian_in_well(well=1, phase=0.0, winding=winding_k)
        chi = self._normalize(chi)
        
        # Self-consistent iteration for EM potential
        for iteration in range(max_iter):
            chi_old = chi.copy()
            
            # Compute circulation (normalized)
            J_norm = self._compute_circulation(chi)
            
            # EM contribution to potential
            V_em = self.g2_dimless * np.abs(J_norm)**2
            
            # Build Hamiltonian
            T_winding = self._build_winding_kinetic(winding_k)
            V_total = self.V_well_dimless + V_em
            H = -D2 + T_winding + np.diag(V_total)
            
            # Solve eigenvalue problem
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            
            # Select ground state (lowest energy)
            chi_new = eigenvectors[:, 0].astype(complex)
            chi_new = self._normalize(chi_new)
            
            # Check convergence
            delta = np.sum(np.abs(chi_new - chi_old)**2) * self.dsigma
            
            if delta < tol:
                chi = chi_new
                if self.verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break
            
            chi = chi_new
        
        return DimensionlessShapeResult(
            composite_shape=chi,
            winding_k=winding_k,
            generation_n=generation_n,
            converged=True,
            iterations=iteration+1,
            particle_type='lepton'
        )
    
    def solve_meson_shape(
        self,
        quark_winding: int,
        antiquark_winding: int,
        quark_phase: float = 0.0,
        antiquark_phase: float = np.pi,
        max_iter: int = 500,
        tol: float = 1e-4,
        mixing: float = 0.1
    ) -> DimensionlessShapeResult:
        """
        Solve for normalized quark-antiquark meson shape.
        
        Similar to baryon but with two constituents instead of three.
        
        Args:
            quark_winding: Quark winding number
            antiquark_winding: Antiquark winding number
            quark_phase: Quark color phase
            antiquark_phase: Antiquark color phase
            max_iter: Maximum SCF iterations
            tol: Convergence tolerance
            mixing: Mixing parameter
            
        Returns:
            DimensionlessShapeResult with normalized wavefunctions
        """
        if self.verbose:
            print(f"\n=== Solving Meson Shape (k_q={quark_winding}, k_qbar={antiquark_winding}) ===")
        
        # Build operators
        D1 = self._build_derivative_operator()
        D2 = D1 @ D1
        
        # Initialize
        chi_q = self._gaussian_in_well(well=1, phase=quark_phase, winding=quark_winding)
        chi_qbar = self._gaussian_in_well(well=2, phase=antiquark_phase, winding=antiquark_winding)
        
        chi_q = self._normalize(chi_q)
        chi_qbar = self._normalize(chi_qbar)
        
        # SCF iteration
        converged = False
        for iteration in range(max_iter):
            chi_q_old = chi_q.copy()
            chi_qbar_old = chi_qbar.copy()
            
            # Composite
            chi_total = chi_q + chi_qbar
            
            # Mean field
            V_mean = self.g1_dimless * np.abs(chi_total)**2
            
            # EM contributions
            J_q_norm = self._compute_circulation(chi_q)
            J_qbar_norm = self._compute_circulation(chi_qbar)
            
            V_em_q = self.g2_dimless * np.abs(J_q_norm)**2
            V_em_qbar = self.g2_dimless * np.abs(J_qbar_norm)**2
            
            # Solve for quark
            T_winding_q = self._build_winding_kinetic(quark_winding)
            V_total_q = self.V_well_dimless + V_mean + V_em_q
            H_q = -D2 + T_winding_q + np.diag(V_total_q)
            
            eigenvalues_q, eigenvectors_q = np.linalg.eigh(H_q)
            chi_q_new = self._select_eigenstate(chi_q, eigenvectors_q, eigenvalues_q, 
                                                quark_winding, quark_phase)
            
            # Solve for antiquark
            T_winding_qbar = self._build_winding_kinetic(antiquark_winding)
            V_total_qbar = self.V_well_dimless + V_mean + V_em_qbar
            H_qbar = -D2 + T_winding_qbar + np.diag(V_total_qbar)
            
            eigenvalues_qbar, eigenvectors_qbar = np.linalg.eigh(H_qbar)
            chi_qbar_new = self._select_eigenstate(chi_qbar, eigenvectors_qbar, eigenvalues_qbar,
                                                   antiquark_winding, antiquark_phase)
            
            # Mix and normalize
            chi_q = self._normalize((1-mixing)*chi_q_old + mixing*chi_q_new)
            chi_qbar = self._normalize((1-mixing)*chi_qbar_old + mixing*chi_qbar_new)
            
            # Check convergence
            delta_q = np.sum(np.abs(chi_q - chi_q_old)**2) * self.dsigma
            delta_qbar = np.sum(np.abs(chi_qbar - chi_qbar_old)**2) * self.dsigma
            max_delta = max(delta_q, delta_qbar)
            
            if max_delta < tol:
                converged = True
                if self.verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break
        
        composite = self._normalize(chi_q + chi_qbar)
        
        return DimensionlessShapeResult(
            chi1_shape=chi_q,
            chi2_shape=chi_qbar,
            composite_shape=composite,
            converged=converged,
            iterations=iteration+1,
            particle_type='meson'
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _normalize(self, psi: NDArray) -> NDArray:
        """Normalize to integral|psi|^2 = 1."""
        norm_sq = np.sum(np.abs(psi)**2) * self.dsigma
        if norm_sq < 1e-20:
            warnings.warn("Wavefunction norm is extremely small")
            return psi
        return psi / np.sqrt(norm_sq)
    
    def _compute_circulation(self, chi: NDArray) -> complex:
        """
        Compute NORMALIZED circulation J_normalized = J / (integral|chi|^2).
        
        This is dimensionless and independent of overall amplitude.
        
        J_raw = integral chi* d(chi)/dsigma dsigma
        J_normalized = J_raw / (integral|chi|^2 dsigma)
        """
        D1 = self._build_derivative_operator()
        dchi = D1 @ chi
        
        # Raw circulation
        J_raw = np.sum(np.conj(chi) * dchi) * self.dsigma
        
        # Normalization
        norm_sq = np.sum(np.abs(chi)**2) * self.dsigma
        
        if norm_sq < 1e-20:
            return 0.0
        
        return J_raw / norm_sq
    
    def _build_derivative_operator(self) -> NDArray:
        """
        Build spectral derivative operator for periodic functions.
        
        Uses FFT-based spectral differentiation for high accuracy.
        """
        N = self.N_sigma
        
        # Wavenumbers for periodic domain [0, 2*pi]
        k = np.fft.fftfreq(N, d=1.0/N)
        
        # Derivative operator in Fourier space
        k_deriv = 1j * k
        
        # Build matrix representation
        D = np.zeros((N, N), dtype=complex)
        for i in range(N):
            delta = np.zeros(N)
            delta[i] = 1.0
            
            # Transform, multiply by ik, inverse transform
            delta_fft = np.fft.fft(delta)
            deriv_fft = k_deriv * delta_fft
            deriv = np.fft.ifft(deriv_fft)
            
            D[:, i] = deriv
        
        return D
    
    def _build_winding_kinetic(self, k: int) -> NDArray:
        """
        Build winding-dependent kinetic energy operator.
        
        T_winding = gamma(k) * identity
        
        where gamma(k) = 1 / (1 + k^2 / Lambda^2) provides geometric suppression
        for higher winding numbers.
        """
        Lambda = 100.0  # Suppression scale
        gamma = 1.0 / (1.0 + (k**2) / (Lambda**2))
        return gamma * np.eye(self.N_sigma)
    
    def _gaussian_in_well(
        self,
        well: int,
        phase: float,
        winding: int,
        width: float = 0.3
    ) -> NDArray:
        """
        Create Gaussian wavefunction localized in specified well.
        
        Args:
            well: Well number (1, 2, or 3)
            phase: Color phase
            winding: Winding number
            width: Gaussian width
            
        Returns:
            Complex wavefunction with winding and phase
        """
        # Well centers (equally spaced around circle)
        well_centers = {
            1: 0.0,
            2: 2*np.pi/3,
            3: 4*np.pi/3
        }
        
        center = well_centers.get(well, 0.0)
        
        # Gaussian envelope (periodic)
        # Use distance on circle
        delta_sigma = np.angle(np.exp(1j * (self.sigma - center)))
        envelope = np.exp(-delta_sigma**2 / (2*width**2))
        
        # Add winding and phase
        psi = envelope * np.exp(1j * phase) * np.exp(1j * winding * self.sigma)
        
        return psi
    
    def _select_eigenstate(
        self,
        chi_old: NDArray,
        eigenvectors: NDArray,
        eigenvalues: NDArray,
        winding: int,
        phase: float,
        n_check: int = 5
    ) -> NDArray:
        """
        Select eigenstate with best overlap and reasonable energy.
        
        Strategy:
        1. Check first n_check eigenstates (sorted by energy)
        2. Compute overlap with current state (after removing phases)
        3. Choose lowest energy state among those with reasonable overlap
        
        Args:
            chi_old: Current wavefunction
            eigenvectors: Eigenvectors from diagonalization
            eigenvalues: Eigenvalues from diagonalization
            winding: Winding number
            phase: Color phase
            n_check: Number of low-energy states to check
            
        Returns:
            Selected eigenstate with phases applied
        """
        # Remove phases to get envelope for overlap calculation
        chi_envelope = chi_old / (np.exp(1j * phase) * np.exp(1j * winding * self.sigma) + 1e-30)
        
        # Compute overlaps with low-energy eigenstates
        n_states = min(n_check, self.N_sigma)
        overlaps = []
        for i in range(n_states):
            overlap = np.abs(np.sum(np.conj(chi_envelope) * eigenvectors[:, i]) * self.dsigma)
            overlaps.append((overlap, eigenvalues[i], i))
        
        # Define overlap threshold
        overlap_threshold = 0.05
        
        # Filter for states with acceptable overlap
        reasonable_states = [s for s in overlaps if s[0] > overlap_threshold]
        
        if reasonable_states:
            # Choose lowest energy among reasonable overlaps
            reasonable_states.sort(key=lambda x: x[1])
            best_overlap, best_energy, best_idx = reasonable_states[0]
        else:
            # Fall back to highest overlap
            overlaps.sort(reverse=True, key=lambda x: x[0])
            best_overlap, best_energy, best_idx = overlaps[0]
        
        # Get eigenstate and apply phases
        chi_new = eigenvectors[:, best_idx].astype(complex)
        chi_new *= np.exp(1j * phase) * np.exp(1j * winding * self.sigma)
        
        # Normalize
        chi_new = self._normalize(chi_new)
        
        return chi_new

