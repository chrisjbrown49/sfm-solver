"""
Full 2D Coupled Solver for SFM Amplitude Quantization.

Solves the coupled (r, σ) eigenvalue problem with the mixed derivative
coupling term H_coupling = -α ∂²/∂r∂σ that links spatial and subspace gradients.

This coupling is essential for the lepton mass hierarchy:
- Higher spatial modes (n) have larger spatial gradients
- These couple more strongly to subspace
- Creating different self-consistent amplitudes A_n

Mass emerges from: m = β * A²
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class FullCoupledState:
    """Solution from the full 2D coupled solver."""
    psi_2d: NDArray          # Full 2D wavefunction ψ(r, σ)
    amplitude_squared: float  # A² = ∫|χ(σ)|² dσ where χ = marginal over r
    mass: float              # m = β A²
    energy: float            # Total eigenvalue
    chi_marginal: NDArray    # Subspace marginal χ(σ)
    phi_marginal: NDArray    # Spatial marginal φ(r)
    spatial_mode: int        # Radial quantum number
    subspace_winding: int    # Winding number k
    converged: bool
    iterations: int


class FullCoupledSolver:
    """
    Full 2D solver on (r, σ) grid with mixed derivative coupling.
    
    H = H_r + H_σ + H_coupling + H_nonlinear
    
    where:
    H_r = -ℏ²/(2m) [d²/dr² + (2/r)d/dr] + V_r(r)
    H_σ = -ℏ²/(2m_σ R²) d²/dσ² + V(σ)
    H_coupling = -α ∂²/∂r∂σ
    H_nonlinear = g₁ |ψ|²
    """
    
    def __init__(
        self,
        N_r: int = 50,
        N_sigma: int = 64,
        r_max: float = 10.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        g1: float = 0.1,
        V0: float = 1.0,
        V1: float = 0.1,
        hbar: float = 1.0,
        R: float = 1.0
    ):
        """
        Initialize 2D coupled solver.
        
        Args:
            N_r: Radial grid points.
            N_sigma: Subspace grid points.
            r_max: Maximum radius.
            alpha: Coupling strength for mixed derivative ∂²/∂r∂σ.
            beta: Mass coupling (m = β A²).
            g1: Nonlinear coupling.
            V0, V1: Three-well potential parameters.
            hbar: Reduced Planck constant.
            R: Subspace radius.
        """
        self.N_r = N_r
        self.N_sigma = N_sigma
        self.r_max = r_max
        self.alpha = alpha
        self.beta = beta
        self.g1 = g1
        self.V0 = V0
        self.V1 = V1
        self.hbar = hbar
        self.R = R
        
        # Build grids
        self.dr = r_max / (N_r + 1)
        self.r = np.linspace(self.dr, r_max - self.dr, N_r)
        self.dsigma = 2 * np.pi / N_sigma
        self.sigma = np.arange(N_sigma) * self.dsigma
        
        # Total grid size
        self.N_total = N_r * N_sigma
        
        # Precompute operators
        self._build_operators()
    
    def _build_operators(self):
        """Build all operators on the 2D grid."""
        N_r, N_sigma = self.N_r, self.N_sigma
        dr, dsigma = self.dr, self.dsigma
        
        # --- Radial operators ---
        # Second derivative (Dirichlet BC)
        D2_r = np.diag(-2*np.ones(N_r)) + np.diag(np.ones(N_r-1), 1) + np.diag(np.ones(N_r-1), -1)
        D2_r = D2_r / dr**2
        
        # First derivative
        D1_r = np.diag(np.ones(N_r-1), 1) - np.diag(np.ones(N_r-1), -1)
        D1_r = D1_r / (2 * dr)
        
        # Radial kinetic (without mass factor)
        two_over_r = np.diag(2.0 / self.r)
        T_r = -(D2_r + two_over_r @ D1_r)
        
        # --- Subspace operators (periodic BC using FFT-based) ---
        k_modes = np.fft.fftfreq(N_sigma, d=dsigma/(2*np.pi))
        D2_sigma = np.zeros((N_sigma, N_sigma), dtype=complex)
        for i in range(N_sigma):
            for j in range(N_sigma):
                D2_sigma[i, j] = np.sum(-k_modes**2 * np.exp(2j*np.pi*k_modes*(i-j)/N_sigma)) / N_sigma
        D2_sigma = np.real(D2_sigma)
        T_sigma = -D2_sigma / self.R**2
        
        # --- First derivatives for coupling ---
        D1_sigma = np.zeros((N_sigma, N_sigma), dtype=complex)
        for i in range(N_sigma):
            for j in range(N_sigma):
                D1_sigma[i, j] = np.sum(1j*k_modes * np.exp(2j*np.pi*k_modes*(i-j)/N_sigma)) / N_sigma
        D1_sigma = np.real(D1_sigma)
        
        # Store for later
        self.T_r = T_r
        self.T_sigma = T_sigma
        self.D1_r = D1_r
        self.D1_sigma = D1_sigma
        
        # Subspace potential V(σ) = V0[1 - cos(3σ)] + V1[1 - cos(6σ)]
        self.V_sigma = self.V0 * (1 - np.cos(3 * self.sigma)) + self.V1 * (1 - np.cos(6 * self.sigma))
    
    def _build_hamiltonian(self, mass: float, psi_2d: Optional[NDArray] = None) -> NDArray:
        """
        Build full 2D Hamiltonian matrix.
        
        H = H_r ⊗ I_σ + I_r ⊗ H_σ + H_coupling + H_nonlinear
        """
        N_r, N_sigma = self.N_r, self.N_sigma
        N_total = self.N_total
        
        if mass < 1e-10:
            mass = 1e-10
        
        # Kinetic energy: H_r ⊗ I_σ
        H = np.kron(self.hbar**2 / (2 * mass) * self.T_r, np.eye(N_sigma))
        
        # Subspace kinetic: I_r ⊗ H_σ
        H += np.kron(np.eye(N_r), self.hbar**2 / 2 * self.T_sigma)
        
        # Subspace potential: I_r ⊗ V(σ)
        H += np.kron(np.eye(N_r), np.diag(self.V_sigma))
        
        # Radial confining potential (harmonic oscillator)
        omega = mass / self.hbar
        V_r = 0.5 * mass * omega**2 * self.r**2
        H += np.kron(np.diag(V_r), np.eye(N_sigma))
        
        # Coupling term: -α ∂²/∂r∂σ = -α (∂/∂r)(∂/∂σ)
        H_coupling = -self.alpha * np.kron(self.D1_r, self.D1_sigma)
        H += H_coupling
        
        # Nonlinear term: g1 |ψ|²
        if psi_2d is not None:
            density = np.abs(psi_2d)**2
            H += np.diag(self.g1 * density.flatten())
        
        # Ensure Hermitian
        H = 0.5 * (H + np.conj(H.T))
        
        return H
    
    def _reshape_2d(self, psi_flat: NDArray) -> NDArray:
        """Reshape flat vector to 2D (N_r, N_sigma)."""
        return psi_flat.reshape(self.N_r, self.N_sigma)
    
    def _compute_amplitude(self, psi_2d: NDArray) -> float:
        """
        Compute subspace amplitude A² = ∫|χ(σ)|² dσ.
        
        First marginalize over r, then integrate over σ.
        """
        # Marginalize over r: χ(σ) = ∫|ψ(r,σ)|² r² dr
        r_weight = self.r**2 * self.dr * 4 * np.pi
        chi_sq = np.sum(np.abs(psi_2d)**2 * r_weight[:, np.newaxis], axis=0)
        
        # This is |χ(σ)|² - now integrate over σ
        A_sq = np.sum(chi_sq) * self.dsigma
        
        return A_sq
    
    def _get_subspace_marginal(self, psi_2d: NDArray) -> NDArray:
        """Get marginal over r: χ(σ) ∝ √(∫|ψ(r,σ)|² r² dr)."""
        r_weight = self.r**2 * self.dr * 4 * np.pi
        chi_sq = np.sum(np.abs(psi_2d)**2 * r_weight[:, np.newaxis], axis=0)
        return np.sqrt(chi_sq)
    
    def _get_spatial_marginal(self, psi_2d: NDArray) -> NDArray:
        """Get marginal over σ: φ(r) ∝ √(∫|ψ(r,σ)|² dσ)."""
        phi_sq = np.sum(np.abs(psi_2d)**2, axis=1) * self.dsigma
        return np.sqrt(phi_sq)
    
    def _select_mode(
        self,
        energies: NDArray,
        wavefunctions: NDArray,
        target_n: int
    ) -> Tuple[float, NDArray]:
        """
        Select the eigenstate corresponding to target spatial mode n.
        
        Uses node counting in the radial marginal to identify modes.
        """
        best_idx = 0
        best_match = float('inf')
        
        for i in range(min(10, len(energies))):
            psi_2d = self._reshape_2d(wavefunctions[:, i])
            phi = self._get_spatial_marginal(psi_2d)
            
            # Count nodes in radial wavefunction
            phi_norm = phi / np.max(np.abs(phi)) if np.max(np.abs(phi)) > 0 else phi
            sign_changes = np.sum(np.diff(np.sign(phi_norm)) != 0)
            n_nodes = sign_changes // 2
            
            # Match to target (n-1 nodes for n-th mode)
            match = abs(n_nodes - (target_n - 1))
            if match < best_match:
                best_match = match
                best_idx = i
        
        return energies[best_idx], wavefunctions[:, best_idx]
    
    def solve(
        self,
        spatial_mode: int = 1,
        k: int = 1,
        initial_A: float = 1.0,
        max_iter: int = 200,
        tol_A: float = 1e-6,
        tol_E: float = 1e-6,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> FullCoupledState:
        """
        Solve the full 2D coupled eigenvalue problem self-consistently.
        
        Args:
            spatial_mode: Target radial quantum number n (1, 2, 3, ...).
            k: Target subspace winding number.
            initial_A: Initial amplitude guess.
            max_iter: Maximum iterations.
            tol_A: Tolerance for amplitude convergence.
            tol_E: Tolerance for energy convergence.
            mixing: Mixing parameter.
            verbose: Print progress.
            
        Returns:
            FullCoupledState with converged solution.
        """
        # Initialize 2D wavefunction
        # Radial: n-th mode of harmonic oscillator (Gaussian with nodes)
        # Subspace: exp(i k σ) modulated by wells
        phi_init = self.r**(spatial_mode-1) * np.exp(-self.r**2 / 4)
        chi_init = np.exp(1j * k * self.sigma) * np.exp(-((self.sigma - np.pi)**2) / 2)
        
        psi_2d = np.outer(phi_init, chi_init)
        
        # Normalize and scale to initial amplitude
        norm = np.sqrt(np.sum(np.abs(psi_2d)**2) * self.dr * self.dsigma)
        if norm > 1e-15:
            psi_2d = psi_2d / norm
        
        A_sq = self._compute_amplitude(psi_2d)
        if A_sq > 1e-15:
            psi_2d = psi_2d * np.sqrt(initial_A**2 / A_sq)
        
        A_sq = initial_A**2
        mass = self.beta * A_sq
        E_old = 0.0
        converged = False
        
        if verbose:
            print("  Full 2D Coupled Solver: n=%d, k=%d" % (spatial_mode, k))
            print("  Initial: A^2=%.4f, mass=%.4f" % (A_sq, mass))
        
        for iteration in range(max_iter):
            A_sq_old = A_sq
            
            # Build Hamiltonian with current density and mass
            H = self._build_hamiltonian(mass, psi_2d)
            
            # Solve eigenvalue problem
            try:
                energies, wavefunctions = np.linalg.eigh(H)
            except:
                if verbose:
                    print("  Eigenvalue solver failed")
                break
            
            # Select mode corresponding to target spatial quantum number
            E_new, psi_new_flat = self._select_mode(energies, wavefunctions, spatial_mode)
            psi_new_2d = self._reshape_2d(psi_new_flat)
            
            # Preserve amplitude from previous iteration
            A_sq_new = self._compute_amplitude(psi_new_2d)
            if A_sq_new > 1e-15 and A_sq_old > 1e-15:
                psi_new_2d = psi_new_2d * np.sqrt(A_sq_old / A_sq_new)
            
            # Mix for stability
            psi_2d = (1 - mixing) * psi_2d + mixing * psi_new_2d
            
            # Update amplitude and mass
            A_sq = self._compute_amplitude(psi_2d)
            mass = self.beta * A_sq
            
            # Check convergence
            dA = abs(A_sq - A_sq_old)
            dE = abs(E_new - E_old)
            
            if verbose and iteration % 20 == 0:
                print("    Iter %d: A^2=%.6f, m=%.6f, E=%.4f" % (iteration, A_sq, mass, E_new))
            
            if dA < tol_A and dE < tol_E and iteration > 5:
                converged = True
                if verbose:
                    print("    Converged at iteration %d" % (iteration + 1))
                break
            
            E_old = E_new
        
        chi_marginal = self._get_subspace_marginal(psi_2d)
        phi_marginal = self._get_spatial_marginal(psi_2d)
        
        return FullCoupledState(
            psi_2d=psi_2d,
            amplitude_squared=A_sq,
            mass=mass,
            energy=E_new,
            chi_marginal=chi_marginal,
            phi_marginal=phi_marginal,
            spatial_mode=spatial_mode,
            subspace_winding=k,
            converged=converged,
            iterations=iteration + 1
        )
    
    def solve_lepton_spectrum(
        self,
        verbose: bool = True
    ) -> Dict[str, FullCoupledState]:
        """
        Solve for electron (n=1), muon (n=2), tau (n=3).
        """
        results = {}
        particles = [('electron', 1), ('muon', 2), ('tau', 3)]
        
        if verbose:
            print("=" * 60)
            print("FULL 2D COUPLED SOLVER - LEPTON SPECTRUM")
            print("  alpha (coupling) = %.4f" % self.alpha)
            print("  beta (mass) = %.4f" % self.beta)
            print("=" * 60)
        
        for name, n in particles:
            if verbose:
                print("\n%s (n=%d, k=1):" % (name, n))
            
            state = self.solve(
                spatial_mode=n,
                k=1,
                initial_A=float(n),
                verbose=verbose
            )
            results[name] = state
        
        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS:")
            e_m = results['electron'].mass
            if e_m > 0:
                print("  m_mu/m_e = %.4f (target: 206.77)" % (results['muon'].mass / e_m))
                print("  m_tau/m_e = %.4f (target: 3477.23)" % (results['tau'].mass / e_m))
            print("=" * 60)
        
        return results


def scan_coupling_strength(verbose: bool = True) -> Dict[float, Dict[str, float]]:
    """
    Scan different coupling strengths to find mass ratio dependence.
    """
    results = {}
    
    if verbose:
        print("SCANNING COUPLING STRENGTH alpha")
        print("=" * 60)
    
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if verbose:
            print("\nalpha = %.2f:" % alpha)
        
        solver = FullCoupledSolver(
            N_r=30,
            N_sigma=32,
            alpha=alpha,
            beta=1.0
        )
        
        states = solver.solve_lepton_spectrum(verbose=False)
        
        e_m = states['electron'].mass
        if e_m > 0:
            ratio_mu = states['muon'].mass / e_m
            ratio_tau = states['tau'].mass / e_m
        else:
            ratio_mu = ratio_tau = 0
        
        results[alpha] = {
            'ratio_mu_e': ratio_mu,
            'ratio_tau_e': ratio_tau,
            'A_e': states['electron'].amplitude_squared,
            'A_mu': states['muon'].amplitude_squared,
            'A_tau': states['tau'].amplitude_squared,
        }
        
        if verbose:
            print("  m_mu/m_e = %.4f, m_tau/m_e = %.4f" % (ratio_mu, ratio_tau))
    
    return results

