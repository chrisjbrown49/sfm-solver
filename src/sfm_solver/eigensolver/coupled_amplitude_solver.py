"""
Coupled Subspace-Spatial Amplitude Eigensolver for SFM.

Solves the full coupled 5D problem where amplitude A_chi emerges from
equilibrium between subspace energy and spatial energy.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators


@dataclass
class CoupledAmplitudeState:
    """Self-consistent coupled state."""
    chi: NDArray            # Subspace wavefunction (NOT normalized)
    phi: NDArray            # Spatial wavefunction (normalized)
    amplitude_squared: float  # A^2 = integral|chi|^2
    mass: float             # m = beta * A^2
    energy_subspace: float
    energy_spatial: float
    energy_total: float
    spatial_radius: float
    spatial_mode: int
    converged: bool
    iterations: int
    history: Dict[str, List[float]]


class SpatialGrid:
    """Radial grid for spatial dimension."""
    
    def __init__(self, N: int = 100, r_max: float = 10.0):
        self.N = N
        self.r_max = r_max
        self.dr = r_max / (N + 1)
        self.r = np.linspace(self.dr, r_max - self.dr, N)
    
    def integrate(self, f: NDArray) -> float:
        return np.trapezoid(f * self.r**2, self.r) * 4 * np.pi


class CoupledAmplitudeSolver:
    """
    Coupled subspace-spatial eigensolver for amplitude quantization.
    
    Different spatial modes (n=1,2,3) with same subspace winding (k=1)
    give different self-consistent amplitudes via energy balance.
    """
    
    def __init__(
        self,
        subspace_grid: SpectralGrid,
        potential: ThreeWellPotential,
        spatial_N: int = 100,
        spatial_r_max: float = 20.0,
        beta: float = 1.0,
        g1: float = 0.1,
        hbar: float = 1.0,
        c: float = 1.0
    ):
        self.subspace_grid = subspace_grid
        self.potential = potential
        self.spatial_grid = SpatialGrid(spatial_N, spatial_r_max)
        self.beta = beta
        self.g1 = g1
        self.hbar = hbar
        self.c = c
        
        self.subspace_ops = SpectralOperators(subspace_grid, m_eff=1.0, hbar=hbar)
        self._V_subspace = potential(subspace_grid.sigma)
        self._build_spatial_kinetic()
    
    def _build_spatial_kinetic(self):
        """Build spatial kinetic operator (without 1/2m factor)."""
        N = self.spatial_grid.N
        dr = self.spatial_grid.dr
        r = self.spatial_grid.r
        
        diag = -2.0 * np.ones(N)
        off_diag = np.ones(N - 1)
        D2 = (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dr**2
        D1 = (np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1), -1)) / (2 * dr)
        two_over_r = np.diag(2.0 / r)
        self._T_spatial_base = -self.hbar**2 * (D2 + two_over_r @ D1)
    
    def _build_spatial_hamiltonian(self, mass: float) -> NDArray:
        """Build spatial Hamiltonian with harmonic confinement."""
        if mass < 1e-10:
            mass = 1e-10
        
        H = self._T_spatial_base / (2 * mass)
        omega = mass * self.c**2 / self.hbar
        V = 0.5 * mass * omega**2 * self.spatial_grid.r**2
        H = H + np.diag(V)
        return 0.5 * (H + H.T)
    
    def _solve_subspace_step(self, chi: NDArray, k: int = 1) -> Tuple[NDArray, float, float]:
        """Solve subspace step, preserving amplitude."""
        A_sq_current = np.real(self.subspace_grid.integrate(np.abs(chi)**2))
        
        V_eff = self._V_subspace + self.g1 * np.abs(chi)**2
        H = self.subspace_ops.build_hamiltonian_matrix(V_eff)
        H = 0.5 * (H + np.conj(H.T))
        
        energies, wavefunctions = np.linalg.eigh(H)
        
        chi_new = wavefunctions[:, 0]
        E_sigma = energies[0]
        for i in range(len(energies)):
            psi = wavefunctions[:, i]
            winding = self.subspace_grid.winding_number(psi)
            if abs(winding - k) < 0.5:
                chi_new = psi
                E_sigma = energies[i]
                break
        
        # CRITICAL: Scale to preserve amplitude
        chi_new_norm = np.real(self.subspace_grid.integrate(np.abs(chi_new)**2))
        if chi_new_norm > 1e-15 and A_sq_current > 1e-15:
            chi_new = chi_new * np.sqrt(A_sq_current / chi_new_norm)
        
        A_sq = np.real(self.subspace_grid.integrate(np.abs(chi_new)**2))
        return chi_new, A_sq, E_sigma
    
    def _solve_spatial_step(self, mass: float, n: int = 1) -> Tuple[NDArray, float, float]:
        """Solve spatial step for given mass."""
        H = self._build_spatial_hamiltonian(mass)
        energies, wavefunctions = np.linalg.eigh(H)
        
        idx = min(n - 1, len(energies) - 1)
        phi = wavefunctions[:, idx]
        E_spatial = energies[idx]
        
        norm = np.sqrt(self.spatial_grid.integrate(np.abs(phi)**2))
        if norm > 1e-15:
            phi = phi / norm
        
        r2_avg = self.spatial_grid.integrate(np.abs(phi)**2 * self.spatial_grid.r**2)
        radius = np.sqrt(r2_avg)
        return phi, E_spatial, radius
    
    def _compute_amplitude_feedback(self, phi: NDArray, n: int) -> float:
        """
        Compute amplitude adjustment from spatial solution.
        
        The key insight: spatial mode n constrains subspace amplitude.
        Higher n modes have larger spatial extent, requiring larger amplitude.
        """
        # Spatial expectation: larger radius -> larger A needed
        r2_avg = self.spatial_grid.integrate(np.abs(phi)**2 * self.spatial_grid.r**2)
        
        # For harmonic oscillator: <r^2> ~ (2n+1) * hbar / (m*omega)
        # Larger n gives larger <r^2>
        # The mass m = beta*A^2, so A^2 ~ 1 / <r^2> for fixed total energy
        # But also E ~ m*omega ~ m^2 ~ A^4
        # Energy balance: E_spatial ~ n * omega ~ n * m ~ n * A^2
        # This creates self-consistency: A^2 must match spatial mode
        
        return r2_avg
    
    def solve_coupled(
        self,
        spatial_mode: int = 1,
        k: int = 1,
        initial_A: float = 1.0,
        max_iter: int = 500,
        tol_A: float = 1e-8,
        tol_E: float = 1e-8,
        mixing: float = 0.3,
        verbose: bool = False
    ) -> CoupledAmplitudeState:
        """Solve coupled subspace-spatial eigenvalue problem."""
        
        # Initialize subspace wavefunction with amplitude
        chi = np.exp(1j * k * self.subspace_grid.sigma)
        chi = chi * np.exp(-((self.subspace_grid.sigma - np.pi)**2) / 2.0)
        A_sq = np.real(self.subspace_grid.integrate(np.abs(chi)**2))
        if A_sq > 1e-15:
            chi = chi * np.sqrt(initial_A**2 / A_sq)
        
        A_sq = initial_A**2
        mass = self.beta * A_sq
        
        # Initialize spatial wavefunction
        phi = np.exp(-self.spatial_grid.r / 1.0)
        norm = np.sqrt(self.spatial_grid.integrate(np.abs(phi)**2))
        phi = phi / norm
        
        history = {'A_squared': [A_sq], 'mass': [mass], 'E_total': []}
        E_total_old = 0.0
        converged = False
        
        if verbose:
            print("  Spatial mode n=%d, initial A^2=%.4f" % (spatial_mode, A_sq))
        
        for iteration in range(max_iter):
            A_sq_old = A_sq
            
            # SUBSPACE STEP
            chi_new, A_sq_sub, E_sigma = self._solve_subspace_step(chi, k)
            chi = (1 - mixing) * chi + mixing * chi_new
            
            # SPATIAL STEP with current mass
            mass = self.beta * A_sq
            phi, E_spatial, radius = self._solve_spatial_step(mass, n=spatial_mode)
            
            # FEEDBACK: Spatial mode constrains amplitude
            # Higher modes need more amplitude for energy balance
            r2 = self._compute_amplitude_feedback(phi, spatial_mode)
            
            # Adjust amplitude based on spatial feedback
            # The idea: A^2 should scale with spatial mode
            # For harmonic oscillator: E_n = (n + 1/2) * hbar * omega
            # With omega = m*c^2/hbar and m = beta*A^2:
            # E_n = (n + 1/2) * beta * A^2 * c^2
            # Total energy balance requires specific A for each n
            target_A_sq = A_sq * (1 + 0.01 * (spatial_mode - 1) * (r2 - 1.0))
            target_A_sq = max(0.01, target_A_sq)
            
            # Mix amplitude
            A_sq = (1 - mixing) * A_sq + mixing * target_A_sq
            
            # Scale chi to new amplitude
            chi_norm = np.real(self.subspace_grid.integrate(np.abs(chi)**2))
            if chi_norm > 1e-15:
                chi = chi * np.sqrt(A_sq / chi_norm)
            
            # Total energy
            E_coupling = mass * self.c**2
            E_total = E_sigma + E_spatial + E_coupling
            
            history['A_squared'].append(A_sq)
            history['mass'].append(mass)
            history['E_total'].append(E_total)
            
            dA = abs(A_sq - A_sq_old)
            dE = abs(E_total - E_total_old)
            
            if verbose and iteration % 50 == 0:
                print("    Iter %d: A^2=%.4f, m=%.4f, r=%.4f" % (iteration, A_sq, mass, radius))
            
            if dA < tol_A and dE < tol_E and iteration > 10:
                converged = True
                if verbose:
                    print("    Converged at iteration %d" % (iteration + 1))
                break
            
            E_total_old = E_total
        
        return CoupledAmplitudeState(
            chi=chi, phi=phi, amplitude_squared=A_sq, mass=mass,
            energy_subspace=E_sigma, energy_spatial=E_spatial,
            energy_total=E_total, spatial_radius=radius,
            spatial_mode=spatial_mode, converged=converged,
            iterations=iteration + 1, history=history
        )
    
    def solve_lepton_spectrum(self, verbose: bool = True) -> Dict[str, CoupledAmplitudeState]:
        """Solve for electron (n=1), muon (n=2), tau (n=3)."""
        results = {}
        particles = [('electron', 1), ('muon', 2), ('tau', 3)]
        
        if verbose:
            print("=" * 50)
            print("SOLVING LEPTON SPECTRUM (Coupled Amplitude)")
            print("=" * 50)
        
        for name, n in particles:
            if verbose:
                print("\n%s (n=%d):" % (name, n))
            state = self.solve_coupled(spatial_mode=n, k=1, initial_A=float(n), verbose=verbose)
            results[name] = state
        
        if verbose:
            print("\n" + "=" * 50)
            print("RESULTS:")
            e_m = results['electron'].mass
            if e_m > 0:
                print("  m_mu/m_e = %.4f (target: 206.77)" % (results['muon'].mass / e_m))
                print("  m_tau/m_e = %.4f (target: 3477.23)" % (results['tau'].mass / e_m))
        
        return results

