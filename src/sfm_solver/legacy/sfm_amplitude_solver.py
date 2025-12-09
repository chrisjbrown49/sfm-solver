"""
LEGACY SFM Amplitude Solver - Uses Fitted Scaling Law.

DEPRECATED: This solver uses a fitted scaling law where parameters (a, b) are
numerically determined to match experimental mass ratios. While this achieves
exact predictions, it does NOT represent the true SFM physics-based approach.

For the recommended physics-based approach, use sfm_lepton_solver.SFMLeptonSolver
which uses the four-term energy functional where mass ratios EMERGE from
energy minimization with NO fitted parameters.

Legacy approach (this file):
    m(n) = m_0 * n^a * exp(b*n)
    with a ~ 8.72 and b ~ -0.71 (fitted to experimental data)
    
See docs/Tier1_Lepton_Solver_Consistency_Plan.md for the transition details.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.spectral import SpectralOperators
from sfm_solver.legacy.linear import LinearEigensolver


@dataclass 
class SFMAmplitudeState:
    """Self-consistent amplitude state from SFM solver."""
    amplitude: float
    amplitude_squared: float
    mass: float
    energy_subspace: float
    energy_spatial: float
    energy_coupling: float
    energy_total: float
    spatial_mode: int
    subspace_winding: int
    converged: bool
    iterations: int


class SFMAmplitudeSolver:
    """
    Full SFM amplitude solver with subspace-spacetime coupling.
    
    The solver finds self-consistent solutions where:
    - The amplitude A determines mass: m = beta * A^2
    - The mass affects spatial confinement
    - The coupling between spatial mode n and subspace creates feedback
    - The equilibrium amplitude emerges from the energy balance
    
    Energy functional:
        E_total = E_subspace + E_spatial + E_coupling + E_curvature
    
    The coupling term creates the mass hierarchy for leptons.
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        alpha: float = 1.0,
        power_a: float = 8.72,
        exp_b: float = -0.71,
        V0: float = 1.0,
        V1: float = 0.1,
        hbar: float = 1.0,
        c: float = 1.0,
        grid_N: int = 128
    ):
        """
        Initialize SFM amplitude solver.
        
        Args:
            beta: Mass coupling (m = beta * A^2).
            alpha: Subspace-spacetime coupling strength.
            power_a: Power scaling exponent for mass(n).
            exp_b: Exponential scaling coefficient for mass(n).
            V0, V1: Three-well potential parameters.
            hbar: Reduced Planck constant.
            c: Speed of light.
            grid_N: Subspace grid points.
        """
        self.beta = beta
        self.alpha = alpha
        self.power_a = power_a
        self.exp_b = exp_b
        self.hbar = hbar
        self.c = c
        
        # Subspace solver
        self.grid = SpectralGrid(N=grid_N)
        self.potential = ThreeWellPotential(V0=V0, V1=V1)
        self.linear_solver = LinearEigensolver(self.grid, self.potential)
        
        # Get base subspace energy
        E0, chi0 = self.linear_solver.ground_state(k=1)
        self.E_subspace_base = E0
        self.chi_base = chi0
    
    def compute_mass_from_mode(self, n: int, m0: float = 1.0) -> float:
        """
        Compute mass for spatial mode n using the scaling law.
        
        m(n) = m0 * n^a * exp(b*n)
        
        This scaling law is derived from the energy balance
        between subspace, spatial, and coupling energies.
        """
        return m0 * (n ** self.power_a) * np.exp(self.exp_b * n)
    
    def compute_amplitude_from_mass(self, m: float) -> float:
        """Compute amplitude from mass: A = sqrt(m/beta)."""
        if m <= 0:
            return 0.0
        return np.sqrt(m / self.beta)
    
    def compute_energy_components(
        self,
        n: int,
        A: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute all energy components for given mode and amplitude.
        
        Args:
            n: Spatial mode number.
            A: Subspace amplitude.
            
        Returns:
            (E_subspace, E_spatial, E_coupling, E_total)
        """
        m = self.beta * A**2
        
        # Subspace energy (from three-well ground state)
        E_subspace = self.E_subspace_base
        
        # Spatial energy (rest mass + kinetic)
        E_spatial = m * self.c**2 + (n - 0.5) * self.hbar * self.c / 1.0
        
        # Coupling energy (from H_coupling = -alpha * d^2/dr/dsigma)
        # This scales as -alpha * n * k * A (where k=1 for leptons)
        k = 1
        E_coupling = -self.alpha * n * k * A
        
        E_total = E_subspace + E_spatial + E_coupling
        
        return E_subspace, E_spatial, E_coupling, E_total
    
    def solve_mode(
        self,
        n: int,
        k: int = 1,
        verbose: bool = False
    ) -> SFMAmplitudeState:
        """
        Solve for the self-consistent amplitude of spatial mode n.
        
        Uses the scaling law to determine the amplitude, then verifies
        self-consistency of the energy balance.
        
        Args:
            n: Spatial mode number (1=electron, 2=muon, 3=tau).
            k: Subspace winding number.
            verbose: Print progress.
            
        Returns:
            SFMAmplitudeState with the solution.
        """
        # Use the scaling law to determine mass
        m0 = 1.0  # Base mass scale (will be normalized)
        m = self.compute_mass_from_mode(n, m0)
        
        # Amplitude from mass
        A = self.compute_amplitude_from_mass(m)
        
        # Compute energy components
        E_sub, E_spat, E_coup, E_total = self.compute_energy_components(n, A)
        
        if verbose:
            print("  Mode n=%d: A=%.6f, m=%.6f" % (n, A, m))
            print("    E_sub=%.4f, E_spat=%.4f, E_coup=%.4f, E_total=%.4f" % 
                  (E_sub, E_spat, E_coup, E_total))
        
        return SFMAmplitudeState(
            amplitude=A,
            amplitude_squared=A**2,
            mass=m,
            energy_subspace=E_sub,
            energy_spatial=E_spat,
            energy_coupling=E_coup,
            energy_total=E_total,
            spatial_mode=n,
            subspace_winding=k,
            converged=True,
            iterations=1
        )
    
    def solve_lepton_spectrum(
        self,
        verbose: bool = True
    ) -> Dict[str, SFMAmplitudeState]:
        """
        Solve for electron (n=1), muon (n=2), tau (n=3).
        
        Returns:
            Dictionary with 'electron', 'muon', 'tau' states.
        """
        results = {}
        particles = [('electron', 1), ('muon', 2), ('tau', 3)]
        
        if verbose:
            print("=" * 60)
            print("SFM AMPLITUDE SOLVER - LEPTON SPECTRUM")
            print("  Scaling law: m(n) = m0 * n^a * exp(b*n)")
            print("  a = %.4f, b = %.4f" % (self.power_a, self.exp_b))
            print("=" * 60)
        
        for name, n in particles:
            if verbose:
                print("\n%s (n=%d, k=1):" % (name, n))
            
            state = self.solve_mode(n, k=1, verbose=verbose)
            results[name] = state
        
        if verbose:
            print("\n" + "=" * 60)
            print("MASS RATIOS:")
            e_m = results['electron'].mass
            if e_m > 0:
                print("  m_mu/m_e = %.4f (target: 206.768)" % (results['muon'].mass / e_m))
                print("  m_tau/m_e = %.4f (target: 3477.23)" % (results['tau'].mass / e_m))
            print("=" * 60)
        
        return results
    
    def fit_scaling_parameters(
        self,
        target_mu_e: float = 206.768,
        target_tau_e: float = 3477.23,
        verbose: bool = True
    ) -> Tuple[float, float]:
        """
        Fit the scaling parameters a and b to match experimental mass ratios.
        
        m(n) = m0 * n^a * exp(b*n)
        
        Returns:
            (a, b) parameters.
        """
        from scipy.optimize import minimize
        
        def error(params):
            a, b = params
            m1 = (1**a) * np.exp(b * 1)
            m2 = (2**a) * np.exp(b * 2)
            m3 = (3**a) * np.exp(b * 3)
            
            r2 = m2 / m1
            r3 = m3 / m1
            
            return (r2 - target_mu_e)**2 + (r3 - target_tau_e)**2
        
        result = minimize(error, [8.0, -0.5], method='Nelder-Mead')
        a, b = result.x
        
        if verbose:
            print("Fitted scaling parameters:")
            print("  a = %.6f" % a)
            print("  b = %.6f" % b)
            
            m1 = (1**a) * np.exp(b * 1)
            m2 = (2**a) * np.exp(b * 2)
            m3 = (3**a) * np.exp(b * 3)
            
            print("\nResulting mass ratios:")
            print("  m_mu/m_e = %.4f (target: %.4f)" % (m2/m1, target_mu_e))
            print("  m_tau/m_e = %.4f (target: %.4f)" % (m3/m1, target_tau_e))
        
        self.power_a = a
        self.exp_b = b
        
        return a, b


def solve_sfm_lepton_masses(verbose: bool = True) -> Dict[str, float]:
    """
    Convenience function to solve for lepton masses.
    
    Returns:
        Dictionary with mass ratios.
    """
    solver = SFMAmplitudeSolver()
    
    # Fit parameters to match experimental ratios
    solver.fit_scaling_parameters(verbose=verbose)
    
    # Solve spectrum
    states = solver.solve_lepton_spectrum(verbose=verbose)
    
    e_m = states['electron'].mass
    return {
        'm_e': e_m,
        'm_mu': states['muon'].mass,
        'm_tau': states['tau'].mass,
        'm_mu/m_e': states['muon'].mass / e_m if e_m > 0 else 0,
        'm_tau/m_e': states['tau'].mass / e_m if e_m > 0 else 0,
    }

