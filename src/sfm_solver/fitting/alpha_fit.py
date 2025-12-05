"""
Fitting routines for the spacetime-subspace coupling constant α.

The coupling constant α in the Hamiltonian:
    Ĥ_coupling = -α (∂²/∂r∂σ)

determines the mass hierarchy between particles with the same subspace
winding number but different spatial quantum numbers. For leptons (k=1):

    Electron (n=1): A²_e with minimal coupling
    Muon (n=2): A²_μ ≈ 207 × A²_e with enhanced coupling  
    Tau (n=3): A²_τ ≈ 3477 × A²_e with further enhanced coupling

The fitting procedure:
1. Solve coupled (r,σ) eigenvalue problem for n=1,2 states
2. Adjust α until A²_μ/A²_e matches experimental m_μ/m_e ≈ 206.768
3. Use fitted α to predict tau mass: m_τ = β × A²_τ(α)

This is a key test of the theory: if the predicted tau mass matches
the experimental value (1776.86 MeV), the framework is validated.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from sfm_solver.spatial.radial import RadialGrid
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.coupled_solver import (
    CoupledGrids, CoupledEigensolver, CoupledSolution, compute_mass_ratios
)
from sfm_solver.core.constants import (
    MUON_ELECTRON_RATIO, TAU_ELECTRON_RATIO,
    ELECTRON_MASS_GEV, MUON_MASS_GEV, TAU_MASS_GEV,
)


@dataclass
class FittingResult:
    """
    Result of parameter fitting.
    
    Attributes:
        alpha: Fitted coupling constant.
        ratio_achieved: Mass ratio achieved with fitted α.
        ratio_target: Target mass ratio.
        error: Relative error in ratio.
        electron_solution: Solution for n=1 state.
        muon_solution: Solution for n=2 state.
        tau_solution: Solution for n=3 state (prediction).
        tau_mass_prediction: Predicted tau mass in MeV.
        tau_mass_experimental: Experimental tau mass in MeV.
        tau_error: Relative error in tau mass prediction.
    """
    alpha: float
    ratio_achieved: float
    ratio_target: float
    error: float
    electron_solution: CoupledSolution
    muon_solution: CoupledSolution
    tau_solution: Optional[CoupledSolution] = None
    tau_mass_prediction: Optional[float] = None
    tau_mass_experimental: float = TAU_MASS_GEV * 1000  # MeV
    tau_error: Optional[float] = None


class LeptonMassFitter:
    """
    Fits the coupling constant α to reproduce the lepton mass hierarchy.
    
    The fitter solves the coupled (r,σ) eigenvalue problem for different
    radial quantum numbers (n=1,2,3 for electron, muon, tau) and adjusts
    α to match the observed muon/electron mass ratio.
    
    Once α is determined, the tau mass becomes a parameter-free prediction.
    
    Attributes:
        grids: CoupledGrids instance.
        potential: Three-well subspace potential.
        g1: Nonlinear coupling constant.
        solver_kwargs: Additional arguments for CoupledEigensolver.
    """
    
    def __init__(
        self,
        N_radial: int = 64,
        r_max: float = 10.0,
        N_subspace: int = 64,
        V0: float = 1.0,
        V1: float = 0.1,
        g1: float = 0.0,
        m_spatial: float = 1.0,
        m_subspace: float = 1.0,
        R_subspace: float = 1.0,
        hbar: float = 1.0
    ):
        """
        Initialize the fitter.
        
        Args:
            N_radial: Number of radial grid points.
            r_max: Maximum radius.
            N_subspace: Number of subspace grid points.
            V0: Primary potential depth.
            V1: Secondary potential depth.
            g1: Nonlinear coupling (0 for linear problem).
            m_spatial: Spatial effective mass.
            m_subspace: Subspace effective mass.
            R_subspace: Subspace radius.
            hbar: Reduced Planck constant.
        """
        self.radial_grid = RadialGrid(N=N_radial, r_max=r_max)
        self.subspace_grid = SpectralGrid(N=N_subspace)
        self.grids = CoupledGrids(
            radial=self.radial_grid,
            subspace=self.subspace_grid
        )
        
        self.potential = ThreeWellPotential(V0=V0, V1=V1)
        self.g1 = g1
        
        self.solver_kwargs = {
            'm_spatial': m_spatial,
            'm_subspace': m_subspace,
            'R_subspace': R_subspace,
            'hbar': hbar,
        }
    
    def compute_ratio(
        self, 
        alpha: float, 
        k: int = 1,
        verbose: bool = False
    ) -> Tuple[float, CoupledSolution, CoupledSolution]:
        """
        Compute the muon/electron amplitude ratio for given α.
        
        Args:
            alpha: Coupling constant to test.
            k: Subspace winding number (1 for leptons).
            verbose: Print progress.
            
        Returns:
            Tuple of (ratio, electron_solution, muon_solution).
        """
        solver = CoupledEigensolver(
            grids=self.grids,
            potential=self.potential,
            alpha=alpha,
            g1=self.g1,
            **self.solver_kwargs
        )
        
        # Solve for electron (n=1)
        if verbose:
            print(f"  Solving for electron (n=1) with α={alpha:.6f}...")
        e_sol = solver.solve(n_radial=1, k_subspace=k, verbose=False)
        
        # Solve for muon (n=2)
        if verbose:
            print(f"  Solving for muon (n=2) with α={alpha:.6f}...")
        mu_sol = solver.solve(n_radial=2, k_subspace=k, verbose=False)
        
        # Compute ratio
        ratio = mu_sol.amplitude_squared / e_sol.amplitude_squared
        
        if verbose:
            print(f"  A²_μ/A²_e = {ratio:.4f} (target: {MUON_ELECTRON_RATIO:.4f})")
        
        return ratio, e_sol, mu_sol
    
    def fit(
        self,
        alpha_min: float = 0.0,
        alpha_max: float = 10.0,
        target_ratio: float = MUON_ELECTRON_RATIO,
        tol: float = 0.01,
        k: int = 1,
        verbose: bool = True
    ) -> FittingResult:
        """
        Fit α to reproduce the target mass ratio.
        
        Uses Brent's method to find α such that A²_μ/A²_e = target_ratio.
        
        Args:
            alpha_min: Minimum α to search.
            alpha_max: Maximum α to search.
            target_ratio: Target ratio m_μ/m_e (default: experimental value).
            tol: Tolerance on ratio (relative).
            k: Subspace winding number.
            verbose: Print progress.
            
        Returns:
            FittingResult with fitted parameters and predictions.
        """
        if verbose:
            print(f"Fitting α to achieve m_μ/m_e = {target_ratio:.4f}")
            print(f"Search range: α ∈ [{alpha_min}, {alpha_max}]")
            print()
        
        # Define objective: ratio - target
        def objective(alpha):
            ratio, _, _ = self.compute_ratio(alpha, k, verbose=False)
            return ratio - target_ratio
        
        # Check bounds
        ratio_min, e_min, mu_min = self.compute_ratio(alpha_min, k, verbose)
        ratio_max, e_max, mu_max = self.compute_ratio(alpha_max, k, verbose)
        
        if verbose:
            print(f"\nα = {alpha_min:.4f}: ratio = {ratio_min:.4f}")
            print(f"α = {alpha_max:.4f}: ratio = {ratio_max:.4f}")
        
        # Check if solution exists in range
        if (ratio_min - target_ratio) * (ratio_max - target_ratio) > 0:
            # No zero crossing, find best approximation
            if verbose:
                print("\nWarning: Target ratio not achievable in search range.")
                print("Finding best approximation...")
            
            result = minimize_scalar(
                lambda a: (self.compute_ratio(a, k, False)[0] - target_ratio)**2,
                bounds=(alpha_min, alpha_max),
                method='bounded'
            )
            alpha_fit = result.x
        else:
            # Find zero crossing
            try:
                alpha_fit = brentq(objective, alpha_min, alpha_max, xtol=tol*target_ratio)
            except ValueError:
                # Fallback to minimization
                result = minimize_scalar(
                    lambda a: (self.compute_ratio(a, k, False)[0] - target_ratio)**2,
                    bounds=(alpha_min, alpha_max),
                    method='bounded'
                )
                alpha_fit = result.x
        
        # Get final solutions
        ratio_fit, e_sol, mu_sol = self.compute_ratio(alpha_fit, k, verbose)
        error = abs(ratio_fit - target_ratio) / target_ratio
        
        if verbose:
            print(f"\nFitted α = {alpha_fit:.6f}")
            print(f"Achieved ratio = {ratio_fit:.4f}")
            print(f"Target ratio = {target_ratio:.4f}")
            print(f"Relative error = {error*100:.2f}%")
        
        return FittingResult(
            alpha=alpha_fit,
            ratio_achieved=ratio_fit,
            ratio_target=target_ratio,
            error=error,
            electron_solution=e_sol,
            muon_solution=mu_sol,
        )
    
    def predict_tau(
        self,
        fitting_result: FittingResult,
        k: int = 1,
        verbose: bool = True
    ) -> FittingResult:
        """
        Predict tau mass using fitted α.
        
        This is the key test of the theory: the tau mass should emerge
        from the same α that produces the muon/electron ratio.
        
        Args:
            fitting_result: Result from fit() containing α.
            k: Subspace winding number.
            verbose: Print progress.
            
        Returns:
            Updated FittingResult with tau prediction.
        """
        alpha = fitting_result.alpha
        
        if verbose:
            print(f"\nPredicting tau mass with α = {alpha:.6f}")
        
        solver = CoupledEigensolver(
            grids=self.grids,
            potential=self.potential,
            alpha=alpha,
            g1=self.g1,
            **self.solver_kwargs
        )
        
        # Solve for tau (n=3)
        tau_sol = solver.solve(n_radial=3, k_subspace=k, verbose=False)
        
        # Compute tau/electron ratio
        tau_e_ratio = tau_sol.amplitude_squared / fitting_result.electron_solution.amplitude_squared
        
        # Predict tau mass from ratio
        # m_τ = m_e × (A²_τ/A²_e)
        m_e_MeV = ELECTRON_MASS_GEV * 1000
        m_tau_pred = m_e_MeV * tau_e_ratio
        m_tau_exp = TAU_MASS_GEV * 1000
        tau_error = abs(m_tau_pred - m_tau_exp) / m_tau_exp
        
        if verbose:
            print(f"A²_τ/A²_e = {tau_e_ratio:.2f} (experimental: {TAU_ELECTRON_RATIO:.2f})")
            print(f"Predicted m_τ = {m_tau_pred:.2f} MeV")
            print(f"Experimental m_τ = {m_tau_exp:.2f} MeV")
            print(f"Prediction error = {tau_error*100:.1f}%")
        
        # Update result
        fitting_result.tau_solution = tau_sol
        fitting_result.tau_mass_prediction = m_tau_pred
        fitting_result.tau_error = tau_error
        
        return fitting_result


def fit_alpha_to_mass_ratio(
    target_ratio: float = MUON_ELECTRON_RATIO,
    alpha_min: float = 0.0,
    alpha_max: float = 10.0,
    N_radial: int = 64,
    N_subspace: int = 64,
    **kwargs
) -> FittingResult:
    """
    Convenience function to fit α and predict tau mass.
    
    Args:
        target_ratio: Target m_μ/m_e ratio.
        alpha_min: Minimum α to search.
        alpha_max: Maximum α to search.
        N_radial: Radial grid points.
        N_subspace: Subspace grid points.
        **kwargs: Additional arguments for LeptonMassFitter.
        
    Returns:
        FittingResult with α and tau prediction.
    """
    fitter = LeptonMassFitter(
        N_radial=N_radial,
        N_subspace=N_subspace,
        **kwargs
    )
    
    result = fitter.fit(
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        target_ratio=target_ratio
    )
    
    result = fitter.predict_tau(result)
    
    return result


def predict_tau_mass(
    alpha: float,
    N_radial: int = 64,
    N_subspace: int = 64,
    **kwargs
) -> Dict[str, float]:
    """
    Predict tau mass for given α.
    
    Args:
        alpha: Coupling constant.
        N_radial: Radial grid points.
        N_subspace: Subspace grid points.
        **kwargs: Additional arguments.
        
    Returns:
        Dictionary with mass ratios and predictions.
    """
    fitter = LeptonMassFitter(
        N_radial=N_radial,
        N_subspace=N_subspace,
        **kwargs
    )
    
    # Compute all three states
    solver = CoupledEigensolver(
        grids=fitter.grids,
        potential=fitter.potential,
        alpha=alpha,
        g1=fitter.g1,
        **fitter.solver_kwargs
    )
    
    solutions = solver.solve_spectrum(n_radial_values=[1, 2, 3])
    ratios = compute_mass_ratios(solutions)
    
    # Add mass predictions
    m_e = ELECTRON_MASS_GEV * 1000
    ratios['m_e_MeV'] = m_e
    ratios['m_mu_pred_MeV'] = m_e * ratios.get('mu_e_ratio', 1.0)
    ratios['m_tau_pred_MeV'] = m_e * ratios.get('tau_e_ratio', 1.0)
    ratios['m_mu_exp_MeV'] = MUON_MASS_GEV * 1000
    ratios['m_tau_exp_MeV'] = TAU_MASS_GEV * 1000
    
    return ratios

