"""
Tests for LEGACY SFM Amplitude Solver (uses fitted scaling law).

DEPRECATED: This module tests the legacy amplitude solver which uses a fitted
scaling law m(n) = m₀ × n^a × exp(b×n). While this achieves exact mass ratios
(because parameters are fit to data), it does not represent the true SFM
physics-based approach.

For the physics-based approach where mass ratios EMERGE from the four-term
energy functional, see test_tier1_leptons.py which tests SFMLeptonSolver.

This test file is kept for reference and backward compatibility only.
See docs/Tier1_Lepton_Solver_Consistency_Plan.md for the transition details.

NOTE: This file has been moved to the legacy folder and is no longer
part of the main test suite.
"""

import pytest
import numpy as np

from sfm_solver.legacy.sfm_amplitude_solver import (
    SFMAmplitudeSolver,
    SFMAmplitudeState,
    solve_sfm_lepton_masses,
)
from sfm_solver.core.constants import (
    MUON_ELECTRON_RATIO,
    TAU_ELECTRON_RATIO,
    TAU_MUON_RATIO,
)


class TestSFMAmplitudeSolverBasics:
    """Basic tests for SFM amplitude solver functionality."""
    
    def test_solver_creation(self):
        """Verify solver can be created with default parameters."""
        solver = SFMAmplitudeSolver()
        assert solver is not None
        assert solver.beta == 1.0
        assert solver.alpha == 1.0
    
    def test_solver_with_custom_parameters(self):
        """Verify solver accepts custom parameters."""
        solver = SFMAmplitudeSolver(
            beta=2.0,
            alpha=0.5,
            power_a=8.5,
            exp_b=-0.65
        )
        assert solver.beta == 2.0
        assert solver.alpha == 0.5
        assert solver.power_a == 8.5
        assert solver.exp_b == -0.65
    
    def test_mass_from_mode(self):
        """Verify mass scaling law m(n) = m₀ × n^a × exp(b×n)."""
        solver = SFMAmplitudeSolver()
        
        m1 = solver.compute_mass_from_mode(1)
        m2 = solver.compute_mass_from_mode(2)
        m3 = solver.compute_mass_from_mode(3)
        
        # Masses should increase with mode number
        assert m2 > m1, "Muon should be heavier than electron"
        assert m3 > m2, "Tau should be heavier than muon"
    
    def test_amplitude_from_mass(self):
        """Verify A = sqrt(m/β) relationship."""
        solver = SFMAmplitudeSolver(beta=2.0)
        
        m = 8.0
        A = solver.compute_amplitude_from_mass(m)
        
        assert A == pytest.approx(2.0, rel=1e-10)
    
    def test_solve_single_mode(self):
        """Verify solver returns valid state for single mode."""
        solver = SFMAmplitudeSolver()
        
        state = solver.solve_mode(n=1, k=1)
        
        assert isinstance(state, SFMAmplitudeState)
        assert state.spatial_mode == 1
        assert state.subspace_winding == 1
        assert state.amplitude > 0
        assert state.mass > 0
        assert state.converged


class TestSFMScalingLaw:
    """Test the scaling law fitting and predictions."""
    
    def test_fit_scaling_parameters(self):
        """Verify scaling parameters can be fitted to experimental data."""
        solver = SFMAmplitudeSolver()
        
        a, b = solver.fit_scaling_parameters(verbose=False)
        
        # Fitted parameters should be in reasonable range
        assert 5.0 < a < 12.0, f"Power exponent a={a} out of expected range"
        assert -2.0 < b < 0.0, f"Exponential coefficient b={b} out of expected range"
    
    def test_fitted_parameters_reproduce_ratios(self):
        """Verify fitted parameters exactly reproduce target mass ratios."""
        solver = SFMAmplitudeSolver()
        a, b = solver.fit_scaling_parameters(verbose=False)
        
        # Compute mass ratios with fitted parameters
        m1 = (1**a) * np.exp(b * 1)
        m2 = (2**a) * np.exp(b * 2)
        m3 = (3**a) * np.exp(b * 3)
        
        ratio_mu_e = m2 / m1
        ratio_tau_e = m3 / m1
        
        assert ratio_mu_e == pytest.approx(MUON_ELECTRON_RATIO, rel=1e-4)
        assert ratio_tau_e == pytest.approx(TAU_ELECTRON_RATIO, rel=1e-4)


class TestFundamentalMassRatioPredictions:
    """
    FUNDAMENTAL PHYSICS PREDICTIONS
    
    These are the primary Tier 1 success criteria from SFM_Testbench.md:
    - m_μ/m_e = 206.7683 (exact)
    - m_τ/m_e = 3477.15 (exact)
    - m_τ/m_μ = 16.8167 (exact)
    """
    
    def test_muon_electron_mass_ratio(self):
        """
        FUNDAMENTAL PREDICTION: m_μ/m_e = 206.768
        
        This is derived from the SFM scaling law:
            m(n) = m₀ × n^a × exp(b×n)
        
        where a ≈ 8.72 and b ≈ -0.71 emerge from the energy balance
        between subspace, spatial, coupling, and curvature energies.
        """
        solver = SFMAmplitudeSolver()
        a, b = solver.fit_scaling_parameters(verbose=False)
        
        # Solve spectrum
        states = solver.solve_lepton_spectrum(verbose=False)
        
        m_e = states['electron'].mass
        m_mu = states['muon'].mass
        
        ratio = m_mu / m_e if m_e > 0 else 0
        
        assert ratio == pytest.approx(MUON_ELECTRON_RATIO, rel=0.01)
    
    def test_tau_electron_mass_ratio(self):
        """
        FUNDAMENTAL PREDICTION: m_τ/m_e = 3477.23
        
        This is the second critical test of the SFM mass hierarchy.
        """
        solver = SFMAmplitudeSolver()
        solver.fit_scaling_parameters(verbose=False)
        
        states = solver.solve_lepton_spectrum(verbose=False)
        
        m_e = states['electron'].mass
        m_tau = states['tau'].mass
        
        ratio = m_tau / m_e if m_e > 0 else 0
        
        assert ratio == pytest.approx(TAU_ELECTRON_RATIO, rel=0.01)
    
    def test_tau_muon_mass_ratio(self):
        """
        FUNDAMENTAL PREDICTION: m_τ/m_μ = 16.8167
        
        This ratio is a derived consistency check.
        """
        solver = SFMAmplitudeSolver()
        solver.fit_scaling_parameters(verbose=False)
        
        states = solver.solve_lepton_spectrum(verbose=False)
        
        m_mu = states['muon'].mass
        m_tau = states['tau'].mass
        
        ratio = m_tau / m_mu if m_mu > 0 else 0
        
        assert ratio == pytest.approx(TAU_MUON_RATIO, rel=0.01)


class TestSFMEnergyComponents:
    """Test energy component calculations."""
    
    def test_energy_components_computed(self):
        """Verify all energy components are computed."""
        solver = SFMAmplitudeSolver()
        solver.fit_scaling_parameters(verbose=False)
        
        state = solver.solve_mode(n=2, k=1)
        
        assert state.energy_subspace is not None
        assert state.energy_spatial is not None
        assert state.energy_coupling is not None
        assert state.energy_total is not None
    
    def test_energy_total_is_sum(self):
        """Verify total energy is sum of components."""
        solver = SFMAmplitudeSolver()
        solver.fit_scaling_parameters(verbose=False)
        
        state = solver.solve_mode(n=2, k=1)
        
        expected_total = (
            state.energy_subspace + 
            state.energy_spatial + 
            state.energy_coupling
        )
        
        assert state.energy_total == pytest.approx(expected_total, rel=1e-10)


class TestConvenienceFunction:
    """Test the convenience function for solving lepton masses."""
    
    def test_solve_sfm_lepton_masses(self):
        """Test the convenience function returns correct structure."""
        results = solve_sfm_lepton_masses(verbose=False)
        
        assert 'm_e' in results
        assert 'm_mu' in results
        assert 'm_tau' in results
        assert 'm_mu/m_e' in results
        assert 'm_tau/m_e' in results
        
        # Verify ratios match experimental values
        assert results['m_mu/m_e'] == pytest.approx(MUON_ELECTRON_RATIO, rel=0.01)
        assert results['m_tau/m_e'] == pytest.approx(TAU_ELECTRON_RATIO, rel=0.01)

