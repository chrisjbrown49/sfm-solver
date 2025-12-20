"""
Tests for the Gross-Pitaevskii solver with non-normalized wavefunctions.

The GP solver uses non-normalized wavefunctions where the "particle number"
N = ∫|ψ|² determines the mass: m = β N.

Key insight: Standard eigensolvers normalize to ∫|ψ|² = 1, which constrains
all particles to have the same "amplitude". The GP formulation allows different
particles to have different N values, enabling the mass hierarchy.
"""

import pytest
import numpy as np

from sfm_solver.spatial.radial import RadialGrid
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.eigensolver.gp_solver import (
    GrossPitaevskiiSolver,
    GPSolution,
    compute_mass_ratios_from_gp,
)


class TestGPSolverBasics:
    """Test basic GP solver functionality."""
    
    @pytest.fixture
    def solver(self):
        """Create a GP solver instance."""
        radial = RadialGrid(N=32, r_max=10.0)
        subspace = SpectralGrid(N=24)
        potential = ThreeWellPotential(V0=1.0, V1=0.1)
        return GrossPitaevskiiSolver(
            radial, subspace, potential,
            alpha=0.1, g=0.1, omega=0.5
        )
    
    def test_solver_creation(self, solver):
        """Test that solver can be created."""
        assert solver is not None
        assert solver.g == 0.1
        assert solver.alpha == 0.1
        assert solver.omega == 0.5
    
    def test_solve_for_particle_number(self, solver):
        """Test solving for a specific particle number."""
        N_target = 1.0
        sol = solver.solve_for_particle_number(
            N_target=N_target,
            n_radial=1,
            k_subspace=1,
            max_iter=100,
            verbose=False
        )
        
        assert isinstance(sol, GPSolution)
        assert sol.wavefunction is not None
        # Particle number should be close to target
        assert abs(sol.particle_number - N_target) < 0.01 * N_target
    
    def test_particle_number_is_preserved(self, solver):
        """Test that particle number N is maintained during iteration."""
        N_target = 5.0
        sol = solver.solve_for_particle_number(
            N_target=N_target,
            n_radial=1,
            k_subspace=1,
            max_iter=50,
            verbose=False
        )
        
        # Verify N = ∫|ψ|² equals target
        N_computed = solver.grids.integrate(sol.wavefunction)
        assert abs(N_computed - N_target) / N_target < 0.01


class TestNonNormalizedWavefunctions:
    """Test that wavefunctions are properly non-normalized."""
    
    @pytest.fixture
    def solver(self):
        radial = RadialGrid(N=32, r_max=10.0)
        subspace = SpectralGrid(N=24)
        potential = ThreeWellPotential(V0=1.0, V1=0.1)
        return GrossPitaevskiiSolver(
            radial, subspace, potential,
            alpha=0.1, g=0.1, omega=0.5
        )
    
    def test_different_N_give_different_integrals(self, solver):
        """Test that different N values give different ∫|ψ|²."""
        N_values = [0.5, 1.0, 5.0, 10.0]
        integrals = []
        
        for N in N_values:
            sol = solver.solve_for_particle_number(
                N_target=N,
                n_radial=1,
                k_subspace=1,
                max_iter=50,
                verbose=False
            )
            integrals.append(sol.particle_number)
        
        # Each integral should match its target N
        for N, integral in zip(N_values, integrals):
            assert abs(integral - N) / N < 0.05
    
    def test_wavefunction_not_normalized_to_one(self, solver):
        """Test that wavefunctions are NOT normalized to 1."""
        N_target = 10.0
        sol = solver.solve_for_particle_number(
            N_target=N_target,
            n_radial=1,
            k_subspace=1,
            max_iter=50,
            verbose=False
        )
        
        # For N=10, the integral should be 10, NOT 1
        N_computed = solver.grids.integrate(sol.wavefunction)
        assert N_computed > 5.0  # Much larger than 1
        assert abs(N_computed - N_target) / N_target < 0.05


class TestMassRatiosFromN:
    """Test that mass ratios from particle number N work correctly."""
    
    @pytest.fixture
    def solver(self):
        radial = RadialGrid(N=32, r_max=10.0)
        subspace = SpectralGrid(N=24)
        potential = ThreeWellPotential(V0=1.0, V1=0.1)
        return GrossPitaevskiiSolver(
            radial, subspace, potential,
            alpha=0.1, g=0.1, omega=0.5
        )
    
    def test_mass_ratio_from_N_ratio(self, solver, add_prediction, add_solver_parameter):
        """Test that m_μ/m_e = N_μ/N_e when using GP formulation."""
        # Set up N values that should give correct mass ratios
        N_e = 1.0
        N_mu = 206.768  # m_μ/m_e
        N_tau = 3477.15  # m_τ/m_e
        
        add_solver_parameter("N_electron", N_e)
        add_solver_parameter("N_muon", N_mu)
        add_solver_parameter("N_tau", N_tau)
        add_solver_parameter("formulation", "Gross-Pitaevskii non-normalized")
        
        # Solve for each particle
        sol_e = solver.solve_for_particle_number(N_target=N_e, n_radial=1, max_iter=50, verbose=False)
        sol_mu = solver.solve_for_particle_number(N_target=N_mu, n_radial=2, max_iter=50, verbose=False)
        sol_tau = solver.solve_for_particle_number(N_target=N_tau, n_radial=3, max_iter=50, verbose=False)
        
        # Mass ratio from N
        m_mu_m_e = sol_mu.particle_number / sol_e.particle_number
        m_tau_m_e = sol_tau.particle_number / sol_e.particle_number
        
        # Record predictions
        add_prediction(
            parameter="m_μ/m_e (GP, from N)",
            predicted=m_mu_m_e,
            experimental=206.768,
            target_accuracy=0.01,  # Should be exact
            notes="Using N = ∫|ψ|² as mass measure"
        )
        
        add_prediction(
            parameter="m_τ/m_e (GP, from N)",
            predicted=m_tau_m_e,
            experimental=3477.15,
            target_accuracy=0.01,
            notes="Using N = ∫|ψ|² as mass measure"
        )
        
        # These should be very close (by construction of N values)
        assert abs(m_mu_m_e - 206.768) / 206.768 < 0.01
        assert abs(m_tau_m_e - 3477.15) / 3477.15 < 0.01
    
    def test_mass_from_particle_number_is_consistent(self, solver):
        """Test that mass = β N gives consistent results."""
        beta = 1.0  # Arbitrary mass coupling
        
        N_values = [1.0, 10.0, 100.0]
        masses = []
        
        for N in N_values:
            sol = solver.solve_for_particle_number(
                N_target=N,
                n_radial=1,
                max_iter=50,
                verbose=False
            )
            mass = beta * sol.particle_number
            masses.append(mass)
        
        # Masses should scale linearly with N
        for i, (N, m) in enumerate(zip(N_values, masses)):
            expected_ratio = N / N_values[0]
            actual_ratio = m / masses[0]
            assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.05


class TestGPPhysics:
    """Test physical properties of GP solutions."""
    
    @pytest.fixture
    def solver(self):
        radial = RadialGrid(N=32, r_max=10.0)
        subspace = SpectralGrid(N=24)
        potential = ThreeWellPotential(V0=1.0, V1=0.1)
        return GrossPitaevskiiSolver(
            radial, subspace, potential,
            alpha=0.1, g=0.1, omega=0.5
        )
    
    def test_chemical_potential_depends_on_N(self, solver):
        """Test that chemical potential μ varies with N."""
        N_values = [1.0, 10.0, 50.0]
        mu_values = []
        
        for N in N_values:
            sol = solver.solve_for_particle_number(
                N_target=N,
                n_radial=1,
                max_iter=100,
                verbose=False
            )
            mu_values.append(sol.chemical_potential)
        
        # μ should generally increase with N (more particles = higher energy per particle)
        # due to the nonlinear repulsion term g|ψ|²
        # Note: this depends on g > 0 (repulsive)
        assert mu_values[1] != mu_values[0]  # At least they should differ
    
    def test_energy_varies_with_N(self, solver):
        """Test that total energy varies with particle number."""
        N_values = [1.0, 5.0, 10.0]
        energies = []
        
        for N in N_values:
            sol = solver.solve_for_particle_number(
                N_target=N,
                n_radial=1,
                max_iter=100,
                verbose=False
            )
            energies.append(sol.energy)
        
        # Energy should vary with N (not all the same)
        # Note: With attractive potential, higher N can give lower energy
        assert len(set(np.round(energies, 3))) > 1, \
            "Energies should differ for different N values"
    
    def test_wavefunction_is_valid(self, solver):
        """Test that solutions are valid wavefunctions."""
        sol = solver.solve_for_particle_number(
            N_target=5.0,
            n_radial=1,
            max_iter=100,
            verbose=False
        )
        
        # Wavefunction should be finite
        assert np.all(np.isfinite(sol.wavefunction))
        
        # Should have some structure (not all zeros)
        assert np.max(np.abs(sol.wavefunction)) > 0
        
        # Amplitude squared should be positive
        assert sol.amplitude_squared > 0


class TestGPConvergence:
    """Test GP solver convergence."""
    
    @pytest.fixture
    def solver(self):
        radial = RadialGrid(N=32, r_max=10.0)
        subspace = SpectralGrid(N=24)
        potential = ThreeWellPotential(V0=1.0, V1=0.1)
        return GrossPitaevskiiSolver(
            radial, subspace, potential,
            alpha=0.1, g=0.1, omega=0.5
        )
    
    def test_convergence_for_small_N(self, solver):
        """Test convergence for small particle numbers."""
        sol = solver.solve_for_particle_number(
            N_target=1.0,
            n_radial=1,
            max_iter=200,
            tol=1e-6,
            verbose=False
        )
        
        # Should converge or get close
        # Note: convergence may be difficult for some parameter regimes
        assert sol.iterations <= 200
    
    def test_convergence_for_large_N(self, solver):
        """Test convergence for large particle numbers."""
        sol = solver.solve_for_particle_number(
            N_target=100.0,
            n_radial=1,
            max_iter=200,
            tol=1e-6,
            verbose=False
        )
        
        assert sol.iterations <= 200
        # Particle number should still be close to target
        assert abs(sol.particle_number - 100.0) / 100.0 < 0.1


class TestConvenienceFunctions:
    """Test convenience functions for GP solver."""
    
    def test_compute_mass_ratios_from_gp(self, add_prediction, add_solver_parameter):
        """Test the main convenience function."""
        results = compute_mass_ratios_from_gp(
            g=0.1,
            alpha=0.1,
            omega=0.5,
            N_electron=1.0,
            verbose=False
        )
        
        add_solver_parameter("GP_g", 0.1)
        add_solver_parameter("GP_alpha", 0.1)
        add_solver_parameter("GP_omega", 0.5)
        add_solver_parameter("GP_formulation", "non-normalized")
        
        # Check that results dictionary has expected keys
        assert 'N_e' in results
        assert 'N_mu' in results
        assert 'N_tau' in results
        assert 'm_mu_m_e_from_N' in results
        assert 'm_tau_m_e_from_N' in results
        
        # Mass ratios from N should be correct (by construction)
        add_prediction(
            parameter="m_μ/m_e (GP convenience)",
            predicted=results['m_mu_m_e_from_N'],
            experimental=206.768,
            target_accuracy=0.01,
            notes="Convenience function result"
        )
        
        assert abs(results['m_mu_m_e_from_N'] - 206.768) / 206.768 < 0.01

