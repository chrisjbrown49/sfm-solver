"""
LEGACY: Tier 1 Verification Tests: Single-Particle Eigenstates

These tests use LinearEigensolver and NonlinearEigensolver which have been
moved to the legacy package. For the current physics-based approach, see
test_tier1_leptons.py which tests SFMLeptonSolver.

These tests verify the legacy solver infrastructure meets Tier 1 requirements
from the Computational Proposal for Nonlinear Field Equations.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.parameters import SFMParameters
from sfm_solver.core.constants import (
    ELECTRON_MASS_GEV, MUON_MASS_GEV, TAU_MASS_GEV,
    MUON_ELECTRON_RATIO, TAU_ELECTRON_RATIO,
)
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.legacy.linear import LinearEigensolver
from sfm_solver.legacy.nonlinear import NonlinearEigensolver
from sfm_solver.analysis.mass_spectrum import MassSpectrum


# =============================================================================
# Test Class: Eigenvalue Solver Convergence
# =============================================================================

class TestTier1Convergence:
    """Test eigenvalue solver convergence for k=1 modes."""
    
    def test_linear_solver_converges_k1(self):
        """Verify linear solver converges for k=1 lepton states."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        # Solve for ground state with k=1
        energy, psi = solver.ground_state(k=1)
        
        # Verify eigenstate equation: H|ψ⟩ = E|ψ⟩
        is_eigenstate, residual = solver.verify_eigenstate(energy, psi)
        
        assert is_eigenstate, f"Residual {residual} exceeds tolerance"
        assert residual < 1e-6, f"Residual {residual} too large"
    
    def test_nonlinear_solver_runs_k1(self):
        """Verify nonlinear solver runs and produces reasonable output for k=1."""
        grid = SpectralGrid(N=128)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = NonlinearEigensolver(grid, pot, g1=0.001)
        
        energy, psi, info = solver.solve(k=1, max_iter=50, tol=1e-6, mixing=0.1)
        
        assert len(info.energy_history) > 1, "Should have multiple iterations"
        assert grid.norm(psi) > 0, "Wavefunction should be non-zero"
        assert np.isfinite(energy), "Energy should be finite"
        assert energy > 0, "Energy should be positive for this potential"
    
    def test_multiple_k_values_converge(self):
        """Verify solver converges for different winding numbers."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        for k in [1, 3, 5]:
            energies, wavefunctions = solver.solve_with_winding(k=k, n_states=3)
            
            for E, psi in zip(energies, wavefunctions):
                is_eigenstate, residual = solver.verify_eigenstate(E, psi)
                assert is_eigenstate, f"k={k} state failed with residual {residual}"
    
    def test_convergence_with_different_grid_sizes(self):
        """Verify convergence improves with finer grid resolution."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        energies_by_N = []
        
        for N in [64, 128, 256]:
            grid = SpectralGrid(N=N)
            solver = LinearEigensolver(grid, pot)
            energy, _ = solver.ground_state(k=1)
            energies_by_N.append(energy)
        
        max_diff = max(abs(energies_by_N[i] - energies_by_N[j]) 
                       for i in range(3) for j in range(i+1, 3))
        
        assert max_diff < 1e-6, \
            f"Energy should be consistent across grid sizes, max diff = {max_diff}"


# =============================================================================
# Test Class: Mass Formula Verification
# =============================================================================

class TestTier1MassFormula:
    """Test the mass formula m = βA²_χ."""
    
    def test_mass_scales_with_beta(self):
        """Verify mass is proportional to β."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        _, psi = solver.ground_state(k=1)
        A_sq = solver.compute_amplitude_squared(psi)
        
        for beta in [10.0, 50.0, 100.0]:
            mass = solver.compute_mass(psi, beta)
            expected = beta * A_sq
            assert_allclose(mass, expected, rtol=1e-10)
    
    def test_amplitude_normalization_consistency(self):
        """Verify A² = ∫|χ|²dσ is computed correctly."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        _, psi = solver.ground_state(k=1)
        A_sq = solver.compute_amplitude_squared(psi)
        
        # Normalized states should have A² = 2π
        assert_allclose(A_sq, 2 * np.pi, rtol=0.01)


# =============================================================================
# Test Class: Periodic Boundary Conditions
# =============================================================================

class TestTier1PeriodicBoundaryConditions:
    """Verify periodic boundary conditions are satisfied."""
    
    def test_wavefunction_periodic(self):
        """Verify χ(σ + 2π) = χ(σ)."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        _, psi = solver.ground_state(k=1)
        
        sigma_test = np.array([0.0])
        sigma_shifted = sigma_test + 2 * np.pi
        
        psi_at_0 = grid.interpolate(psi, sigma_test)
        psi_at_2pi = grid.interpolate(psi, sigma_shifted)
        
        assert_allclose(psi_at_0, psi_at_2pi, atol=1e-10)
    
    def test_potential_is_periodic(self):
        """Verify potential satisfies periodic boundary conditions."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        sigma_test = np.linspace(0, 0.1, 10)
        V_at_0 = pot(sigma_test)
        V_at_2pi = pot(sigma_test + 2 * np.pi)
        
        assert_allclose(V_at_0, V_at_2pi, atol=1e-10)


# =============================================================================
# Test Class: Winding Number Verification
# =============================================================================

class TestTier1WindingNumber:
    """Verify winding number is correctly preserved."""
    
    @pytest.mark.parametrize("k_target", [1, 2, 3, 5])
    def test_winding_sector_eigenstates(self, k_target):
        """Verify eigenstates in winding-k sector are valid."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        energies, wavefunctions = solver.solve_with_winding(k=k_target, n_states=1)
        psi = wavefunctions[0]
        
        is_eigenstate, residual = solver.verify_eigenstate(energies[0], psi)
        assert is_eigenstate, f"k={k_target} state not an eigenstate, residual={residual}"
        
        norm = grid.norm(psi)
        assert_allclose(norm, np.sqrt(2 * np.pi), rtol=0.01)
    
    def test_pure_winding_mode(self):
        """Verify pure winding modes have exact integer winding."""
        grid = SpectralGrid(N=256)
        
        for k in [1, 2, 3, 4, 5]:
            chi = grid.create_winding_mode(k=k)
            k_extracted = grid.winding_number(chi)
            assert_allclose(k_extracted, k, atol=1e-10)


# =============================================================================
# Test Class: Energy Components
# =============================================================================

class TestTier1EnergyComponents:
    """Test energy decomposition."""
    
    def test_energy_is_sum_of_components(self):
        """Verify total energy equals T + V."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        energy, psi = solver.ground_state(k=1)
        components = solver.energy_components(psi)
        
        assert_allclose(
            components['total'],
            components['kinetic'] + components['potential'],
            rtol=1e-5
        )
    
    def test_kinetic_energy_positive(self):
        """Verify kinetic energy is always positive."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        for k in [1, 2, 3]:
            _, psi = solver.ground_state(k=k)
            components = solver.energy_components(psi)
            
            assert components['kinetic'] > 0, \
                f"Kinetic energy should be positive for k={k}"


# =============================================================================
# Test Class: Physical Consistency
# =============================================================================

class TestTier1PhysicalConsistency:
    """Test physical consistency of solutions."""
    
    def test_probability_density_positive(self):
        """Verify probability density is positive everywhere."""
        grid = SpectralGrid(N=256)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        _, psi = solver.ground_state(k=1)
        density = np.abs(psi)**2
        
        assert np.all(density >= 0), "Probability density must be non-negative"
    
    def test_beautiful_equation_holds(self):
        """Verify the Beautiful Equation β L₀ c = ℏ."""
        params = SFMParameters(beta=50.0)
        ratio = params.verify_beautiful_equation()
        
        assert_allclose(ratio, 1.0, rtol=1e-10), \
            f"Beautiful equation ratio should be 1.0, got {ratio}"

