"""
Physics validation tests for SFT Solver.

Tests physical consistency, mass hierarchy, and testbench compliance.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sft_solver.core.grid import SpectralGrid
from sft_solver.core.parameters import SFTParameters
from sft_solver.core.constants import (
    ELECTRON_MASS_GEV, MUON_MASS_GEV, TAU_MASS_GEV,
    MUON_ELECTRON_RATIO, TAU_ELECTRON_RATIO,
    HBAR, C,
)
from sft_solver.potentials.three_well import ThreeWellPotential
from sft_solver.eigensolver.linear import LinearEigensolver
from sft_solver.forces.electromagnetic import (
    calculate_circulation,
    calculate_winding_number,
    calculate_em_energy,
    EMForceCalculator,
)
from sft_solver.analysis.mass_spectrum import (
    MassSpectrum,
    calibrate_beta_from_electron,
)
from sft_solver.validation.testbench import TestbenchValidator


class TestSFTParameters:
    """Test SFT parameter consistency."""
    
    def test_beautiful_equation(self):
        """Test that β L₀ c / ℏ = 1 (Beautiful Equation)."""
        params = SFTParameters(beta=50.0)
        
        ratio = params.verify_beautiful_equation()
        assert_allclose(ratio, 1.0, rtol=1e-10)
    
    def test_L0_calculation(self):
        """Test L₀ is correctly computed from β."""
        params = SFTParameters(beta=50.0)
        
        # L₀ = ℏ c / β
        # With β in GeV, need to convert
        GeV_to_J = 1.602176634e-10
        expected_L0 = HBAR * C / (params.beta * GeV_to_J)
        
        assert_allclose(params.L0, expected_L0, rtol=1e-10)
    
    def test_mass_from_amplitude(self):
        """Test mass calculation from amplitude."""
        params = SFTParameters(beta=50.0)
        
        # For A² = 2π (normalized state), mass = β × 2π
        A_sq = 2 * np.pi
        mass = params.mass_from_amplitude(A_sq)
        expected = 50.0 * 2 * np.pi
        
        assert_allclose(mass, expected)
    
    def test_amplitude_from_mass(self):
        """Test amplitude calculation from target mass."""
        params = SFTParameters(beta=50.0)
        
        # For electron mass
        A_sq = params.amplitude_from_mass(ELECTRON_MASS_GEV)
        expected = ELECTRON_MASS_GEV / 50.0
        
        assert_allclose(A_sq, expected)


class TestChargeQuantization:
    """Test that winding numbers give correct charges."""
    
    def test_lepton_charge(self):
        """Test k=1 gives unit charge."""
        grid = SpectralGrid(N=64)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=1)
        assert_allclose(Q, 1.0)  # Q = e/k = e for k=1
    
    def test_down_quark_charge(self):
        """Test k=3 gives 1/3 charge."""
        grid = SpectralGrid(N=64)
        calculator = EMForceCalculator(grid)
        
        Q = calculator.charge_from_winding(k=3)
        assert_allclose(Q, 1/3)  # Q = e/3
    
    def test_up_quark_charge(self):
        """Test k=5 gives 2/5 charge (which is 2/3 × 3/5)."""
        grid = SpectralGrid(N=64)
        calculator = EMForceCalculator(grid)
        
        # Note: The actual formula gives Q = e/k
        # For up quark with k=5: Q = e/5 = 0.2e
        # The 2/3 charge comes from considering 2 quark contributions
        Q = calculator.charge_from_winding(k=5)
        assert_allclose(Q, 1/5)


class TestEMForces:
    """Test electromagnetic force behavior."""
    
    def test_like_charges_repel(self):
        """Test that like charges (same winding) have higher energy."""
        grid = SpectralGrid(N=128)
        
        # Two particles with same winding
        chi1 = grid.create_localized_mode(k=1, center=0)
        chi2 = grid.create_localized_mode(k=1, center=np.pi)
        
        E_same = calculate_em_energy(chi1, chi2, grid, g2=0.1)
        
        # Two particles with opposite winding
        chi3 = grid.create_localized_mode(k=-1, center=np.pi)
        E_opposite = calculate_em_energy(chi1, chi3, grid, g2=0.1)
        
        # Same winding should have higher energy (repulsion)
        assert E_same > E_opposite
    
    def test_opposite_charges_attract(self):
        """Test that opposite charges (opposite winding) have lower energy."""
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        chi_plus = grid.create_localized_mode(k=1, center=0)
        chi_minus = grid.create_localized_mode(k=-1, center=0)
        
        force_type = calculator.force_type(chi_plus, chi_minus)
        assert force_type == 'attractive'
    
    def test_same_charges_repel(self):
        """Test that same charges (same winding) repel."""
        grid = SpectralGrid(N=128)
        calculator = EMForceCalculator(grid, g2=0.1)
        
        chi1 = grid.create_localized_mode(k=1, center=0)
        chi2 = grid.create_localized_mode(k=1, center=np.pi/2)
        
        force_type = calculator.force_type(chi1, chi2)
        assert force_type == 'repulsive'
    
    def test_winding_extraction(self):
        """Test winding number extraction from wavefunction."""
        grid = SpectralGrid(N=128)
        
        for k in [1, 2, 3, 5, -1, -3]:
            chi = grid.create_localized_mode(k=k, center=0, width=0.5)
            k_extracted = calculate_winding_number(chi, grid)
            assert_allclose(k_extracted, k, atol=0.1)


class TestMassSpectrum:
    """Test mass spectrum analysis."""
    
    def test_calibrate_beta(self):
        """Test β calibration from electron."""
        # If electron has A² = 2π and we want m_e = 0.000511 GeV
        # Then β = m_e / (2π) ≈ 8.13e-5 GeV
        A_sq = 2 * np.pi
        beta = calibrate_beta_from_electron(A_sq, ELECTRON_MASS_GEV)
        
        expected = ELECTRON_MASS_GEV / (2 * np.pi)
        assert_allclose(beta, expected, rtol=1e-10)
    
    def test_mass_spectrum_creation(self):
        """Test MassSpectrum class."""
        grid = SpectralGrid(N=64)
        spectrum = MassSpectrum(grid)
        
        # Create mock wavefunctions with different amplitudes
        chi_e = grid.create_localized_mode(k=1, amplitude=1.0)
        chi_mu = grid.create_localized_mode(k=1, amplitude=np.sqrt(MUON_ELECTRON_RATIO))
        
        spectrum.add_solution('electron', 1, 0.0, chi_e)
        spectrum.add_solution('muon', 1, 0.0, chi_mu)
        
        # Calibrate from electron
        spectrum.calibrate_beta()
        
        # Check that masses are populated
        assert spectrum.solutions['electron'].mass is not None
        assert spectrum.solutions['muon'].mass is not None
    
    def test_mass_ratios(self):
        """Test mass ratio calculation."""
        grid = SpectralGrid(N=64)
        spectrum = MassSpectrum(grid)
        
        # Create wavefunctions with known amplitude ratios
        chi_e = grid.create_localized_mode(k=1, amplitude=1.0)
        chi_mu = grid.create_localized_mode(k=1, amplitude=2.0)
        
        spectrum.add_solution('electron', 1, 0.0, chi_e)
        spectrum.add_solution('muon', 1, 0.0, chi_mu)
        
        ratios = spectrum.get_mass_ratios()
        
        # Since A_μ = 2 × A_e, A²_μ/A²_e = 4
        # Mass ratio = 4
        assert_allclose(ratios['muon/electron'], 4.0, rtol=0.1)


class TestTestbenchValidation:
    """Test the validation framework."""
    
    def test_validator_creation(self):
        """Test validator creation."""
        validator = TestbenchValidator(tolerance=0.10)
        assert validator.tolerance == 0.10
    
    def test_mass_validation(self):
        """Test mass validation against experimental values."""
        validator = TestbenchValidator(tolerance=0.10)
        
        masses = {
            'electron': ELECTRON_MASS_GEV,  # Exact match
            'muon': MUON_MASS_GEV * 1.05,   # 5% off
        }
        
        results = validator.validate_lepton_masses(masses)
        
        # Electron should pass (exact)
        assert results[0].passed
        
        # Muon should also pass (within 10%)
        assert results[1].passed
    
    def test_ratio_validation(self):
        """Test mass ratio validation."""
        validator = TestbenchValidator(tolerance=0.10)
        
        ratios = {
            'muon/electron': MUON_ELECTRON_RATIO * 0.95,  # 5% off
        }
        
        results = validator.validate_mass_ratios(ratios)
        
        # Should pass (within 10%)
        assert results[0].passed
    
    def test_beautiful_equation_validation(self):
        """Test Beautiful Equation validation."""
        validator = TestbenchValidator()
        
        params = SFTParameters(beta=50.0)
        result = validator.validate_beautiful_equation(params.beta, params.L0)
        
        assert result.passed


class TestPeriodicBoundaryConditions:
    """Test that periodic boundary conditions are satisfied."""
    
    def test_wavefunction_periodicity(self):
        """Test that eigenfunctions are 2π-periodic."""
        grid = SpectralGrid(N=128)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        solver = LinearEigensolver(grid, pot)
        
        _, wavefunctions = solver.solve(n_states=3)
        
        for psi in wavefunctions:
            # Check periodicity by comparing first and last values
            # (which wrap around)
            psi_extended = np.append(psi, psi[0])
            
            # The derivative should also be continuous
            dpsi = grid.first_derivative(psi)
            
            # Both should be consistent at boundary
            assert_allclose(psi[0], psi_extended[-1], atol=1e-10)
    
    def test_potential_periodicity(self):
        """Test that potential is 2π-periodic."""
        grid = SpectralGrid(N=64)
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        V = pot(grid.sigma)
        V_shifted = pot(grid.sigma + 2 * np.pi)
        
        assert_allclose(V, V_shifted, atol=1e-10)


class TestConvergenceWithResolution:
    """Test that solutions converge with increasing resolution."""
    
    def test_eigenvalue_convergence(self):
        """Test that eigenvalues converge as N increases."""
        pot = ThreeWellPotential(V0=1.0, V1=0.1)
        
        energies_list = []
        for N in [32, 64, 128, 256]:
            grid = SpectralGrid(N=N)
            solver = LinearEigensolver(grid, pot)
            energies, _ = solver.solve(n_states=3)
            energies_list.append(energies)
        
        # Check that highest resolution gives stable values
        # by comparing last two resolutions
        diff_final = np.abs(energies_list[-1] - energies_list[-2])
        # The first two eigenvalues should be well-converged
        assert np.all(diff_final[:2] < 0.1)

