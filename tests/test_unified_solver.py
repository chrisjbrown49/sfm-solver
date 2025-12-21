"""
Integration tests for UnifiedSFMSolver.

End-to-end tests for the two-stage solver architecture.
"""

import pytest
import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver
from sfm_solver.core.calculate_beta import calibrate_beta_from_electron


@pytest.mark.integration
class TestUnifiedSolver:
    """Test suite for unified solver (integration tests)."""
    
    @pytest.fixture
    def solver(self, test_constants):
        """Create a unified solver for testing.
        
        Note: Solver returns amplitudes only. To convert to masses, use:
            beta = calibrate_beta_from_electron(solver)
            mass = beta * A^2
        """
        return UnifiedSFMSolver(
            n_max=test_constants['n_max'],
            l_max=test_constants['l_max'],
            N_sigma=test_constants['N_sigma'],
            verbose=False
        )
    
    @pytest.mark.slow
    def test_lepton_generation_hierarchy(self, solver):
        """Test that leptons show generation hierarchy (e < mu < tau)."""
        # Calibrate beta from electron
        beta = calibrate_beta_from_electron(solver)
        
        # Solve for electron (n=1)
        result_e = solver.solve_lepton(winding_k=1, generation_n=1)
        
        # Solve for muon (n=2)
        result_mu = solver.solve_lepton(winding_k=1, generation_n=2)
        
        # Solve for tau (n=3)
        result_tau = solver.solve_lepton(winding_k=1, generation_n=3)
        
        # Calculate masses from amplitudes
        mass_e = beta * result_e.A**2
        mass_mu = beta * result_mu.A**2
        mass_tau = beta * result_tau.A**2
        
        # Check mass hierarchy: m_e < m_mu < m_tau
        assert mass_e < mass_mu, f"e mass {mass_e} >= mu mass {mass_mu}"
        assert mass_mu < mass_tau, f"mu mass {mass_mu} >= tau mass {mass_tau}"
        
        # Check that all converged
        assert result_e.shape_converged
        assert result_mu.shape_converged
        assert result_tau.shape_converged
        
        assert result_e.energy_converged
        assert result_mu.energy_converged
        assert result_tau.energy_converged
    
    @pytest.mark.slow
    def test_baryon_mass_prediction(self, solver):
        """Test end-to-end baryon mass prediction."""
        # Calibrate beta first
        beta = calibrate_beta_from_electron(solver)
        
        # Solve for proton (uud: windings 5,5,-3)
        result = solver.solve_baryon(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            n_target=1,
            max_scf_iter=200
        )
        
        # Calculate mass from amplitude
        mass = beta * result.A**2
        
        # Check result structure
        assert mass > 0, "Mass should be positive"
        assert result.A > 0, "Amplitude should be positive"
        assert result.Delta_x > 0, "Spatial scale should be positive"
        assert result.Delta_sigma > 0, "Subspace width should be positive"
        
        # Check convergence
        assert result.shape_converged, "Shape did not converge"
        assert result.energy_converged, "Energy minimization did not converge"
        
        # Check that shape structure was created
        assert len(result.shape_structure) > 0, "No shape structure created"
        
        # Mass should be in reasonable range for a baryon (with uncalibrated parameters)
        # Just check it's not absurdly large or small
        assert 0.001 < mass < 1000.0, f"Mass {mass} GeV is outside reasonable range"
    
    @pytest.mark.slow
    def test_neutron_proton_splitting(self, solver):
        """Test that EM creates mass difference between proton and neutron."""
        # Calibrate beta first
        beta = calibrate_beta_from_electron(solver)
        
        # Proton (uud): windings (5, 5, -3)
        result_p = solver.solve_baryon(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            n_target=1,
            max_scf_iter=200
        )
        
        # Neutron (udd): windings (5, -3, -3)
        result_n = solver.solve_baryon(
            quark_windings=(5, -3, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            n_target=1,
            max_scf_iter=200
        )
        
        # Calculate masses from amplitudes
        mass_p = beta * result_p.A**2
        mass_n = beta * result_n.A**2
        
        # Should have different masses
        mass_diff = abs(mass_n - mass_p)
        assert mass_diff > 1e-6, "Proton and neutron have identical masses"
        
        # Both should converge
        assert result_p.shape_converged and result_p.energy_converged
        assert result_n.shape_converged and result_n.energy_converged
    
    @pytest.mark.slow
    def test_meson_solver(self, solver):
        """Test that meson solver runs end-to-end."""
        # Calibrate beta first
        beta = calibrate_beta_from_electron(solver)
        
        # Pion (u-dbar): windings (5, 3)
        result = solver.solve_meson(
            quark_winding=5,
            antiquark_winding=3,
            n_target=1,
            max_scf_iter=200
        )
        
        # Calculate mass from amplitude
        mass = beta * result.A**2
        
        # Check basic result structure
        assert mass > 0
        assert result.A > 0
        assert result.Delta_x > 0
        assert result.particle_type == 'meson'
        
        # Should have converged
        assert result.shape_converged
        assert result.energy_converged
    
    def test_energy_breakdown(self, solver):
        """Test that energy components sum to total."""
        result = solver.solve_lepton(winding_k=1, generation_n=1)
        
        # Sum individual components
        component_sum = (
            result.energy_components['E_sigma'] +
            result.energy_components['E_spatial'] +
            result.energy_components['E_coupling'] +
            result.energy_components['E_curvature'] +
            result.energy_components['E_em']
        )
        
        # Should equal total
        assert abs(component_sum - result.E_total) < 1e-6, \
            f"Components sum {component_sum} != total {result.E_total}"
    
    def test_quantum_numbers_stored(self, solver):
        """Test that quantum numbers are properly stored in result."""
        result = solver.solve_baryon(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            n_target=1
        )
        
        # Check quantum numbers are stored
        assert 'quark_windings' in result.quantum_numbers
        assert result.quantum_numbers['quark_windings'] == (5, 5, -3)
        assert 'color_phases' in result.quantum_numbers
        assert 'n_target' in result.quantum_numbers


@pytest.mark.integration
class TestBetaCalibration:
    """Test suite for β calibration functionality using new architecture."""
    
    def test_beta_calibration_from_electron(self, test_constants):
        """Test β calibration from electron mass using helper function."""
        # Create solver (no beta parameter in constructor)
        solver = UnifiedSFMSolver(
            n_max=test_constants['n_max'],
            l_max=test_constants['l_max'],
            N_sigma=test_constants['N_sigma'],
            verbose=False
        )
        
        # Calibrate from electron using helper function
        beta_calibrated = calibrate_beta_from_electron(solver, electron_mass_exp=0.000510999)
        
        # β should be positive and reasonable
        assert beta_calibrated > 0
        
        # β should be in reasonable range (order 10^-3 to 10^-2 GeV)
        assert 0.0001 < beta_calibrated < 0.1
        
        # Verify electron mass matches by construction
        result_e = solver.solve_lepton(winding_k=1, generation_n=1)
        m_e_pred = beta_calibrated * result_e.A**2 * 1000  # MeV
        m_e_exp = 0.510999  # MeV
        
        # Should match exactly (within numerical precision)
        assert abs(m_e_pred - m_e_exp) < 1e-5, \
            f"Electron mass mismatch: predicted {m_e_pred:.6f} vs expected {m_e_exp:.6f}"
    
    def test_mass_ratios_from_amplitude_ratios(self, test_constants):
        """Test that mass ratios equal amplitude ratios (beta-independent)."""
        solver = UnifiedSFMSolver(
            n_max=test_constants['n_max'],
            l_max=test_constants['l_max'],
            N_sigma=test_constants['N_sigma'],
            verbose=False
        )
        
        # Calibrate beta
        beta = calibrate_beta_from_electron(solver)
        
        # Solve electron and muon
        result_e = solver.solve_lepton(winding_k=1, generation_n=1)
        result_mu = solver.solve_lepton(winding_k=1, generation_n=2)
        
        # Calculate masses from amplitudes
        mass_e = beta * result_e.A**2
        mass_mu = beta * result_mu.A**2
        
        # Mass ratio should equal amplitude ratio squared
        mass_ratio = mass_mu / mass_e
        amplitude_ratio_sq = (result_mu.A / result_e.A)**2
        
        # Should match within numerical precision
        assert abs(mass_ratio - amplitude_ratio_sq) < 1e-6, \
            f"Mass ratio {mass_ratio:.6f} != amplitude ratio {amplitude_ratio_sq:.6f}"
        
        # This is the fundamental relationship: m_μ/m_e = A_μ²/A_e²
        # Independent of β value - this is a CRITICAL test of the architecture
    
    def test_beta_consistency_across_particles(self, test_constants):
        """Test that the same beta works correctly for all particle types."""
        solver = UnifiedSFMSolver(
            n_max=test_constants['n_max'],
            l_max=test_constants['l_max'],
            N_sigma=test_constants['N_sigma'],
            verbose=False
        )
        
        # Calibrate beta from electron
        beta = calibrate_beta_from_electron(solver)
        
        # Solve different particle types
        result_e = solver.solve_lepton(winding_k=1, generation_n=1)
        result_mu = solver.solve_lepton(winding_k=1, generation_n=2)
        
        # Calculate masses
        mass_e = beta * result_e.A**2 * 1000  # MeV
        mass_mu = beta * result_mu.A**2 * 1000  # MeV
        
        # Electron should match by construction
        assert abs(mass_e - 0.510999) < 1e-5
        
        # Muon mass should be physically reasonable (even if not exact)
        # Just check it's larger than electron and in ballpark
        assert mass_mu > mass_e, "Muon should be heavier than electron"
        assert mass_mu > 50, "Muon mass should be > 50 MeV (experimental is 105.7 MeV)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

