"""
Integration tests for UnifiedSFMSolver.

End-to-end tests for the two-stage solver architecture.
"""

import pytest
import numpy as np
from sfm_solver.core.unified_solver import UnifiedSFMSolver


@pytest.mark.integration
class TestUnifiedSolver:
    """Test suite for unified solver (integration tests)."""
    
    @pytest.fixture
    def solver(self, test_constants):
        """Create a unified solver for testing."""
        # Use test constants (loads defaults from constants.json, overrides some values)
        return UnifiedSFMSolver(
            beta=test_constants['beta'],
            n_max=test_constants['n_max'],
            l_max=test_constants['l_max'],
            N_sigma=test_constants['N_sigma'],
            verbose=False
        )
    
    @pytest.mark.slow
    def test_lepton_generation_hierarchy(self, solver):
        """Test that leptons show generation hierarchy (e < mu < tau)."""
        # Solve for electron (n=1)
        result_e = solver.solve_lepton(winding_k=1, generation_n=1)
        
        # Solve for muon (n=2)
        result_mu = solver.solve_lepton(winding_k=1, generation_n=2)
        
        # Solve for tau (n=3)
        result_tau = solver.solve_lepton(winding_k=1, generation_n=3)
        
        # Check mass hierarchy: m_e < m_mu < m_tau
        assert result_e.mass < result_mu.mass, f"e mass {result_e.mass} >= mu mass {result_mu.mass}"
        assert result_mu.mass < result_tau.mass, f"mu mass {result_mu.mass} >= tau mass {result_tau.mass}"
        
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
        # Solve for proton (uud: windings 5,5,-3)
        result = solver.solve_baryon(
            quark_windings=(5, 5, -3),
            color_phases=(0, 2*np.pi/3, 4*np.pi/3),
            n_target=1,
            max_scf_iter=200
        )
        
        # Check result structure
        assert result.mass > 0, "Mass should be positive"
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
        assert 0.001 < result.mass < 1000.0, f"Mass {result.mass} GeV is outside reasonable range"
    
    @pytest.mark.slow
    def test_neutron_proton_splitting(self, solver):
        """Test that EM creates mass difference between proton and neutron."""
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
        
        # Should have different masses
        mass_diff = abs(result_n.mass - result_p.mass)
        assert mass_diff > 1e-6, "Proton and neutron have identical masses"
        
        # Both should converge
        assert result_p.shape_converged and result_p.energy_converged
        assert result_n.shape_converged and result_n.energy_converged
    
    @pytest.mark.slow
    def test_meson_solver(self, solver):
        """Test that meson solver runs end-to-end."""
        # Pion (u-dbar): windings (5, 3)
        result = solver.solve_meson(
            quark_winding=5,
            antiquark_winding=3,
            n_target=1,
            max_scf_iter=200
        )
        
        # Check basic result structure
        assert result.mass > 0
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

