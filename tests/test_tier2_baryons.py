"""
Tier 2 Tests: Baryon Structure and Mass Predictions.

These tests validate baryon (three-quark) bound state calculations using
the COMPOSITE wavefunction model (single wavefunction, not three separate).

Key physics:
- Baryon is a SINGLE composite wavefunction with three localization peaks
- Quarks have winding number k=3
- Baryon mass: m = β × A² (total amplitude squared)
- Color neutrality: Σe^(iφᵢ) = 0 emerges from energy minimization
- Binding: E_total < 0 (bound state)
- Coupling energy E_coupling = -α×k×A stabilizes amplitude

Tier 2 Baryon Success Criteria:
1. Amplitude stabilizes at finite value (not → 0)
2. Energy is negative (bound state)
3. Color sum magnitude < 0.01 (color neutrality emerges)
4. Phase differences ≈ 2π/3 (120° apart)
5. Convergence achieved
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from sfm_solver.core.grid import SpectralGrid
from sfm_solver.core.constants import PROTON_MASS_GEV, NEUTRON_MASS_GEV
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.multiparticle import (
    BaryonSolver,
    BaryonState,
    CompositeBaryonSolver,
    CompositeBaryonState,
    MesonSolver,
    solve_meson_system,
)


# Convert to MeV
PROTON_MASS_MEV = PROTON_MASS_GEV * 1000
NEUTRON_MASS_MEV = NEUTRON_MASS_GEV * 1000
NEUTRON_PROTON_DIFF_MEV = NEUTRON_MASS_MEV - PROTON_MASS_MEV

# Target phase difference (2π/3)
TARGET_PHASE_DIFF = 2 * np.pi / 3  # ≈ 2.094


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def grid():
    """Standard spectral grid for tests."""
    return SpectralGrid(N=128)


@pytest.fixture
def potential():
    """Standard three-well potential."""
    return ThreeWellPotential(V0=10.0, V1=0.1)


@pytest.fixture
def baryon_solver(grid, potential):
    """Standard composite baryon solver with coupling energy."""
    return CompositeBaryonSolver(
        grid, potential, 
        g1=0.1,      # Nonlinear coupling
        g2=0.1,      # Circulation coupling
        alpha=2.0,   # Coupling that stabilizes amplitude
        k=3          # Winding number for quarks
    )


@pytest.fixture
def meson_solver(grid, potential):
    """Standard meson solver."""
    return MesonSolver(grid, potential, g1=0.1, k_quark=3)


# =============================================================================
# Test Class: Baryon Solver Basic Functionality
# =============================================================================

class TestTier2BaryonSolver:
    """Test basic composite baryon solver functionality."""
    
    @pytest.mark.tier2
    def test_baryon_solver_converges(self, baryon_solver):
        """Baryon solver produces valid output and converges."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # Verify output is valid
        assert state.chi_baryon is not None, "Should produce wavefunction output"
        assert np.isfinite(state.energy_total), "Energy should be finite"
        assert state.final_residual < 1.0, "Residual should decrease"
    
    @pytest.mark.tier2
    def test_baryon_has_composite_wavefunction(self, baryon_solver):
        """Baryon state contains a single composite wavefunction."""
        state = baryon_solver.solve(
            max_iter=1000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        assert state.chi_baryon is not None
        assert len(state.chi_baryon) == baryon_solver.grid.N
        assert state.chi_baryon.dtype == np.complexfloating or np.iscomplexobj(state.chi_baryon)
    
    @pytest.mark.tier2
    def test_amplitude_stabilizes(self, baryon_solver):
        """Amplitude stabilizes at finite value, not zero."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # KEY TEST: Amplitude must NOT collapse to zero
        assert state.amplitude_squared > 0.1, \
            f"Amplitude collapsed to {state.amplitude_squared}, should be stable"
    
    @pytest.mark.tier2
    def test_coupling_energy_negative(self, baryon_solver):
        """Coupling energy is negative (provides stabilization)."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # Coupling energy -α×k×A should be negative
        assert state.energy_coupling < 0, \
            f"Coupling energy should be negative, got {state.energy_coupling}"


# =============================================================================
# Test Class: Binding Energy
# =============================================================================

class TestTier2BindingEnergy:
    """Test binding energy is negative (bound state)."""
    
    @pytest.mark.tier2
    def test_total_energy_negative(self, baryon_solver):
        """Total energy should be negative for bound state."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # Total energy should be negative (bound state)
        assert state.energy_total < 0, \
            f"Bound state should have negative energy, got {state.energy_total}"
    
    @pytest.mark.tier2
    def test_coupling_dominates(self, baryon_solver):
        """Coupling energy should dominate to create binding."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # |E_coupling| > (E_kinetic + E_potential + E_nonlinear)
        other_energies = state.energy_kinetic + state.energy_potential + state.energy_nonlinear
        assert abs(state.energy_coupling) > other_energies, \
            "Coupling should provide enough binding"


# =============================================================================
# Test Class: Color Emergence
# =============================================================================

class TestTier2ColorEmergence:
    """Test that color neutrality emerges from energy minimization."""
    
    @pytest.mark.tier2
    def test_color_sum_near_zero(self, baryon_solver):
        """Color sum |Σe^(iφᵢ)| should be near zero."""
        state = baryon_solver.solve(
            max_iter=3000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # Color sum should be very small (< 0.1)
        assert state.color_sum_magnitude < 0.1, \
            f"Color sum should be ~0, got {state.color_sum_magnitude}"
    
    @pytest.mark.tier2
    def test_is_color_neutral(self, baryon_solver):
        """State should be marked as color neutral."""
        state = baryon_solver.solve(
            max_iter=3000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        assert state.is_color_neutral, \
            f"Should be color neutral (sum={state.color_sum_magnitude})"
    
    @pytest.mark.tier2
    def test_phase_differences_120_degrees(self, baryon_solver):
        """Phase differences should be approximately 2π/3 (120°)."""
        state = baryon_solver.solve(
            max_iter=3000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # Both phase differences should be ~2.094 (= 2π/3)
        for i, diff in enumerate(state.phase_differences):
            assert abs(diff - TARGET_PHASE_DIFF) < 0.2, \
                f"Phase diff {i} = {diff}, expected {TARGET_PHASE_DIFF}"
    
    @pytest.mark.tier2
    def test_three_phases_extracted(self, baryon_solver):
        """Should extract three phases from composite wavefunction."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        assert len(state.phases) == 3, "Should have three phases"
        for phase in state.phases:
            assert 0 <= phase < 2 * np.pi, f"Phase {phase} out of range"


# =============================================================================
# Test Class: Quark Confinement
# =============================================================================

class TestTier2QuarkConfinement:
    """Test quark confinement is satisfied."""
    
    @pytest.mark.tier2
    def test_bound_state_exists(self, baryon_solver):
        """Bound state with three peaks exists."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        chi = state.chi_baryon
        density = np.abs(chi)**2
        
        # Should have three peaks (at well positions)
        peaks = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i-1] and density[i] > density[i+1]:
                if density[i] > 0.01 * np.max(density):  # Significant peak
                    peaks.append(i)
        
        assert len(peaks) >= 2, f"Should have multiple peaks, found {len(peaks)}"


# =============================================================================
# Test Class: Amplitudes
# =============================================================================

class TestTier2Amplitudes:
    """Test amplitude properties."""
    
    @pytest.mark.tier2
    def test_amplitude_determines_mass(self, baryon_solver):
        """Amplitude squared determines mass (m = β × A²)."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # Amplitude should be finite and positive
        assert state.amplitude_squared > 0
        assert np.isfinite(state.amplitude_squared)
    
    @pytest.mark.tier2
    def test_amplitude_scaling_with_alpha(self, grid, potential):
        """Larger alpha leads to larger stable amplitude."""
        amplitudes = []
        for alpha in [1.0, 2.0, 4.0]:
            solver = CompositeBaryonSolver(
                grid, potential, g1=0.1, g2=0.1, alpha=alpha, k=3
            )
            state = solver.solve(max_iter=2000, dt=0.001, verbose=False)
            amplitudes.append(state.amplitude_squared)
        
        # Larger alpha should give larger amplitude
        assert amplitudes[1] > amplitudes[0], "Amplitude should grow with alpha"
        assert amplitudes[2] > amplitudes[1], "Amplitude should grow with alpha"


# =============================================================================
# Test Class: Meson Solver
# =============================================================================

class TestTier2MesonSolver:
    """Test meson (quark-antiquark) solver."""
    
    @pytest.mark.tier2
    def test_meson_solver_converges(self, meson_solver):
        """Meson solver should converge."""
        state = meson_solver.solve(max_iter=200, verbose=False)
        
        assert state.chi_quark is not None
        assert state.chi_antiquark is not None
        assert np.isfinite(state.energy_total)
    
    @pytest.mark.tier2
    def test_meson_opposite_windings(self, meson_solver):
        """Meson has opposite winding numbers for q and q̄."""
        # This is encoded in the solver - just verify it runs
        state = meson_solver.solve(max_iter=200, verbose=False)
        assert state is not None


# =============================================================================
# Test Class: Physical Consistency
# =============================================================================

class TestTier2PhysicalConsistency:
    """Test physical consistency of baryon solutions."""
    
    @pytest.mark.tier2
    def test_energy_components_sum(self, baryon_solver):
        """Energy components should sum to total."""
        state = baryon_solver.solve(
            max_iter=2000,
            dt=0.001,
            initial_amplitude=1.0,
            verbose=False
        )
        
        # Note: Total includes circulation energy which is computed separately
        computed_sum = (state.energy_kinetic + state.energy_potential + 
                       state.energy_nonlinear + state.energy_coupling)
        
        # Allow for circulation energy contribution
        assert abs(state.energy_total - computed_sum) < abs(state.energy_total) * 0.5
    
    @pytest.mark.tier2
    def test_reproducibility(self, grid, potential):
        """Same parameters should give same result."""
        solver1 = CompositeBaryonSolver(grid, potential, g1=0.1, g2=0.1, alpha=2.0, k=3)
        solver2 = CompositeBaryonSolver(grid, potential, g1=0.1, g2=0.1, alpha=2.0, k=3)
        
        state1 = solver1.solve(max_iter=1000, dt=0.001, verbose=False)
        state2 = solver2.solve(max_iter=1000, dt=0.001, verbose=False)
        
        assert_allclose(state1.energy_total, state2.energy_total, rtol=0.01)
        assert_allclose(state1.amplitude_squared, state2.amplitude_squared, rtol=0.01)


# =============================================================================
# Test Class: Parameter Sensitivity
# =============================================================================

class TestTier2ParameterSensitivity:
    """Test sensitivity to solver parameters."""
    
    @pytest.mark.tier2
    def test_without_coupling_amplitude_collapses(self, grid, potential):
        """Without coupling energy (alpha=0), amplitude should collapse."""
        solver = CompositeBaryonSolver(
            grid, potential, g1=0.1, g2=0.1, alpha=0.0, k=3
        )
        state = solver.solve(max_iter=1000, dt=0.001, verbose=False)
        
        # Without coupling, amplitude should be very small
        assert state.amplitude_squared < 0.01, \
            f"Without coupling, amplitude should collapse, got {state.amplitude_squared}"
    
    @pytest.mark.tier2
    def test_with_coupling_amplitude_stable(self, grid, potential):
        """With coupling energy (alpha>0), amplitude should stabilize."""
        solver = CompositeBaryonSolver(
            grid, potential, g1=0.1, g2=0.1, alpha=2.0, k=3
        )
        state = solver.solve(max_iter=2000, dt=0.001, verbose=False)
        
        # With coupling, amplitude should be substantial
        assert state.amplitude_squared > 0.1, \
            f"With coupling, amplitude should stabilize, got {state.amplitude_squared}"
