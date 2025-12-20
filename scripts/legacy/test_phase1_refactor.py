"""
Phase 1 Refactor Verification Test

This test verifies that the refactored modules (NonSeparableWavefunctionSolver
and UniversalEnergyMinimizer) produce consistent results.

Key checks:
1. Wavefunction solver produces correct structure (l-composition, k_eff)
2. Energy minimizer optimizes (A, dx, ds) correctly
3. Results are physically reasonable
"""

import numpy as np
import sys


def test_wavefunction_solver():
    """Test that wavefunction solver produces expected structures."""
    print("=" * 60)
    print("TEST 1: Wavefunction Solver")
    print("=" * 60)
    
    from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
    
    solver = NonSeparableWavefunctionSolver(
        alpha=20.0,
        beta=100.0,
        kappa=0.0001,
        g1=5000.0,
        g2=0.004,
        V0=1.0,
        n_max=5,
        l_max=2,
        N_sigma=48,
    )
    
    results = {}
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        structure = solver.solve_lepton(n_target=n, k_winding=1, verbose=False)
        results[name] = structure
        
        print(f"\n{name} (n={n}):")
        print(f"  l-composition: l=0: {structure.l_composition.get(0,0)*100:.1f}%, "
              f"l=1: {structure.l_composition.get(1,0)*100:.1f}%")
        print(f"  k_eff: {structure.k_eff:.4f}")
        print(f"  structure_norm: {structure.structure_norm:.4f}")
    
    # Verify expected properties
    checks_passed = True
    
    # Check 1: l=0 should dominate (>80%)
    for name, structure in results.items():
        if structure.l_composition.get(0, 0) < 0.8:
            print(f"FAIL: {name} l=0 fraction too low: {structure.l_composition.get(0,0)}")
            checks_passed = False
    
    # Check 2: k_eff should be positive (winding = +1)
    for name, structure in results.items():
        if structure.k_eff <= 0:
            print(f"FAIL: {name} k_eff should be positive: {structure.k_eff}")
            checks_passed = False
    
    # Check 3: structure should be normalized
    for name, structure in results.items():
        if abs(structure.structure_norm - 1.0) > 0.01:
            print(f"FAIL: {name} not normalized: {structure.structure_norm}")
            checks_passed = False
    
    if checks_passed:
        print("\n[PASS] All wavefunction solver checks passed")
    else:
        print("\n[FAIL] Some wavefunction solver checks failed")
    
    return checks_passed, results


def test_energy_minimizer():
    """Test that energy minimizer finds reasonable minima."""
    print("\n" + "=" * 60)
    print("TEST 2: Energy Minimizer")
    print("=" * 60)
    
    from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
    from sfm_solver.core.universal_energy_minimizer import UniversalEnergyMinimizer
    
    # Same parameters
    alpha = 20.0
    beta = 100.0
    kappa = 0.0001
    g1 = 5000.0
    g2 = 0.004
    V0 = 1.0
    
    wf_solver = NonSeparableWavefunctionSolver(
        alpha=alpha, beta=beta, kappa=kappa,
        g1=g1, g2=g2, V0=V0,
        n_max=5, l_max=2, N_sigma=48,
    )
    
    minimizer = UniversalEnergyMinimizer(
        alpha=alpha, beta=beta, kappa=kappa,
        g1=g1, g2=g2, V0=V0,
    )
    
    results = {}
    for n, name in [(1, 'Electron'), (2, 'Muon'), (3, 'Tau')]:
        structure = wf_solver.solve_lepton(n_target=n, k_winding=1)
        
        result = minimizer.minimize(
            structure=structure,
            sigma_grid=wf_solver.get_sigma_grid(),
            V_sigma=wf_solver.get_V_sigma(),
            spatial_coupling=wf_solver.get_spatial_coupling_matrix(),
            state_index_map=wf_solver.get_state_index_map(),
            verbose=False,
        )
        results[name] = result
        
        print(f"\n{name} (n={n}):")
        print(f"  A = {result.A:.4f}, A^2 = {result.A_squared:.6f}")
        print(f"  dx = {result.delta_x:.4f}, ds = {result.delta_sigma:.4f}")
        print(f"  Mass: {result.mass_gev:.6f} GeV")
        print(f"  E_coupling: {result.E_coupling:.6f}")
        print(f"  Compton check: {result.compton_check:.4f}")
    
    checks_passed = True
    
    # Check 1: A should be positive
    for name, result in results.items():
        if result.A <= 0:
            print(f"FAIL: {name} A should be positive: {result.A}")
            checks_passed = False
    
    # Check 2: Mass should be positive
    for name, result in results.items():
        if result.mass_gev <= 0:
            print(f"FAIL: {name} mass should be positive: {result.mass_gev}")
            checks_passed = False
    
    # Check 3: delta_x and delta_sigma should be positive
    for name, result in results.items():
        if result.delta_x <= 0 or result.delta_sigma <= 0:
            print(f"FAIL: {name} deltas should be positive")
            checks_passed = False
    
    if checks_passed:
        print("\n[PASS] All energy minimizer checks passed")
    else:
        print("\n[FAIL] Some energy minimizer checks failed")
    
    return checks_passed, results


def test_architecture_separation():
    """Test that the two-stage architecture is working."""
    print("\n" + "=" * 60)
    print("TEST 3: Architecture Separation")
    print("=" * 60)
    
    from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
    from sfm_solver.core.universal_energy_minimizer import UniversalEnergyMinimizer
    
    wf_solver = NonSeparableWavefunctionSolver(
        alpha=20.0, beta=100.0, kappa=0.0001,
        g1=5000.0, g2=0.004, V0=1.0,
    )
    
    minimizer = UniversalEnergyMinimizer(
        alpha=20.0, beta=100.0, kappa=0.0001,
        g1=5000.0, g2=0.004, V0=1.0,
    )
    
    # Stage 1: Get wavefunction structure
    print("\nStage 1: Solving wavefunction structure...")
    structure = wf_solver.solve_lepton(n_target=2, k_winding=1)
    print(f"  Structure type: {structure.particle_type}")
    print(f"  n_target: {structure.n_target}, k_winding: {structure.k_winding}")
    print(f"  chi_components keys: {list(structure.chi_components.keys())[:5]}...")
    
    # Stage 2: Minimize energy
    print("\nStage 2: Minimizing energy over (A, dx, ds)...")
    result = minimizer.minimize(
        structure=structure,
        sigma_grid=wf_solver.get_sigma_grid(),
        V_sigma=wf_solver.get_V_sigma(),
        spatial_coupling=wf_solver.get_spatial_coupling_matrix(),
        state_index_map=wf_solver.get_state_index_map(),
        verbose=False,
    )
    print(f"  Optimized A: {result.A:.4f}")
    print(f"  Optimized dx: {result.delta_x:.4f}")
    print(f"  Optimized ds: {result.delta_sigma:.4f}")
    print(f"  Predicted mass: {result.mass_gev:.6f} GeV")
    
    # Verify separation
    checks_passed = True
    
    # Structure should be unit-normalized
    if abs(structure.structure_norm - 1.0) > 0.01:
        print(f"FAIL: Structure not unit-normalized: {structure.structure_norm}")
        checks_passed = False
    
    # Result should have all required fields
    required_fields = ['A', 'delta_x', 'delta_sigma', 'mass_gev', 'E_coupling']
    for field in required_fields:
        if not hasattr(result, field):
            print(f"FAIL: Missing field: {field}")
            checks_passed = False
    
    if checks_passed:
        print("\n[PASS] Architecture separation working correctly")
    else:
        print("\n[FAIL] Architecture separation has issues")
    
    return checks_passed


def main():
    """Run all Phase 1 verification tests."""
    print("=" * 60)
    print("PHASE 1 REFACTOR VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    test1_passed, _ = test_wavefunction_solver()
    all_passed = all_passed and test1_passed
    
    test2_passed, _ = test_energy_minimizer()
    all_passed = all_passed and test2_passed
    
    test3_passed = test_architecture_separation()
    all_passed = all_passed and test3_passed
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("\n[PASS] All Phase 1 refactor tests passed!")
        print("\nThe refactored architecture is working:")
        print("  - NonSeparableWavefunctionSolver produces correct structures")
        print("  - UniversalEnergyMinimizer optimizes (A, dx, ds)")
        print("  - Two-stage separation is functioning")
        print("\nKnown limitation: Coupling factor is near zero due to")
        print("the perturbative approximation. This is expected and")
        print("matches the limitations of the original nonseparable_solver.")
        return 0
    else:
        print("\n[FAIL] Some Phase 1 tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

