"""
Phase 2 Refactor Verification Test

Tests that the refactored solvers (lepton, meson, baryon) work correctly
with the new non-separable wavefunction architecture.
"""

import numpy as np
import sys


def test_lepton_solver():
    """Test refactored lepton solver."""
    print("=" * 60)
    print("TEST: Lepton Solver (Task 2.1)")
    print("=" * 60)
    
    from sfm_solver.eigensolver.sfm_lepton_solver import SFMLeptonSolver
    
    solver = SFMLeptonSolver()
    
    results = {}
    for particle in ['electron', 'muon', 'tau']:
        print(f"\nSolving {particle}...")
        result = solver.solve_lepton(particle=particle, verbose=False)
        results[particle] = result
        print(f"  n_spatial: {result.n_spatial}")
        print(f"  A^2: {result.amplitude_squared:.6e}")
        print(f"  Mass: {solver.beta * result.amplitude_squared * 1000:.4f} MeV")
        print(f"  l_composition: {result.l_composition}")
        print(f"  Converged: {result.converged}")
    
    # Check mass ordering
    mass_e = results['electron'].amplitude_squared
    mass_mu = results['muon'].amplitude_squared
    mass_tau = results['tau'].amplitude_squared
    
    print(f"\nMass ordering check:")
    print(f"  m_tau > m_mu: {mass_tau > mass_mu}")
    print(f"  m_mu > m_e: {mass_mu > mass_e}")
    
    checks_passed = True
    if not (results['electron'].converged and results['muon'].converged and results['tau'].converged):
        print("FAIL: Not all leptons converged")
        checks_passed = False
    
    if results['electron'].l_composition is None:
        print("FAIL: l_composition not set")
        checks_passed = False
    
    if checks_passed:
        print("\n[PASS] Lepton solver refactor complete")
    else:
        print("\n[FAIL] Lepton solver has issues")
    
    return checks_passed


def test_meson_solver():
    """Test refactored meson solver."""
    print("\n" + "=" * 60)
    print("TEST: Meson Solver (Task 2.2)")
    print("=" * 60)
    
    from sfm_solver.core.grid import SpectralGrid
    from sfm_solver.potentials.three_well import ThreeWellPotential
    from sfm_solver.multiparticle.composite_meson import CompositeMesonSolver
    
    grid = SpectralGrid(N=128)
    potential = ThreeWellPotential(V0=1.0, V1=0.1)
    solver = CompositeMesonSolver(grid, potential)
    
    mesons = ['pion_plus', 'jpsi', 'upsilon_1s']
    results = {}
    
    for meson in mesons:
        print(f"\nSolving {meson}...")
        result = solver.solve(meson_type=meson, verbose=False)
        results[meson] = result
        print(f"  Quark content: {result.quark}, {result.antiquark}")
        print(f"  n_spatial: {result.n_spatial}")
        print(f"  A^2: {result.amplitude_squared:.6e}")
        print(f"  Mass: {solver.beta * result.amplitude_squared * 1000:.2f} MeV")
        print(f"  k_net: {result.k_net}, k_eff: {result.k_eff:.4f}")
        print(f"  Converged: {result.converged}")
    
    checks_passed = True
    for meson, result in results.items():
        if not result.converged:
            print(f"FAIL: {meson} did not converge")
            checks_passed = False
    
    if checks_passed:
        print("\n[PASS] Meson solver refactor complete")
    else:
        print("\n[FAIL] Meson solver has issues")
    
    return checks_passed


def test_baryon_solver():
    """Test refactored baryon solver."""
    print("\n" + "=" * 60)
    print("TEST: Baryon Solver (Task 2.3)")
    print("=" * 60)
    
    from sfm_solver.core.grid import SpectralGrid
    from sfm_solver.potentials.three_well import ThreeWellPotential
    from sfm_solver.multiparticle.composite_baryon import CompositeBaryonSolver, PROTON_QUARKS, NEUTRON_QUARKS
    
    grid = SpectralGrid(N=128)
    potential = ThreeWellPotential(V0=1.0, V1=0.1)
    solver = CompositeBaryonSolver(grid, potential)
    
    baryons = [('proton', PROTON_QUARKS), ('neutron', NEUTRON_QUARKS)]
    results = {}
    
    for name, quarks in baryons:
        print(f"\nSolving {name}...")
        result = solver.solve(quark_types=quarks, verbose=False)
        results[name] = result
        print(f"  Quark content: {result.quark_types}")
        print(f"  n_spatial: {result.n_spatial}")
        print(f"  A^2: {result.amplitude_squared:.6e}")
        print(f"  Mass: {solver.beta * result.amplitude_squared * 1000:.2f} MeV")
        print(f"  k_total: {result.k_total}, k_eff: {result.k_eff:.4f}")
        print(f"  Color neutral: {result.is_color_neutral}")
        print(f"  Converged: {result.converged}")
    
    checks_passed = True
    for name, result in results.items():
        if not result.converged:
            print(f"FAIL: {name} did not converge")
            checks_passed = False
    
    if checks_passed:
        print("\n[PASS] Baryon solver refactor complete")
    else:
        print("\n[FAIL] Baryon solver has issues")
    
    return checks_passed


def main():
    """Run all Phase 2 verification tests."""
    print("=" * 60)
    print("PHASE 2 REFACTOR VERIFICATION")
    print("Integrating new architecture with particle solvers")
    print("=" * 60)
    
    all_passed = True
    
    test1_passed = test_lepton_solver()
    all_passed = all_passed and test1_passed
    
    test2_passed = test_meson_solver()
    all_passed = all_passed and test2_passed
    
    test3_passed = test_baryon_solver()
    all_passed = all_passed and test3_passed
    
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    
    print(f"\n  Task 2.1 (Lepton Solver): {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Task 2.2 (Meson Solver): {'PASS' if test2_passed else 'FAIL'}")
    print(f"  Task 2.3 (Baryon Solver): {'PASS' if test3_passed else 'FAIL'}")
    
    if all_passed:
        print("\n[PASS] All Phase 2 tasks completed!")
        print("\nAll particle solvers now use:")
        print("  - NonSeparableWavefunctionSolver (Stage 1: structure)")
        print("  - UniversalEnergyMinimizer (Stage 2: scale)")
        return 0
    else:
        print("\n[FAIL] Some Phase 2 tasks failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

