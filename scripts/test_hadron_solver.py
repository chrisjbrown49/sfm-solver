#!/usr/bin/env python3
"""Test self-consistent meson and baryon solvers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1

def main():
    print("=" * 60)
    print("HADRON SOLVER TEST")
    print("=" * 60)
    
    solver = NonSeparableWavefunctionSolver(
        alpha=ALPHA, g_internal=G_INTERNAL, g1=G1,
        g2=0.004, V0=1.0, n_max=5, l_max=2, N_sigma=64,
    )
    
    # Solve all particles (max_iter_nl=0 = Step 1 only, no nonlinear iteration)
    e = solver.solve_lepton_self_consistent(n_target=1, max_iter_nl=0)
    mu = solver.solve_lepton_self_consistent(n_target=2, max_iter_nl=0)
    tau = solver.solve_lepton_self_consistent(n_target=3, max_iter_nl=0)
    pion = solver.solve_meson_self_consistent(quark_wells=(1, 2))
    proton = solver.solve_baryon_self_consistent(quark_wells=(1, 2, 3))
    
    # Derive beta from electron
    beta_out = 0.000511 / (e.structure_norm ** 2)
    
    # Compute masses
    particles = [('Electron', e, 0.511), ('Muon', mu, 105.66), 
                 ('Tau', tau, 1776.86), ('Pion', pion, 139.57),
                 ('Proton', proton, 938.27)]
    
    print(f"\nbeta_output = {beta_out:.6f} GeV\n")
    print(f"{'Particle':<10} {'A':>8} {'dx':>10} {'Pred MeV':>10} {'Exp MeV':>10} {'Err%':>8}")
    print("-" * 60)
    
    for name, p, exp in particles:
        pred = beta_out * p.structure_norm**2 * 1000
        err = (pred - exp) / exp * 100
        print(f"{name:<10} {p.structure_norm:>8.4f} {p.delta_x_final:>10.6f} {pred:>10.2f} {exp:>10.2f} {err:>7.1f}%")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

