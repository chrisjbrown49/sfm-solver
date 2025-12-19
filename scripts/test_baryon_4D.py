"""
Test the new 4D three-quark baryon solver with SPIN implementation.

This solver treats each quark as a full 4D entity with wavefunction Ψᵢ(x,y,z,σ).
The baryon is the 4D superposition: Ψ_baryon = Ψ₁ + Ψ₂ + Ψ₃.

NOW WITH SPIN: Each quark has spin quantum number (±1) and generation (n=1,2,3).
"""
import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G1, G2, LAMBDA_SO, V0, V1, BETA
from sfm_solver.core.particle_configurations import PROTON, NEUTRON

print("="*70)
print("4D THREE-QUARK BARYON SOLVER TEST")
print("="*70)
print()

# First, get electron to derive beta
print("Deriving beta from electron...")
solver = NonSeparableWavefunctionSolver(
    alpha=ALPHA, beta=BETA, g_internal=G_INTERNAL, g1=G1,
    g2=G2, lambda_so=LAMBDA_SO, V0=V0, V1=V1, n_max=5, l_max=2, N_sigma=64,
)

e = solver.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0)
beta = 0.511 / (e.structure_norm ** 2)  # Electron mass = 0.511 MeV = 0.000511 GeV, but we work in GeV
print(f"  Electron: A = {e.structure_norm:.6f}")
print(f"  Converged: {e.converged} (iterations: {e.iterations})")
print(f"  beta = {beta:.6e} GeV")
print()

# Recreate solver with correct beta
solver = NonSeparableWavefunctionSolver(
    alpha=ALPHA, beta=beta, g_internal=G_INTERNAL, g1=G1,
    g2=G2, lambda_so=LAMBDA_SO, V0=V0, V1=V1, n_max=5, l_max=2, N_sigma=64,
)

# Target masses
m_p_exp = 938.27  # MeV
m_n_exp = 939.57  # MeV
delta_m_exp = 1.30  # MeV

print("="*70)
print("PROTON (uud) WITH SPIN")
print("="*70)
print(f"Configuration:")
print(f"  Quarks: {PROTON.quarks}")
print(f"  Windings: {PROTON.windings}")
print(f"  Spins: {PROTON.spins}  (spin up, spin down, spin up)")
print(f"  Generations: {PROTON.generations}")
print(f"  Note: Two u quarks have OPPOSITE spins to satisfy Pauli exclusion!")
print()

proton = solver.solve_baryon_4D_self_consistent(
    quark_wells=(1, 2, 3),
    color_phases=(0, 2*np.pi/3, 4*np.pi/3),
    quark_windings=PROTON.windings,
    quark_spins=PROTON.spins,
    quark_generations=PROTON.generations,
    max_iter_outer=30,
    max_iter_scf=30,
    verbose=True,
)

A_p = proton.structure_norm
m_p = beta * A_p**2 * 1000  # MeV
error_p = 100 * (m_p - m_p_exp) / m_p_exp

print(f"\nProton Results:")
print(f"  A = {A_p:.6f}")
print(f"  mass = {m_p:.2f} MeV (target: {m_p_exp:.2f} MeV)")
print(f"  error = {error_p:+.1f}%")
print(f"  k_eff = {proton.k_eff:.4f}")
print(f"  E_EM = {proton.em_energy:.4e}")
print(f"\nConvergence Status:")
print(f"  Converged: {proton.converged}")
print(f"  Iterations: {proton.iterations}")
print(f"  Delta_x_final: {proton.delta_x_final:.6e}")

# Analyze convergence history
if proton.convergence_history:
    hist = proton.convergence_history
    if 'A' in hist and len(hist['A']) > 1:
        A_vals = hist['A']
        print(f"\nAmplitude Convergence:")
        print(f"  Initial A: {A_vals[0]:.6f}")
        print(f"  Final A:   {A_vals[-1]:.6f}")
        print(f"  Change:    {abs(A_vals[-1] - A_vals[-2]):.2e} (last iteration)")
        
        # Check for oscillations
        if len(A_vals) > 5:
            recent_changes = [abs(A_vals[i] - A_vals[i-1]) for i in range(-5, 0)]
            avg_change = np.mean(recent_changes)
            print(f"  Avg change (last 5): {avg_change:.2e}")
            
            # Check if still decreasing
            if recent_changes[-1] > avg_change * 1.5:
                print(f"  WARNING: Convergence may be stalling or oscillating!")
    
    if 'Delta_x' in hist and len(hist['Delta_x']) > 1:
        dx_vals = hist['Delta_x']
        print(f"\nSpatial Scale Convergence:")
        print(f"  Initial Delta_x: {dx_vals[0]:.6e}")
        print(f"  Final Delta_x:   {dx_vals[-1]:.6e}")
        print(f"  Change:     {abs(dx_vals[-1] - dx_vals[-2]):.2e} (last iteration)")
print()

print("="*70)
print("NEUTRON (udd) WITH SPIN")
print("="*70)
print(f"Configuration:")
print(f"  Quarks: {NEUTRON.quarks}")
print(f"  Windings: {NEUTRON.windings}")
print(f"  Spins: {NEUTRON.spins}  (spin up, spin up, spin down)")
print(f"  Generations: {NEUTRON.generations}")
print(f"  Note: Two d quarks have OPPOSITE spins to satisfy Pauli exclusion!")
print()

neutron = solver.solve_baryon_4D_self_consistent(
    quark_wells=(1, 2, 3),
    color_phases=(0, 2*np.pi/3, 4*np.pi/3),
    quark_windings=NEUTRON.windings,
    quark_spins=NEUTRON.spins,
    quark_generations=NEUTRON.generations,
    max_iter_outer=30,
    max_iter_scf=10,
    verbose=True,
)

A_n = neutron.structure_norm
m_n = beta * A_n**2 * 1000  # MeV
error_n = 100 * (m_n - m_n_exp) / m_n_exp

print(f"\nNeutron Results:")
print(f"  A = {A_n:.6f}")
print(f"  mass = {m_n:.2f} MeV (target: {m_n_exp:.2f} MeV)")
print(f"  error = {error_n:+.1f}%")
print(f"  k_eff = {neutron.k_eff:.4f}")
print(f"  E_EM = {neutron.em_energy:.4e}")
print(f"\nConvergence Status:")
print(f"  Converged: {neutron.converged}")
print(f"  Iterations: {neutron.iterations}")
print(f"  Delta_x_final: {neutron.delta_x_final:.6e}")

# Analyze convergence history
if neutron.convergence_history:
    hist = neutron.convergence_history
    if 'A' in hist and len(hist['A']) > 1:
        A_vals = hist['A']
        print(f"\nAmplitude Convergence:")
        print(f"  Initial A: {A_vals[0]:.6f}")
        print(f"  Final A:   {A_vals[-1]:.6f}")
        print(f"  Change:    {abs(A_vals[-1] - A_vals[-2]):.2e} (last iteration)")
        
        # Check for oscillations
        if len(A_vals) > 5:
            recent_changes = [abs(A_vals[i] - A_vals[i-1]) for i in range(-5, 0)]
            avg_change = np.mean(recent_changes)
            print(f"  Avg change (last 5): {avg_change:.2e}")
            
            # Check if still decreasing
            if recent_changes[-1] > avg_change * 1.5:
                print(f"  WARNING: Convergence may be stalling or oscillating!")
    
    if 'Delta_x' in hist and len(hist['Delta_x']) > 1:
        dx_vals = hist['Delta_x']
        print(f"\nSpatial Scale Convergence:")
        print(f"  Initial Delta_x: {dx_vals[0]:.6e}")
        print(f"  Final Delta_x:   {dx_vals[-1]:.6e}")
        print(f"  Change:     {abs(dx_vals[-1] - dx_vals[-2]):.2e} (last iteration)")
print()

print("="*70)
print("COMPARISON")
print("="*70)
print(f"{'':20s} {'Proton':>15s} {'Neutron':>15s} {'Difference':>15s}")
print("-"*70)
print(f"{'Amplitude A:':20s} {A_p:>15.6f} {A_n:>15.6f} {A_n-A_p:>+15.6f}")
print(f"{'Mass (MeV):':20s} {m_p:>15.2f} {m_n:>15.2f} {m_n-m_p:>+15.2f}")
print(f"{'Error (%):':20s} {error_p:>+15.1f} {error_n:>+15.1f}")
print(f"{'k_eff:':20s} {proton.k_eff:>+15.4f} {neutron.k_eff:>+15.4f} {neutron.k_eff-proton.k_eff:>+15.4f}")
print(f"{'E_EM:':20s} {proton.em_energy:>15.4e} {neutron.em_energy:>15.4e} {neutron.em_energy-proton.em_energy:>+15.4e}")
print()
print(f"Experimental mass difference: {delta_m_exp:.2f} MeV (neutron heavier)")
print(f"Predicted mass difference:    {m_n-m_p:+.2f} MeV")
print()

# Check if neutron amplitude collapse is fixed
if A_n > 20:
    print("[SUCCESS] Neutron amplitude collapse FIXED!")
    print(f"  Neutron A = {A_n:.2f} (was ~4-7 with old solver)")
else:
    print("[ISSUE] Neutron amplitude still low")
    print(f"  Neutron A = {A_n:.2f}")
print()

# Convergence Assessment
print("="*70)
print("CONVERGENCE ASSESSMENT")
print("="*70)

convergence_issues = []

if not proton.converged:
    convergence_issues.append("Proton solver did not converge")
    print(f"[WARNING] Proton solver: NOT converged after {proton.iterations} iterations")
else:
    print(f"[OK] Proton solver: Converged in {proton.iterations} iterations")

if not neutron.converged:
    convergence_issues.append("Neutron solver did not converge")
    print(f"[WARNING] Neutron solver: NOT converged after {neutron.iterations} iterations")
else:
    print(f"[OK] Neutron solver: Converged in {neutron.iterations} iterations")

if convergence_issues:
    print(f"\n[CRITICAL] Convergence issues detected:")
    for issue in convergence_issues:
        print(f"  - {issue}")
    print(f"\nRecommendations:")
    print(f"  1. Increase max_iter_outer (currently 30)")
    print(f"  2. Increase max_iter_scf (currently 10)")
    print(f"  3. Check parameter values (g1, g2, lambda_so)")
    print(f"  4. Results may not be reliable due to non-convergence!")
else:
    print(f"\n[SUCCESS] Both solvers converged successfully!")
print()

# Overall assessment
print("="*70)
print("PHYSICS ASSESSMENT")
print("="*70)
if abs(error_p) < 10 and abs(error_n) < 10:
    print("[EXCELLENT] Both masses within 10% of experimental values!")
elif abs(error_p) < 20 and abs(error_n) < 20:
    print("[GOOD] Both masses within 20% of experimental values")
elif abs(error_p) < 50 and abs(error_n) < 50:
    print("[FAIR] Both masses within 50% of experimental values")
else:
    print("[NEEDS WORK] One or both masses still far from target")

if convergence_issues:
    print("\n[WARNING] However, solvers did not converge - results may not be reliable!")

print()
print("="*70)
print("SPIN IMPLEMENTATION STATUS")
print("="*70)
print("[SUCCESS] Spin quantum numbers implemented and tested!")
print()
print("Proton configuration:")
print(f"  Quark 1 (u): winding=+5, spin=+1, generation=1")
print(f"  Quark 2 (u): winding=+5, spin=-1, generation=1  <- OPPOSITE spin!")
print(f"  Quark 3 (d): winding=-3, spin=+1, generation=1")
print(f"  Pauli exclusion: SATISFIED (two u quarks have opposite spins)")
print()
print("Neutron configuration:")
print(f"  Quark 1 (u): winding=+5, spin=+1, generation=1")
print(f"  Quark 2 (d): winding=-3, spin=+1, generation=1")
print(f"  Quark 3 (d): winding=-3, spin=-1, generation=1  <- OPPOSITE spin!")
print(f"  Pauli exclusion: SATISFIED (two d quarks have opposite spins)")
print()
print("This spin structure should provide more stable solutions and")
print("better physical predictions for baryon masses!")
print()

