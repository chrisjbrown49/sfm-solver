"""
2D scan of g1 and lambda_so for neutron mass prediction.
"""
import numpy as np
from sfm_solver.core.nonseparable_wavefunction_solver import NonSeparableWavefunctionSolver
from sfm_solver.core.constants import ALPHA, G_INTERNAL, G2, V0, V1

# First, solve for electron to get beta
print("Solving for electron to derive beta...")
solver_e = NonSeparableWavefunctionSolver(
    alpha=ALPHA, beta=0.000511, g_internal=G_INTERNAL,
    g1=1.0, g2=G2, lambda_so=0.2, V0=V0, V1=V1,
    n_max=5, l_max=2, N_sigma=64,
)
electron = solver_e.solve_lepton_self_consistent(n_target=1, max_iter_outer=30, max_iter_nl=0, verbose=False)
m_e_exp = 0.511  # MeV
A_e = electron.structure_norm
beta = 0.000511 / (A_e**2)  # GeV

print(f"Electron: A = {A_e:.6f}, beta = {beta:.6f} GeV\n")

# Target masses
m_p_exp = 938.27  # MeV
m_n_exp = 939.57  # MeV

# Parameter ranges
g1_values = [0.5, 1.0, 2.0]
lambda_so_values = [0.1, 0.2, 0.5, 1.0, 2.0]

print("=" * 80)
print("2D SCAN: g1 vs lambda_so")
print("=" * 80)
print(f"{'g1':>6s} {'lam_so':>6s} {'A_p':>8s} {'m_p':>10s} {'err_p%':>8s} | "
      f"{'A_n':>8s} {'m_n':>10s} {'err_n%':>8s} | {'Dm':>8s}")
print("-" * 80)

results = []

for g1 in g1_values:
    for lambda_so in lambda_so_values:
        try:
            solver = NonSeparableWavefunctionSolver(
                alpha=ALPHA,
                beta=beta,
                g_internal=G_INTERNAL,
                g1=g1,
                g2=G2,
                lambda_so=lambda_so,
                V0=V0,
                V1=V1,
                n_max=5,
                l_max=2,
                N_sigma=64,
            )
            
            # Solve proton (uud)
            proton = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3),
                color_phases=(0, 2*np.pi/3, 4*np.pi/3),
                quark_windings=(5, 5, -3),
                max_iter_outer=30,
                verbose=False,
            )
            
            # Solve neutron (udd)
            neutron = solver.solve_baryon_self_consistent(
                quark_wells=(1, 2, 3),
                color_phases=(0, 2*np.pi/3, 4*np.pi/3),
                quark_windings=(5, -3, -3),
                max_iter_outer=30,
                verbose=False,
            )
            
            # Compute masses
            A_p = proton.structure_norm
            A_n = neutron.structure_norm
            m_p = beta * A_p**2 * 1000  # MeV
            m_n = beta * A_n**2 * 1000  # MeV
            delta_m = m_n - m_p
            
            error_p = 100 * (m_p - m_p_exp) / m_p_exp
            error_n = 100 * (m_n - m_n_exp) / m_n_exp
            
            print(f"{g1:>6.1f} {lambda_so:>6.2f} {A_p:>8.4f} {m_p:>10.2f} {error_p:>+7.1f}% | "
                  f"{A_n:>8.4f} {m_n:>10.2f} {error_n:>+7.1f}% | {delta_m:>+7.2f}")
            
            results.append({
                'g1': g1,
                'lambda_so': lambda_so,
                'A_p': A_p,
                'A_n': A_n,
                'm_p': m_p,
                'm_n': m_n,
                'error_p': error_p,
                'error_n': error_n,
                'delta_m': delta_m,
            })
            
        except Exception as e:
            print(f"{g1:>6.1f} {lambda_so:>6.2f} {'ERROR':>8s} {str(e)[:40]:>10s}")

# Find best results
print("\n" + "=" * 80)
print("BEST RESULTS")
print("=" * 80)

valid = [r for r in results if not np.isnan(r['m_p'])]

if valid:
    # Best proton
    best_p = min(valid, key=lambda x: abs(x['error_p']))
    print(f"\nBest proton: g1={best_p['g1']}, lam_so={best_p['lambda_so']}")
    print(f"  m_p = {best_p['m_p']:.2f} MeV (error = {best_p['error_p']:+.1f}%)")
    
    # Best neutron
    best_n = min(valid, key=lambda x: abs(x['error_n']))
    print(f"\nBest neutron: g1={best_n['g1']}, lam_so={best_n['lambda_so']}")
    print(f"  m_n = {best_n['m_n']:.2f} MeV (error = {best_n['error_n']:+.1f}%)")
    
    # Best combined (minimize sum of squared errors)
    best_combined = min(valid, key=lambda x: x['error_p']**2 + x['error_n']**2)
    print(f"\nBest combined: g1={best_combined['g1']}, lam_so={best_combined['lambda_so']}")
    print(f"  m_p = {best_combined['m_p']:.2f} MeV (error = {best_combined['error_p']:+.1f}%)")
    print(f"  m_n = {best_combined['m_n']:.2f} MeV (error = {best_combined['error_n']:+.1f}%)")
    print(f"  Dm = {best_combined['delta_m']:+.2f} MeV (target: +1.30 MeV)")

