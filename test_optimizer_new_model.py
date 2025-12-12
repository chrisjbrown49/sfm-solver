"""Test if optimizer can find parameters for the new harmonic oscillator model."""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer

# Experimental masses
EXP_MASSES = {
    'electron': 0.000511,
    'muon': 0.1057,
    'tau': 1.777,
}

def predict_masses(beta, alpha, kappa, g1, verbose=False):
    """Predict lepton masses with given parameters."""
    N = 64
    grid = SpectralGrid(N=N)
    V0 = 1.0
    potential = ThreeWellPotential(V0)
    g2 = 1.0 / (137.036 * 2)  # Derived from alpha_em
    
    minimizer = WavefunctionEnergyMinimizer(
        grid=grid,
        potential=potential,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        g1=g1,
        g2=g2
    )
    
    results = {}
    for n, name in [(1, 'electron'), (2, 'muon'), (3, 'tau')]:
        sigma = grid.sigma
        k_winding = 1
        envelope = np.exp(-(sigma - np.pi)**2 / 0.5) + 0.01
        chi_initial = envelope * np.exp(1j * k_winding * sigma)
        chi_initial = chi_initial / np.sqrt(np.sum(np.abs(chi_initial)**2) * grid.dsigma)
        
        try:
            result = minimizer.minimize(
                chi_initial=chi_initial,
                n_spatial=n,
                max_iter=5000,
                verbose=False
            )
            mass_gev = beta * result.A_squared
            results[name] = mass_gev
        except:
            results[name] = None
    
    return results

def objective(params):
    """Compute total squared relative error."""
    beta, alpha, g1 = params
    
    if beta <= 0 or alpha <= 0 or g1 <= 0:
        return 1e10
    
    # Kappa from first principles
    kappa = 1.0 / (beta ** 2)
    
    try:
        masses = predict_masses(beta, alpha, kappa, g1)
        
        error = 0
        for name, exp_mass in EXP_MASSES.items():
            pred = masses.get(name)
            if pred is None or pred <= 0:
                error += 100
            else:
                rel_error = ((pred - exp_mass) / exp_mass) ** 2
                error += rel_error
        return error
    except:
        return 1e10

def objective_beta_only(beta):
    """Objective function searching only over beta."""
    if beta <= 0:
        return 1e10
    
    # Derive other parameters from beta (first principles)
    alpha = 0.5 * beta
    kappa = 1.0 / (beta ** 2)
    alpha_em = 1.0 / 137.036
    m_e = 0.000511
    g1 = alpha_em * beta / m_e
    
    try:
        masses = predict_masses(beta, alpha, kappa, g1)
        
        error = 0
        count = 0
        for name, exp_mass in EXP_MASSES.items():
            pred = masses.get(name)
            if pred is not None and pred > 0:
                rel_error = ((pred - exp_mass) / exp_mass) ** 2
                error += rel_error
                count += 1
        
        if count == 0:
            return 1e10
        return error
    except Exception as e:
        print(f"Error: {e}")
        return 1e10

print("=" * 70)
print("TESTING PARAMETER OPTIMIZATION WITH HARMONIC OSCILLATOR MODEL")
print("=" * 70)

# First, let's see what happens with different beta values
print("\n1. Scanning beta with first-principles derived parameters:")
print("-" * 70)

beta_values = [1, 5, 10, 50, 100, 200, 500]
for beta in beta_values:
    alpha = 0.5 * beta
    kappa = 1.0 / (beta ** 2)
    alpha_em = 1.0 / 137.036
    m_e = 0.000511
    g1 = alpha_em * beta / m_e
    
    masses = predict_masses(beta, alpha, kappa, g1)
    
    if all(m is not None for m in masses.values()):
        m_e_pred = masses['electron']
        m_mu_pred = masses['muon']
        m_tau_pred = masses['tau']
        
        ratio_mu_e = m_mu_pred / m_e_pred if m_e_pred > 0 else 0
        ratio_tau_e = m_tau_pred / m_e_pred if m_e_pred > 0 else 0
        
        print(f"beta={beta:4.0f}: m_e={m_e_pred:.2e} GeV, "
              f"m_mu/m_e={ratio_mu_e:.2f}, m_tau/m_e={ratio_tau_e:.2f}")
    else:
        print(f"beta={beta:4.0f}: solver failed")

# Now try optimizing beta
print("\n2. Optimizing beta to minimize mass prediction error:")
print("-" * 70)

from scipy.optimize import minimize_scalar
result = minimize_scalar(
    objective_beta_only,
    bounds=(0.001, 1000),
    method='bounded',
    options={'maxiter': 50}
)

beta_opt = result.x
print(f"Optimal beta: {beta_opt:.4f} GeV")
print(f"Final error: {result.fun:.4f}")

# Show results at optimal beta
alpha_opt = 0.5 * beta_opt
kappa_opt = 1.0 / (beta_opt ** 2)
alpha_em = 1.0 / 137.036
m_e_gev = 0.000511
g1_opt = alpha_em * beta_opt / m_e_gev

print(f"\nDerived parameters:")
print(f"  alpha = {alpha_opt:.4f} GeV")
print(f"  kappa = {kappa_opt:.6e} GeV^-2")
print(f"  g1 = {g1_opt:.2f}")

masses = predict_masses(beta_opt, alpha_opt, kappa_opt, g1_opt)
print(f"\nPredicted masses at optimal beta:")
for name, exp in EXP_MASSES.items():
    pred = masses.get(name, 0)
    error = abs(pred - exp) / exp * 100 if pred > 0 else 100
    print(f"  {name}: {pred:.6f} GeV (exp: {exp:.6f}, error: {error:.1f}%)")

if masses['electron'] and masses['muon'] and masses['tau']:
    m_e = masses['electron']
    m_mu = masses['muon']
    m_tau = masses['tau']
    print(f"\nMass ratios:")
    print(f"  m_mu/m_e = {m_mu/m_e:.2f} (exp: 206.77)")
    print(f"  m_tau/m_e = {m_tau/m_e:.2f} (exp: 3477.23)")

print("\n" + "=" * 70)
print("CONCLUSION: Can parameter tuning fix the mass hierarchy?")
print("=" * 70)

