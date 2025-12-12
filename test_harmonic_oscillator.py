"""Test harmonic oscillator spatial wavefunctions for mass hierarchy."""
import numpy as np
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials.three_well import ThreeWellPotential
from sfm_solver.core.energy_minimizer import WavefunctionEnergyMinimizer

# Create grid and potential
N = 128
grid = SpectralGrid(N=N)
V0 = 1.0
potential = ThreeWellPotential(V0)

# Get fundamental parameters
beta = 110.27  # From our optimization
alpha = 0.5 * beta  # ~ 55 GeV
kappa = 1.0 / (beta ** 2)
alpha_em = 1.0 / 137.036
electron_mass_gev = 0.000511
g1 = alpha_em * beta / electron_mass_gev  # ~ 1575
g2 = alpha_em / 2

print('Testing Lepton Mass Hierarchy with Harmonic Oscillator Wavefunctions')
print('=' * 70)
print(f'Parameters: beta={beta:.2f}, alpha={alpha:.2f}, kappa={kappa:.2e}, g1={g1:.0f}')
print()

# Create minimizer
minimizer = WavefunctionEnergyMinimizer(
    grid=grid,
    potential=potential,
    alpha=alpha,
    beta=beta,
    kappa=kappa,
    g1=g1,
    g2=g2
)

# Test for n=1, 2, 3 (electron, muon, tau)
names = ['Electron', 'Muon', 'Tau']
exp_masses = [0.000511, 0.1057, 1.777]
results = []

for n, name, exp_mass in zip([1, 2, 3], names, exp_masses):
    # Initial wavefunction: Gaussian with winding k=1
    sigma = grid.sigma
    k_winding = 1
    
    # Initial chi with winding
    envelope = np.exp(-(sigma - np.pi)**2 / 0.5) + 0.01
    chi_initial = envelope * np.exp(1j * k_winding * sigma)
    chi_initial = chi_initial / np.sqrt(np.sum(np.abs(chi_initial)**2) * grid.dsigma)
    
    print(f'{name} (n={n}, k={k_winding}):')
    
    result = minimizer.minimize(
        chi_initial=chi_initial,
        n_spatial=n,
        max_iter=10000,
        verbose=False
    )
    
    # Mass from amplitude
    mass_gev = beta * result.A_squared
    
    print(f'  A = {result.A:.6f}, A^2 = {result.A_squared:.6e}')
    print(f'  E_coupling = {result.energy.E_coupling:.4g}')
    print(f'  Spatial factor = {result.energy.spatial_factor:.4f}')
    print(f'  Subspace factor = {result.energy.subspace_factor:.4f}')
    print(f'  Predicted mass = {mass_gev:.6f} GeV')
    print(f'  Experimental   = {exp_mass:.6f} GeV')
    print(f'  Converged: {result.converged} after {result.iterations} iterations')
    print()
    
    results.append({'name': name, 'n': n, 'mass': mass_gev, 'exp': exp_mass, 'A': result.A})

# Check mass ratios
print('Mass Hierarchy Check:')
print('-' * 40)
if len(results) == 3:
    m_e = results[0]['mass']
    m_mu = results[1]['mass']
    m_tau = results[2]['mass']
    
    ratio_mu_e = m_mu / m_e if m_e > 0 else 0
    ratio_tau_e = m_tau / m_e if m_e > 0 else 0
    
    exp_ratio_mu_e = 206.768
    exp_ratio_tau_e = 3477.23
    
    print(f'Predicted m_mu/m_e = {ratio_mu_e:.2f}  (exp: {exp_ratio_mu_e})')
    print(f'Predicted m_tau/m_e = {ratio_tau_e:.2f}  (exp: {exp_ratio_tau_e})')
    
    if m_mu > m_e and m_tau > m_mu:
        print('[OK] Mass ordering correct: m_tau > m_mu > m_e')
    else:
        print('[WRONG] Mass ordering incorrect!')

