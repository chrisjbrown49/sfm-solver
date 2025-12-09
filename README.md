# SFM Solver - Single-Field Model Numerical Solver

A Python implementation of numerical solvers for the Single-Field Model (SFM) framework. This solver computes particle properties from the SFM mathematical formulation using physics-based energy functionals on an extended spacetime that includes a compact circular subspace S¹.

## Overview

The Single-Field Model proposes that all fundamental particles emerge from a single scalar field χ(x,σ) defined on an extended spacetime M⁴ × S¹. Different particles correspond to different winding modes and spatial configurations of this field. The solver is organized into **Tiers** that progressively build from fundamental leptons to composite hadrons.

### Tier Architecture

| Tier | Particles | Physics |
|------|-----------|---------|
| **Tier 1** | Leptons (e, μ, τ) | Mass hierarchy from four-term energy functional |
| **Tier 1b** | Electromagnetic forces | Charge quantization Q = e/k from winding number |
| **Tier 2** | Baryons (p, n) | Color emergence from three-quark composite wavefunctions |
| **Tier 2b** | Mesons (π, J/ψ, Υ) | Quark-antiquark bound states with radial excitations |

## Installation

```bash
# Clone the repository
git clone https://github.com/sfm-project/sfm-solver.git
cd sfm-solver

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install in development mode
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Tier 1: Lepton Masses

```python
from sfm_solver.eigensolver import SFMLeptonSolver

# Create physics-based lepton solver
solver = SFMLeptonSolver()

# Solve for all three charged leptons
results = solver.solve_lepton_spectrum(verbose=True)

# Compute mass ratios (emerge from energy minimization)
ratios = solver.compute_mass_ratios(results)
print(f"m_μ/m_e = {ratios['mu_e']:.2f}")
print(f"m_τ/m_e = {ratios['tau_e']:.1f}")
```

### Tier 2: Composite Baryons

```python
from sfm_solver.multiparticle import CompositeBaryon

# Create proton solver (uud quarks)
proton = CompositeBaryon(quark_types=['u', 'u', 'd'])
state = proton.solve(verbose=True)

# Color phases emerge from energy minimization
phases = proton.extract_color_phases()
print(f"Color neutral: {proton.is_color_neutral()}")
```

## Project Structure

```
sfm-solver/
├── src/sfm_solver/
│   ├── core/               # Fundamental constants and infrastructure
│   │   ├── constants.py    # Physical constants (masses, α, etc.)
│   │   ├── grid.py         # SpectralGrid for FFT operations on S¹
│   │   ├── parameters.py   # SFMParameters configuration
│   │   └── sfm_global.py   # Global β constant (SFM_CONSTANTS)
│   ├── potentials/         # Potential energy functions
│   │   └── three_well.py   # V(σ) = V₀[1-cos(3σ)] + V₁[1-cos(6σ)]
│   ├── eigensolver/        # Eigenvalue and energy solvers
│   │   ├── sfm_lepton_solver.py  # Tier 1 physics-based solver
│   │   └── spectral.py     # Spectral operators for Hamiltonians
│   ├── multiparticle/      # Composite particle solvers
│   │   ├── composite_baryon.py   # Tier 2 three-quark baryons
│   │   ├── composite_meson.py    # Tier 2b quark-antiquark mesons
│   │   └── color_verification.py # Color neutrality checks
│   ├── forces/             # Force calculations
│   │   └── electromagnetic.py    # Tier 1b EM force mechanism
│   ├── spatial/            # Spatial dimension handling
│   │   └── radial.py       # Radial excitation scaling (WKB)
│   ├── analysis/           # Physics analysis tools
│   │   └── mass_spectrum.py      # Mass extraction and calibration
│   ├── validation/         # Experimental comparison
│   │   └── testbench.py    # PDG value validation
│   ├── reporting/          # Results generation
│   │   ├── results_reporter.py   # Markdown report generation
│   │   └── results_viewer.py     # HTML report generation
│   └── legacy/             # Deprecated solvers (for reference)
├── tests/                  # Comprehensive test suite
│   ├── test_tier1_leptons.py     # Lepton solver tests
│   ├── test_tier1b_em_forces.py  # EM force tests
│   ├── test_tier2_hadrons.py     # Baryon/meson tests
│   ├── test_tier2_color.py       # Color emergence tests
│   └── test_tier2b_quarkonia.py  # Radial excitation tests
├── outputs/                # Generated results
├── docs/                   # Research notes and documentation
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Physics Concepts

### The Beautiful Equation

The SFM framework is constrained by the fundamental relation:

```
β L₀ c = ℏ
```

where β is the mass-amplitude coupling, L₀ is the subspace radius, c is the speed of light, and ℏ is the reduced Planck constant. The constant β is calibrated from the electron mass via m = βA².

### Four-Term Energy Functional

Particle masses emerge from minimizing the total energy:

```
E_total = E_subspace + E_spatial + E_coupling + E_curvature
```

| Term | Expression | Physical Role |
|------|------------|---------------|
| E_subspace | Kinetic + potential + nonlinear | Confinement in S¹ |
| E_spatial | ℏ²/(2βA²Δx²) | Prevents collapse to A→0 |
| E_coupling | -α × f(n) × k_eff × A | Subspace-spatial interaction |
| E_curvature | κ(βA²)²/Δx | Enhanced 5D gravity |

### Winding Number and Charge

Electric charge emerges from the winding number k of the subspace mode:

```
Q = ±e/k
```

| Winding k | Particle Type | Charge |
|-----------|---------------|--------|
| k = 1 | Leptons | Q = ±e |
| k = 3 | Down-type quarks | Q = ∓e/3 |
| k = 5 | Up-type quarks | Q = ±2e/3 |

### Mass Formula

Particle mass is determined by the subspace amplitude squared:

```
m = β A²
```

where A² = ∫₀^(2π) |χ(σ)|² dσ. The global constant β is calibrated once from the electron mass and then used consistently across all particle sectors.

### Composite Particles

Baryons and mesons are modeled as composite wavefunctions:

- **Baryons**: Three-quark wavefunctions with emergent color phases {0, 2π/3, 4π/3}
- **Mesons**: Quark-antiquark pairs with effective winding k_eff from interference
- **Radial Excitations**: WKB-derived scaling Δx_n = Δx₀ × n^(2/3)

## Running Tests

```bash
# Run all tests with results report generation
pytest tests/ -v

# Run specific tier tests
pytest tests/test_tier1_leptons.py -v
pytest tests/test_tier2_hadrons.py -v

# Run with coverage
pytest --cov=sfm_solver

# View generated reports
# Markdown: outputs/sfm_results_*.md
# HTML: outputs/results.html
```

## Implementation Methods

### Spectral Methods

The solver uses FFT-based spectral methods on the periodic domain S¹:
- Derivatives computed in Fourier space: ∂/∂σ → ik
- Hamiltonian constructed as sparse matrices
- Periodic boundary conditions automatically satisfied

### Energy Minimization

Particle states are found by minimizing the total energy functional:
- Joint optimization over amplitude A and spatial extent Δx
- Gradient descent with adaptive step sizes
- Convergence monitored via residual norm

### Composite Wavefunctions

Multi-quark systems use a single composite wavefunction approach:
- Phases extracted from well-localized components
- Color neutrality verified via |Σe^(iφ)| < threshold
- Coupling energy scales linearly with amplitude (prevents collapse)

## Documentation

The Single-Field Model documentation library includes:
- Research Notes on Mathematical Formulation (Parts A, B, C)
- Research Notes on the Origin of the Fundamental Forces
- Research Notes on The Beautiful Equation and A Beautiful Balance
- Solver Requirements, Tier Implementation Plans and Completion Checks

## License

MIT License - see LICENSE file for details.
