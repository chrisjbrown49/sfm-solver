# SFM Solver - Single-Field Model Numerical Solver

A Python implementation of numerical solvers for the Single-Field Model (SFM) framework. This solver computes particle properties from first principles using physics-based energy functionals on an extended spacetime that includes a compact circular subspace S¹.

**⚠️ IMPORTANT:** All solvers now operate in **physical mode** (`use_physical=True`) by default. The normalized mode is **deprecated** and should not be used for new development.

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

# Create physics-based lepton solver (physical mode by default)
solver = SFMLeptonSolver()  # use_physical=True is the default

# Solve for all three charged leptons
results = solver.solve_lepton_spectrum(verbose=True)

# Mass emerges from m = β × A² where β = M_W (W boson mass)
from sfm_solver.core.sfm_global import SFM_CONSTANTS
m_electron = SFM_CONSTANTS.beta_physical * results['electron'].amplitude_squared
```

### Tier 2: Composite Mesons

```python
from sfm_solver.multiparticle import CompositeMesonSolver
from sfm_solver.core.grid import SpectralGrid
from sfm_solver.potentials import ThreeWellPotential

# Create pion solver (physical mode by default)
grid = SpectralGrid(N=256)
potential = ThreeWellPotential(V0=1.0)
solver = CompositeMesonSolver(grid, potential)  # use_physical=True

# Solve for pion - mass emerges from m = β × A²
pion = solver.solve('pion_plus', verbose=True)
print(f"Predicted pion mass: {SFM_CONSTANTS.beta_physical * pion.amplitude_squared * 1000:.1f} MeV")
```

## Project Structure

```
sfm-solver/
├── src/sfm_solver/
│   ├── core/               # Fundamental constants and infrastructure
│   │   ├── constants.py    # Physical constants (masses, α, etc.)
│   │   ├── grid.py         # SpectralGrid for FFT operations on S¹
│   │   ├── parameters.py   # SFMParameters configuration
│   │   └── sfm_global.py   # SFM_CONSTANTS: β, α_EM, g₂ (first-principles)
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

where β is the mass-amplitude coupling, L₀ is the subspace radius, c is the speed of light, and ℏ is the reduced Planck constant.

**First-Principles Derivation:** In physical mode, β = M_W ≈ 80.38 GeV (from W boson self-consistency). This determines L₀ = ℏ/(βc) = 1/β in natural units.

### Four-Term Energy Functional

Particle masses emerge from minimizing the total energy:

```
E_total = E_subspace + E_spatial + E_coupling + E_curvature
```

| Term | Physical Mode Expression | Physical Role |
|------|--------------------------|---------------|
| E_subspace | Kinetic + potential + nonlinear | Confinement in S¹ |
| E_spatial | βA²/2 = m/2 | Rest mass contribution (Δx = 1/m) |
| E_coupling | -α × n^p × A / k^(5/6) | 3-well interference suppression |
| E_curvature | κ × β³A⁶ | Enhanced 5D gravity |

**Note:** In physical mode, Δx is self-consistent: Δx = 1/(βA²) = 1/m (Compton wavelength). The k^(5/6) suppression emerges from 3-well geometry.

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
m = β × A²
```

where A² = ∫₀^(2π) |χ(σ)|² dσ and β = M_W ≈ 80.38 GeV (from W boson self-consistency).

### Fine Structure Constant (First-Principles)

The fine structure constant α_EM is **derived** from SFM geometry, not input:

```
α_EM = √(8π × m_e / (3 × β))
```

This gives α_EM ≈ 1/137.03 with **0.0075% error** (0.55 ppm) compared to experiment. The factor 8π/3 emerges from the 3-well potential geometry.

### Composite Particles

Baryons and mesons are modeled as composite wavefunctions:

- **Baryons**: Three-quark wavefunctions with emergent color phases {0, 2π/3, 4π/3}
- **Mesons**: Quark-antiquark pairs with effective winding k_eff from interference
- **Radial Excitations**: WKB-derived scaling Δx_n = Δx₀ × n^(2/3)

## Running Tests

```bash
# Run all tests (214 tests, physical mode)
pytest tests/ -v

# Run specific tier tests
pytest tests/test_tier1_leptons.py -v      # Lepton solver
pytest tests/test_tier1b_em_forces.py -v   # EM forces, α_EM derivation
pytest tests/test_tier2_hadrons.py -v      # Baryons and mesons
pytest tests/test_tier2b_quarkonia.py -v   # Radial excitations

# View generated reports
# HTML: outputs/results.html
# Markdown: outputs/sfm_results_*.md
```

## Implementation Methods

### Physical Mode (Default)

All solvers operate in physical mode (`use_physical=True`) by default:
- β = M_W ≈ 80.38 GeV (W boson self-consistency)
- α_EM derived from 3-well geometry (0.0075% accuracy)
- Δx = 1/m (self-consistent Compton wavelength)
- Mass predictions via m = β × A²

**The normalized mode is deprecated and should not be used.**

### Spectral Methods

The solver uses FFT-based spectral methods on the periodic domain S¹:
- Derivatives computed in Fourier space: ∂/∂σ → ik
- Hamiltonian constructed as sparse matrices
- Periodic boundary conditions automatically satisfied

### Energy Minimization

Particle states are found by minimizing the four-term energy functional:
- Joint optimization over amplitude A and wavefunction χ(σ)
- Gradient descent with mode-specific gradients
- Convergence monitored via residual norm

### Composite Wavefunctions

Multi-quark systems use a single composite wavefunction approach:
- Phases extracted from well-localized components
- Color neutrality verified via |Σe^(iφ)| < threshold
- k^(5/6) coupling suppression from 3-well interference

## Documentation

The Single-Field Model documentation library includes:
- Research Notes on Mathematical Formulation (Parts A, B, C)
- Research Notes on the Origin of the Fundamental Forces
- Research Notes on The Beautiful Equation and A Beautiful Balance
- Solver Requirements, Tier Implementation Plans and Completion Checks

## License

MIT License - see LICENSE file for details.
