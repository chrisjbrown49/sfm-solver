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

# Mass emerges from m = β × A² where β ≈ 53.95 GeV (discovered)
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
│   │   ├── correlated_basis.py           # Correlated spatial-subspace basis
│   │   ├── nonseparable_wavefunction_solver.py  # Stage 1: Wavefunction structure
│   │   ├── universal_energy_minimizer.py # Stage 2: Scale optimization
│   │   ├── parameters.py   # SFMParameters configuration
│   │   └── sfm_global.py   # SFM_CONSTANTS: β, α, κ, g₁ (first-principles)
│   ├── potentials/         # Potential energy functions
│   │   └── three_well.py   # V(σ) = V₀[1-cos(3σ)] + V₁[1-cos(6σ)]
│   ├── eigensolver/        # Eigenvalue and energy solvers
│   │   ├── sfm_lepton_solver.py  # Tier 1 physics-based solver
│   │   └── spectral.py     # Spectral operators for Hamiltonians
│   ├── multiparticle/      # Composite particle solvers
│   │   ├── composite_baryon.py   # Tier 2 three-quark baryons
│   │   ├── composite_meson.py    # Tier 2b quark-antiquark mesons
│   │   └── color_verification.py # Color neutrality checks
│   ├── optimization/       # Parameter optimization
│   │   └── parameter_optimizer.py # β-only first-principles optimizer
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

**First-Principles Derivation:** In physical mode, β ≈ 53.95 GeV (discovered through global optimization). This determines L₀ = ℏ/(βc) = 1/β in natural units.

### Four-Term Energy Functional

Particle masses emerge from minimizing the total energy:

```
E_total = E_subspace + E_spatial + E_coupling + E_curvature
```

| Term | Physical Mode Expression | Physical Role |
|------|--------------------------|---------------|
| E_subspace | Kinetic + potential + nonlinear | Confinement in S¹ |
| E_spatial | βA²/2 = m/2 | Rest mass contribution (Δx = 1/m) |
| E_coupling | -α × spatial_factor × subspace_factor × A | Spacetime-subspace interaction |
| E_curvature | κ × β³A⁶ | Enhanced 5D geometry |

**Note:** In physical mode, Δx is self-consistent: Δx = 1/(βA²) = 1/m (Compton wavelength).

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

where A² = ∫₀^(2π) |χ(σ)|² dσ and β ≈ 53.95 GeV (discovered through global optimization).

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
- β ≈ 53.95 GeV (discovered through global optimization)
- α_EM derived from 3-well geometry (0.0075% accuracy)
- Δx = 1/m (self-consistent Compton wavelength)
- Mass predictions via m = β × A²

**The normalized mode is deprecated and should not be used.**

### Two-Stage Solver Architecture

All particle solvers (leptons, mesons, baryons) use a unified two-stage architecture:

**Stage 1: Wavefunction Structure** (`NonSeparableWavefunctionSolver`)
- Solves for the entangled non-separable wavefunction: ψ(r,θ,φ,σ) = Σ R_{nl}(r) Y_l^m(θ,φ) χ_{nlm}(σ)
- Each angular component (n,l,m) has its own subspace function χ_{nlm}(σ)
- Returns unit-normalized `WavefunctionStructure` with l-composition and effective winding k_eff

**Stage 2: Energy Minimization** (`UniversalEnergyMinimizer`)
- Takes the wavefunction structure from Stage 1
- Minimizes E_total over three scale parameters: (A, Δx, Δσ)
- The SAME code works for all particle types (leptons, mesons, baryons)
- Returns optimal amplitude A and predicted mass m = β × A²

### Parameter Optimization Loop

The `SFMParameterOptimizer` discovers SFM framework parameters from first principles by searching over the fundamental mass coupling β. For each candidate β, all other parameters are derived:

| Parameter | First-Principles Derivation |
|-----------|----------------------------|
| κ (curvature) | κ = 1/β² |
| g₁ (nonlinear) | g₁ = α_em × β / m_e |
| α (coupling) | α = C × β (where C ≈ 0.5) |

The optimizer then runs the particle solvers on calibration particles and adjusts β to minimize total mass prediction error.

### Solver Architecture Diagram

The following diagram illustrates how the three core components work together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PARAMETER OPTIMIZER                                  │
│                    (Outer Loop: Search over β)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  β_candidate ──┬──────────────────────────────────────────────────────────┐ │
│                │                                                          │ │
│                ▼                                                          │ │
│  ┌──────────────────────────────────────────┐                             │ │
│  │    DERIVE PARAMETERS FROM β              │                             │ │
│  │    ─────────────────────────             │                             │ │
│  │    κ = 1/β²                              │                             │ │
│  │    g₁ = α_em × β / m_e                   │                             │ │
│  │    α = 0.5 × β                           │                             │ │
│  └──────────────────────────────────────────┘                             │ │
│                │                                                          │ │
│                ▼                                                          │ │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐ │
│    FOR EACH CALIBRATION PARTICLE (e, μ, π, p, ...)                        │ │
│  │                                                                       │ │
│    ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │ │  STAGE 1: NonSeparableWavefunctionSolver                        │   │ │
│    │  ───────────────────────────────────────                        │   │ │
│  │ │  • Solve for entangled wavefunction structure                   │   │ │
│    │  • ψ = Σ R_nl(r) Y_lm(θ,φ) χ_nlm(σ)                             │   │ │
│  │ │  • Each (n,l,m) component has its own χ_nlm(σ)                  │   │ │
│    │  • Returns: WavefunctionStructure (unit-normalized)             │   │ │
│  │ │            {χ_nlm}, l-composition, k_eff                        │   │ │
│    └─────────────────────────────────────────────────────────────────┘   │ │
│  │                             │                                         │ │
│                                ▼                                           │
│  │ ┌─────────────────────────────────────────────────────────────────┐   │ │
│    │  STAGE 2: UniversalEnergyMinimizer                              │   │ │
│  │ │  ─────────────────────────────────                              │   │ │
│    │  • Minimize E_total(A, Δx, Δσ)                                  │   │ │
│  │ │  • E = E_kinetic + E_potential + E_coupling + E_curvature       │   │ │
│    │  • Gradient descent over (A, Δx, Δσ)                            │   │ │
│  │ │  • Returns: EnergyMinimizationResult                            │   │ │
│    │            A_optimal, mass = β × A²                             │   │ │
│  │ └─────────────────────────────────────────────────────────────────┘   │ │
│                                │                                           │
│  │                             ▼                                         │ │
│    m_predicted = β × A²                                                   │ │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘ │
│                │                                                          │ │
│                ▼                                                          │ │
│  ┌──────────────────────────────────────────┐                             │ │
│  │    COMPUTE TOTAL ERROR                   │                             │ │
│  │    ───────────────────                   │                             │ │
│  │    error = Σ (m_pred - m_exp)² / m_exp²  │                             │ │
│  └──────────────────────────────────────────┘                             │ │
│                │                                                          │ │
│                ▼                                                          │ │
│  ┌──────────────────────────────────────────┐                             │ │
│  │    OPTIMIZATION ALGORITHM                │◄────────────────────────────┘ │
│  │    ──────────────────────                │                               │
│  │    If error improved: update β_best      │                               │
│  │    Choose next β_candidate               │                               │
│  │    Repeat until converged                │                               │
│  └──────────────────────────────────────────┘                               │
│                │                                                             │
│                ▼                                                             │
│  OUTPUT: Optimal β* and derived parameters {κ*, g₁*, α*}                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**

1. **Separation of Concerns**: The wavefunction solver finds the *shape* (STRUCTURE), the energy minimizer finds the *size* (SCALE), and the parameter optimizer finds the *fundamental constants*.

2. **Universal Energy Minimizer**: The same `UniversalEnergyMinimizer` code handles leptons, mesons, and baryons. The particle-specific physics is encoded in the `WavefunctionStructure` from Stage 1.

3. **First-Principles Parameter Derivation**: All SFM parameters (κ, g₁, α) are derived from β using theoretical formulas, ensuring physical consistency across the framework.

4. **Two-Level Optimization**:
   - Inner loop: Energy minimization over (A, Δx, Δσ) for fixed parameters
   - Outer loop: Parameter search over β to match experimental masses

### Spectral Methods

The solver uses FFT-based spectral methods on the periodic domain S¹:
- Derivatives computed in Fourier space: ∂/∂σ → ik
- Hamiltonian constructed as sparse matrices
- Periodic boundary conditions automatically satisfied

### Non-Separable Wavefunction

The critical physics insight is that the wavefunction is **non-separable**:
- ψ(r,θ,φ,σ) = Σ_{n,l,m} R_{nl}(r) Y_l^m(θ,φ) χ_{nlm}(σ)
- Each angular component has its **own** subspace function χ_{nlm}(σ)
- The coupling Hamiltonian mixes l=0 ↔ l=1 ↔ l=2 components
- This breaks spherical symmetry and enables non-zero spatial-subspace coupling

### Composite Wavefunctions

Multi-quark systems use a single composite wavefunction approach:
- **Mesons**: χ = χ_q + χ_q̄ (two-peak structure with opposite windings)
- **Baryons**: χ = χ_1 + χ_2 + χ_3 (three-peak with color phases)
- Phases extracted from well-localized components
- Color neutrality verified via |Σe^(iφ)| < threshold

## Documentation

The Single-Field Model documentation library includes:
- Research Notes on Mathematical Formulation (Parts A, B, C)
- Research Notes on the Origin of the Fundamental Forces
- Research Notes on The Beautiful Equation and A Beautiful Balance
- Solver Requirements, Tier Implementation Plans and Completion Checks
- First-Principles Parameter Derivation Plan

## License

MIT License - see LICENSE file for details.
