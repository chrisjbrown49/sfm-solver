# SFT Solver - Single-Field Theory Tier 1 Eigenvalue Solver

A Python implementation of the Tier 1 numerical solver for the Single-Field Theory (SFT) framework. This solver computes eigenvalues and eigenfunctions for particles in the SFT three-well potential on the S¹ subspace.

## Overview

The Single-Field Theory proposes that all fundamental particles emerge from a single scalar field defined on an extended spacetime that includes a compact circular subspace S¹. Different particles correspond to different winding modes of this field around the subspace.

This Tier 1 solver implements:

- **Three-Well Potential**: V(σ) = V₀[1 - cos(3σ)] + V₁[1 - cos(6σ)]
- **Spectral Grid Methods**: FFT-based differentiation on S¹
- **Linear Eigenvalue Solver**: Sparse matrix methods for finding lowest modes
- **Nonlinear Self-Consistent Solver**: Iterative solution including |χ|² nonlinearity
- **Electromagnetic Forces**: Circulation integrals and energy calculations
- **Mass Spectrum Analysis**: Extraction of particle masses from amplitudes

## Installation

```bash
# Clone the repository
git clone https://github.com/sft-project/sft-solver.git
cd sft-solver

# Install in development mode
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```python
from sft_solver.core import SFTParameters, SpectralGrid
from sft_solver.potentials import ThreeWellPotential
from sft_solver.eigensolver import LinearEigensolver

# Set up parameters
params = SFTParameters(
    beta=50.0,      # Mass coupling (GeV)
    V0=1.0,         # Primary well depth (GeV)
    V1=0.1,         # Secondary modulation (GeV)
)

# Create computational grid
grid = SpectralGrid(N=256)

# Create potential
potential = ThreeWellPotential(params.V0, params.V1)

# Solve for eigenvalues
solver = LinearEigensolver(grid, potential)
energies, wavefunctions = solver.solve(k=1, n_states=5)

print(f"Ground state energy: {energies[0]:.4f} GeV")
```

## Project Structure

```
sft-solver/
├── src/sft_solver/
│   ├── core/           # Constants, parameters, grid
│   │   ├── constants.py
│   │   ├── parameters.py
│   │   └── grid.py
│   ├── potentials/     # Potential energy functions
│   │   ├── three_well.py
│   │   └── effective.py
│   ├── eigensolver/    # Eigenvalue solvers
│   │   ├── linear.py
│   │   ├── nonlinear.py
│   │   └── spectral.py
│   ├── forces/         # EM force calculations
│   │   └── electromagnetic.py
│   ├── analysis/       # Mass spectrum analysis
│   │   └── mass_spectrum.py
│   └── validation/     # Testbench comparison
│       └── testbench.py
├── tests/              # Unit tests
├── notebooks/          # Jupyter demos
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Concepts

### The Beautiful Equation

The SFT framework is constrained by:

```
β L₀ c = ℏ
```

where β is the mass coupling scale, L₀ is the subspace radius, c is the speed of light, and ℏ is the reduced Planck constant.

### Winding Number and Charge

Electric charge emerges from the winding number k of the subspace mode:

```
Q = ±e/k
```

- k=1: Leptons (Q = ±e)
- k=3: Down-type quarks (Q = ∓e/3)
- k=5: Up-type quarks (Q = ±2e/3)

### Mass Formula

Particle mass is determined by the subspace amplitude:

```
m = β A²_χ
```

where A²_χ = ∫₀^(2π) |χ(σ)|² dσ

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sft_solver

# Run specific test file
pytest tests/test_grid.py
```

## Notebooks

Interactive Jupyter notebooks demonstrating the solver:

- `01_single_particle.ipynb`: Visualization of single-particle solutions

## Success Criteria

The solver should satisfy:

1. Convergence for k=1 modes
2. Mass ratio m_μ/m_e within 10% of 206.77 (with parameter tuning)
3. Periodic boundary conditions satisfied: χ(σ + 2π) = χ(σ)
4. EM force shows repulsion for like charges, attraction for opposite
5. All unit tests pass

## References

- Mathematical Formulation Parts A, B, C
- Origin of Mass Research Note
- Origin of Electromagnetism Research Note
- SFT Testbench Documentation

## License

MIT License - see LICENSE file for details.

