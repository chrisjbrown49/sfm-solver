# SFM Solver Test Suite

Unit and integration tests for the two-stage solver architecture.

## Test Files

### New Architecture Tests (Active)

- **`test_shape_solver.py`** - Unit tests for `DimensionlessShapeSolver` (Stage 1)
  - Shape normalization
  - EM effects on eigenstate structure
  - Dimensionless couplings
  - SCF convergence

- **`test_spatial_coupling.py`** - Unit tests for `SpatialCouplingBuilder`
  - 4D structure normalization
  - Induced spatial components
  - Coupling matrix computation

- **`test_energy_minimizer.py`** - Unit tests for `UniversalEnergyMinimizer` (Stage 2)
  - Energy component calculations
  - Wavefunction scaling
  - Energy minimization
  - Scale parameter optimization

- **`test_unified_solver.py`** - Integration tests for `UnifiedSFMSolver`
  - End-to-end particle solving
  - Lepton generation hierarchy
  - Baryon mass predictions
  - Neutron-proton splitting

### Legacy Tests

Legacy tests for the old solver architecture have been moved to `tests/legacy/`.

## Running Tests

### Run All Tests
```powershell
pytest tests/ -v
```

### Run Only Unit Tests (Fast)
```powershell
pytest tests/ -m unit -v
```

### Run Only Integration Tests
```powershell
pytest tests/ -m integration -v
```

### Skip Slow Tests
```powershell
pytest tests/ -m "not slow" -v
```

### Run Specific Test File
```powershell
pytest tests/test_unified_solver.py -v
```

### Run Specific Test
```powershell
pytest tests/test_unified_solver.py::TestUnifiedSolver::test_energy_breakdown -v
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests (slower)
- `@pytest.mark.slow` - Slow tests (can be skipped for quick validation)

## Test Configuration

- **`conftest.py`** - Pytest configuration and fixtures
  - `test_constants` fixture - Standard test constants
  - `small_test_constants` fixture - Smaller grid for faster tests

- **`__init__.py`** - Test package initialization and documentation

## Expected Test Runtime

- **Unit tests:** ~1-5 seconds each
- **Integration tests (fast):** ~10-30 seconds each
- **Integration tests (slow):** ~30-120 seconds each

## Coverage

To run tests with coverage:
```powershell
pytest tests/ --cov=sfm_solver.core --cov-report=html
```

View coverage report at `htmlcov/index.html`

