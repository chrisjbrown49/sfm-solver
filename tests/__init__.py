"""
SFM Solver Test Suite.

Unit and integration tests for the two-stage solver architecture:
- Stage 1: Dimensionless shape solving
- Stage 2: Energy minimization over scale parameters

Test Files:
- test_shape_solver.py: Tests for DimensionlessShapeSolver
- test_spatial_coupling.py: Tests for SpatialCouplingBuilder
- test_energy_minimizer.py: Tests for UniversalEnergyMinimizer
- test_unified_solver.py: Integration tests for UnifiedSFMSolver

Usage:
    # Run all tests
    pytest tests/ -v
    
    # Run specific test file
    pytest tests/test_unified_solver.py -v
    
    # Run only unit tests (fast)
    pytest tests/ -m unit -v
    
    # Run only integration tests
    pytest tests/ -m integration -v
    
    # Skip slow tests
    pytest tests/ -m "not slow" -v
"""
