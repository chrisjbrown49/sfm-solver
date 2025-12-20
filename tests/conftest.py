"""
Pytest configuration for SFM Solver test suite.

Configuration for testing the new two-stage solver architecture.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def pytest_configure(config):
    """Called after command line options have been parsed."""
    # Add markers for test organization
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture
def test_constants():
    """Provide test constants for unit tests."""
    return {
        'beta': 0.0005,
        'g_internal': 0.003,
        'g1': 40.0,
        'g2': 1.0,
        'V0': 5.0,
        'V1': 0.0,
        'alpha': 10.5,
        'n_max': 3,
        'l_max': 2,
        'N_sigma': 64
    }


@pytest.fixture
def small_test_constants():
    """Provide smaller constants for faster unit tests."""
    return {
        'beta': 0.0005,
        'g_internal': 0.003,
        'g1': 40.0,
        'g2': 1.0,
        'V0': 5.0,
        'V1': 0.0,
        'alpha': 10.5,
        'n_max': 2,
        'l_max': 1,
        'N_sigma': 32  # Smaller grid for faster tests
    }
