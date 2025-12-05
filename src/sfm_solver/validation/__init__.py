"""
Validation module for SFM Solver.

Contains testbench comparison against experimental data.
"""

from sfm_solver.validation.testbench import (
    TestbenchValidator,
    ValidationResult,
    TESTBENCH_VALUES,
)

__all__ = [
    "TestbenchValidator",
    "ValidationResult",
    "TESTBENCH_VALUES",
]
