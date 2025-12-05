"""
Validation module for SFT Solver.

Contains testbench comparison against experimental data.
"""

from sft_solver.validation.testbench import (
    TestbenchValidator,
    ValidationResult,
    TESTBENCH_VALUES,
)

__all__ = [
    "TestbenchValidator",
    "ValidationResult",
    "TESTBENCH_VALUES",
]

