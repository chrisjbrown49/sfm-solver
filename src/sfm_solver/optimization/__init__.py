"""
SFM Optimization Module.

This module provides global optimization to discover SFM framework parameters
from first principles, as prescribed by the SFM research notes.
"""

from sfm_solver.optimization.parameter_optimizer import (
    SFMParameterOptimizer,
    ParticleSpec,
    OptimizationResult,
)

__all__ = ['SFMParameterOptimizer', 'ParticleSpec', 'OptimizationResult']

