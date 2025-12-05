"""
Spatial discretization module for SFM Solver.

Provides grid and operator implementations for the spatial (observable)
dimensions of the 5D spacetime. This module enables solving the coupled
subspace-spacetime eigenvalue problem.

Key components:
- RadialGrid: Discretization for spherically symmetric solutions
- RadialOperators: Kinetic and gradient operators in radial coordinates
"""

from sfm_solver.spatial.radial import RadialGrid, RadialOperators

__all__ = ['RadialGrid', 'RadialOperators']

