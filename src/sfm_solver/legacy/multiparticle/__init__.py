"""
Multiparticle module for SFM Solver.

Provides solvers for multi-quark bound states:
- Baryons: Three-quark systems (proton, neutron, etc.)
- Mesons: Quark-antiquark systems (pion, kaon, etc.)

Key physics:
- Quarks have winding number k=3 (charge e/3)
- Color is NOT a fundamental quantum number - it's the emergent
  three-phase structure {0, 2π/3, 4π/3}
- Color phases must EMERGE from energy minimization, not be imposed
- Confinement is geometric: single quarks cannot form stable patterns on S¹
"""

from sfm_solver.multiparticle.color_verification import (
    ColorVerification,
    extract_phases,
    verify_color_neutrality,
    verify_phase_emergence,
)

# Composite baryon solver (single wavefunction - correct physics)
from sfm_solver.multiparticle.composite_baryon import (
    CompositeBaryonSolver,
    CompositeBaryonState,
    PROTON_QUARKS,
    NEUTRON_QUARKS,
)

# Composite meson solver (single wavefunction - correct physics)
from sfm_solver.multiparticle.composite_meson import (
    CompositeMesonSolver,
    CompositeMesonState,
    MESON_CONFIGS,
)

# Aliases for convenience
BaryonSolver = CompositeBaryonSolver
BaryonState = CompositeBaryonState
MesonSolver = CompositeMesonSolver
MesonState = CompositeMesonState


__all__ = [
    # Color verification
    'ColorVerification',
    'extract_phases',
    'verify_color_neutrality',
    'verify_phase_emergence',
    
    # Baryon solver
    'BaryonSolver',
    'BaryonState',
    'CompositeBaryonSolver',
    'CompositeBaryonState',
    'PROTON_QUARKS',
    'NEUTRON_QUARKS',
    
    # Meson solver
    'MesonSolver',
    'MesonState',
    'CompositeMesonSolver',
    'CompositeMesonState',
    'MESON_CONFIGS',
]
