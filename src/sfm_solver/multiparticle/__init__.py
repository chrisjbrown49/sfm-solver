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

# Composite baryon solver (CORRECT physics - single wavefunction)
from sfm_solver.multiparticle.composite_baryon import (
    CompositeBaryonSolver,
    CompositeBaryonState,
)

# Legacy baryon solver (deprecated - three separate wavefunctions)
from sfm_solver.multiparticle.baryon import (
    BaryonSolver as LegacyBaryonSolver,
    BaryonState as LegacyBaryonState,
    solve_baryon_system,
)

# Use composite solver as the default BaryonSolver
BaryonSolver = CompositeBaryonSolver
BaryonState = CompositeBaryonState

# Composite meson solver (CORRECT physics - single wavefunction like baryons)
from sfm_solver.multiparticle.composite_meson import (
    CompositeMesonSolver,
    CompositeMesonState,
)

# Legacy meson solver (eigenvalue-based)
from sfm_solver.multiparticle.meson import (
    MesonSolver as LegacyMesonSolver,
    MesonState as LegacyMesonState,
    solve_meson_system,
)

# Use composite solver as the default MesonSolver
MesonSolver = CompositeMesonSolver
MesonState = CompositeMesonState


__all__ = [
    # Color verification
    'ColorVerification',
    'extract_phases',
    'verify_color_neutrality',
    'verify_phase_emergence',
    
    # Baryon solver (composite - correct physics)
    'BaryonSolver',  # Alias for CompositeBaryonSolver
    'BaryonState',   # Alias for CompositeBaryonState
    'CompositeBaryonSolver',
    'CompositeBaryonState',
    
    # Meson solver (composite - correct physics)
    'MesonSolver',   # Alias for CompositeMesonSolver
    'MesonState',    # Alias for CompositeMesonState
    'CompositeMesonSolver',
    'CompositeMesonState',
    
    # Legacy (deprecated)
    'LegacyBaryonSolver',
    'LegacyBaryonState',
    'solve_baryon_system',
    'LegacyMesonSolver',
    'LegacyMesonState',
    'solve_meson_system',
]

