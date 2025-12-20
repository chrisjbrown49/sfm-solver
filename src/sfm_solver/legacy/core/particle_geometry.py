"""
Particle Geometry Helper Functions for SFM.

Reference: Implementation Note - The Beautiful Balance.md

These functions compute particle-type-dependent parameters that emerge from
the geometric configuration of quarks/leptons in the subspace dimension.

Key insight: The same three-well potential V(σ) = V₀[1 - cos(3σ)] looks different
to different particles based on how their wavefunctions are distributed.

SPATIAL EXTENT (Δx):
===================
Δx is computed from the COMPTON WAVELENGTH (first-principles QM):
    Δx = λ_C = ℏ/(mc) = 1/m = 1/(βA²)  [natural units]

This is NOT from virial balance with curvature (which gives wrong results).
The Compton wavelength is the fundamental quantum mechanical scale for
localizing a particle of mass m.
"""

import numpy as np
from typing import Tuple, List


def compute_delta_x(A_chi: float, beta: float, n_spatial: int = 1) -> float:
    """
    Compute spatial extent Δx from COMPTON WAVELENGTH (first-principles QM).
    
    Reference: Implementation Note - The Beautiful Balance.md, Section 5
    
    FORMULA:
        Δx = λ_C = ℏ/(mc) = 1/m = 1/(βA²)  [natural units, ℏ=c=1]
    
    This is the fundamental quantum mechanical scale for a particle of mass m.
    The Compton wavelength represents the minimum scale at which the particle
    can be localized without creating particle-antiparticle pairs.
    
    NOTE: This is NOT from virial balance with curvature!
    The virial formula Δx ~ 1/A⁶ gives unphysical results.
    
    Parameters:
    -----------
    A_chi : float
        Subspace amplitude (from energy minimization)
    beta : float
        Mass coupling constant (GeV)
    n_spatial : int
        Radial quantum number (1, 2, 3, ...) for excited states
    
    Returns:
    --------
    float : Spatial extent Δx in GeV⁻¹ (natural units)
    
    Physical Interpretation:
    ------------------------
    For pion (m ~ 140 MeV):  Δx ~ 7 GeV⁻¹ ~ 1.4 fm
    For proton (m ~ 938 MeV): Δx ~ 1 GeV⁻¹ ~ 0.2 fm
    
    These are reasonable hadronic scales!
    """
    # Mass from universal formula
    m = beta * A_chi**2
    
    # Prevent division by zero
    if m < 1e-10:
        return 1e10  # Very large Δx for nearly massless
    
    # Compton wavelength in natural units
    dx_base = 1.0 / m
    
    # Radial excitation scaling (WKB approximation)
    # Higher excitations have larger spatial extent
    dx = dx_base * n_spatial**(2.0/3.0)
    
    return dx


def compute_delta_x_from_mass(mass_gev: float, n_spatial: int = 1) -> float:
    """
    Compute spatial extent directly from mass (convenience function).
    
    Δx = 1/m  [natural units]
    
    Parameters:
    -----------
    mass_gev : float
        Particle mass in GeV
    n_spatial : int
        Radial quantum number
    
    Returns:
    --------
    float : Spatial extent Δx in GeV⁻¹
    """
    if mass_gev < 1e-10:
        return 1e10
    
    dx_base = 1.0 / mass_gev
    return dx_base * n_spatial**(2.0/3.0)


def compute_k_values(particle_type: str, k_list: List[int]) -> Tuple[float, float]:
    """
    Compute k_coupling and k_confinement for a particle.
    
    Reference: Implementation Note - The Beautiful Balance.md, Section 3.5
    
    Parameters:
    -----------
    particle_type : str
        'lepton', 'meson', or 'baryon'
    k_list : list of int
        Winding numbers [k1, k2, ...] for constituent particles
        - Leptons: [k] (single winding)
        - Mesons: [k_q, k_qbar] (quark, antiquark)
        - Baryons: [k1, k2, k3] (three quarks)
    
    Returns:
    --------
    tuple : (k_coupling, k_confinement)
        - k_coupling: For E_coupling = -α × k_coupling × A (sum of |k_i|)
        - k_confinement: For Δx = L₀ × k_confinement
    
    Physical Interpretation:
    ------------------------
    k_coupling is ADDITIVE:
        Each quark independently couples its subspace winding to spacetime 
        gradients via the mixed derivative operator.
    
    k_confinement is GEOMETRIC:
        The bound state size is determined by the strongest pairwise bond
        and the geometric configuration (triangular for baryons).
    """
    if particle_type == 'lepton':
        # Single particle
        k = abs(k_list[0])
        return float(k), float(k)
    
    elif particle_type == 'meson':
        # Quark-antiquark pair
        k_q = abs(k_list[0])
        k_qbar = abs(k_list[1])
        
        # Coupling: sum of individual windings (additive)
        k_coupling = k_q + k_qbar
        
        # Confinement: arithmetic mean (represents average winding)
        # This emerges from interference of two wavefunctions
        # Alternative: use emergent k_eff from wavefunction gradient
        k_confinement = (k_q + k_qbar) / 2.0
        
        return float(k_coupling), k_confinement
    
    elif particle_type == 'baryon':
        # Three quarks
        k1 = abs(k_list[0])
        k2 = abs(k_list[1])
        k3 = abs(k_list[2])
        
        # Coupling: sum of individual windings (additive)
        k_coupling = k1 + k2 + k3
        
        # Confinement: strongest pairwise bond × √3 for triangular geometry
        # The bound state size is determined by the STRONGEST pairwise bond
        k_pairs = [k1 * k2, k1 * k3, k2 * k3]
        k_max_pair = max(k_pairs)
        
        # Geometric enhancement from triangular structure (equilateral triangle)
        # The three-quark color phases {0, 2π/3, 4π/3} create additional confinement
        SQRT3 = np.sqrt(3)
        k_confinement = SQRT3 * k_max_pair
        
        return float(k_coupling), k_confinement
    
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")


def get_effective_potential(V0: float, particle_type: str) -> float:
    """
    Compute effective potential depth based on wavefunction distribution.
    
    Reference: Implementation Note - The Beautiful Balance.md, Section 5
    
    The same three-well potential V(σ) = V₀[1 - cos(3σ)] looks different
    to different particles based on how their wavefunctions sample it.
    
    Parameters:
    -----------
    V0 : float
        Base three-well potential depth (GeV)
    particle_type : str
        'lepton', 'meson', or 'baryon'
    
    Returns:
    --------
    float : Effective potential depth V_eff (GeV)
    
    Physical Interpretation:
    ------------------------
    Leptons (single peak in one well):
        Experience full potential depth at well bottom
        V_eff = V₀
    
    Mesons (two peaks in adjacent wells):
        Sample the barrier region between wells
        Slightly higher effective potential
        V_eff = 1.15 × V₀
    
    Baryons (three peaks symmetrically distributed):
        Average over full circle due to three-fold symmetry
        Reduced effective potential (smoothed out)
        V_eff = 0.85 × V₀
    """
    if particle_type == 'lepton':
        # Single peak in one well - experiences full potential depth
        return V0
    
    elif particle_type == 'meson':
        # Two peaks in adjacent wells - samples barrier region
        # Slightly higher effective potential
        return V0 * 1.15
    
    elif particle_type == 'baryon':
        # Three peaks symmetrically distributed
        # Averages over full circle - reduced effective potential
        return V0 * 0.85
    
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")


def get_geometric_factor(particle_type: str) -> float:
    """
    Get geometric enhancement factor from color phase structure.
    
    Reference: Implementation Note - The Beautiful Balance.md, Section 6
    
    For baryons, the three-quark color phase structure creates geometric
    enhancement in the coupling energy.
    
    Color phases for color-neutral baryon: {0, 2π/3, 4π/3}
    These form an equilateral triangle in the complex plane.
    
    Parameters:
    -----------
    particle_type : str
        'lepton', 'meson', or 'baryon'
    
    Returns:
    --------
    float : Geometric enhancement factor g_geom
    
    Physical Interpretation:
    ------------------------
    The cross terms between quarks in the mixed derivative contribute
    factors involving cos(2π/3) = -1/2.
    
    Geometric analysis shows effective coupling enhancement of √3.
    """
    if particle_type == 'lepton':
        return 1.0  # No color structure
    
    elif particle_type == 'meson':
        return 1.0  # Linear configuration, no enhancement
    
    elif particle_type == 'baryon':
        return np.sqrt(3)  # Triangular color phase geometry
    
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")


def get_hadronic_scale(particle_type: str) -> float:
    """
    Get characteristic confinement scale for hadrons.
    
    Reference: Implementation Note - The Beautiful Balance.md, Section 3.3
    
    Parameters:
    -----------
    particle_type : str
        'lepton', 'meson', or 'baryon'
    
    Returns:
    --------
    float : Confinement radius in GeV^-1 (natural units)
            Returns None for leptons (no hadronic confinement)
    
    Physical Values:
    ----------------
    Baryons: ~0.87 fm (triangular confinement)
    Mesons:  ~1.3 fm (linear q-q̄ configuration)
    
    Note: 1 fm ≈ 5 GeV⁻¹ in natural units
    """
    R_CONFINEMENT = 5.0  # 1 fm ≈ 5 GeV⁻¹ in natural units
    
    if particle_type == 'lepton':
        return None  # No hadronic confinement
    
    elif particle_type == 'meson':
        # Two quarks in linear configuration
        # Slightly larger due to q-q̄ oscillation
        return R_CONFINEMENT * 1.3  # ~1.3 fm
    
    elif particle_type == 'baryon':
        # Three quarks in triangular configuration
        # Geometric factor from equilateral triangle
        return R_CONFINEMENT / np.sqrt(3)  # ~0.87 fm
    
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")


# Convenience function for proton
def get_proton_k_values() -> Tuple[float, float]:
    """
    Get k_coupling and k_confinement for proton (uud).
    
    Returns:
    --------
    tuple: (k_coupling=13, k_confinement≈43.3)
    """
    return compute_k_values('baryon', [5, 5, 3])  # uud: k_u=5, k_u=5, k_d=3


def get_neutron_k_values() -> Tuple[float, float]:
    """
    Get k_coupling and k_confinement for neutron (udd).
    
    Returns:
    --------
    tuple: (k_coupling=11, k_confinement≈26)
    """
    return compute_k_values('baryon', [5, 3, 3])  # udd: k_u=5, k_d=3, k_d=3


def get_pion_k_values() -> Tuple[float, float]:
    """
    Get k_coupling and k_confinement for pion (ud̄).
    
    Returns:
    --------
    tuple: (k_coupling=8, k_confinement=4)
    """
    return compute_k_values('meson', [5, 3])  # ud̄: k_u=5, k_d̄=3

