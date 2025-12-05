"""
Testbench validation against experimental values.

Compares SFT solver results with the model-independent
testbench parameters from SFT_Testbench.md.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from sft_solver.core.constants import (
    HBAR, C, E_CHARGE, ALPHA_EM,
    ELECTRON_MASS_GEV, MUON_MASS_GEV, TAU_MASS_GEV,
    MUON_ELECTRON_RATIO, TAU_MUON_RATIO, TAU_ELECTRON_RATIO,
    W_MASS_GEV, Z_MASS_GEV, HIGGS_MASS_GEV,
    PROTON_MASS_GEV, NEUTRON_MASS_GEV,
    PION_CHARGED_MASS_GEV, PION_NEUTRAL_MASS_GEV,
)


# ============================================================================
# Testbench Reference Values
# ============================================================================

TESTBENCH_VALUES = {
    # Fundamental Constants
    'c': {'value': C, 'unit': 'm/s', 'status': 'exact'},
    'hbar': {'value': HBAR, 'unit': 'J·s', 'status': 'exact'},
    'e': {'value': E_CHARGE, 'unit': 'C', 'status': 'exact'},
    'alpha_em': {'value': ALPHA_EM, 'unit': '', 'status': 'measured', 'uncertainty': 1.5e-10},
    
    # Lepton Masses (GeV)
    'electron_mass': {'value': ELECTRON_MASS_GEV, 'unit': 'GeV', 'uncertainty': 1.5e-13},
    'muon_mass': {'value': MUON_MASS_GEV, 'unit': 'GeV', 'uncertainty': 2.3e-8},
    'tau_mass': {'value': TAU_MASS_GEV, 'unit': 'GeV', 'uncertainty': 1.2e-4},
    
    # Lepton Mass Ratios (Critical Tests)
    'mu_e_ratio': {'value': MUON_ELECTRON_RATIO, 'unit': '', 'uncertainty': 4.6e-6},
    'tau_mu_ratio': {'value': TAU_MUON_RATIO, 'unit': '', 'uncertainty': 1e-3},
    'tau_e_ratio': {'value': TAU_ELECTRON_RATIO, 'unit': '', 'uncertainty': 0.31},
    
    # Gauge Boson Masses (GeV)
    'W_mass': {'value': W_MASS_GEV, 'unit': 'GeV', 'uncertainty': 0.023},
    'Z_mass': {'value': Z_MASS_GEV, 'unit': 'GeV', 'uncertainty': 0.002},
    'Higgs_mass': {'value': HIGGS_MASS_GEV, 'unit': 'GeV', 'uncertainty': 0.24},
    
    # Hadron Masses (GeV)
    'proton_mass': {'value': PROTON_MASS_GEV, 'unit': 'GeV', 'uncertainty': 2.9e-10},
    'neutron_mass': {'value': NEUTRON_MASS_GEV, 'unit': 'GeV', 'uncertainty': 5.8e-10},
    'pion_charged_mass': {'value': PION_CHARGED_MASS_GEV, 'unit': 'GeV', 'uncertainty': 1.8e-7},
    'pion_neutral_mass': {'value': PION_NEUTRAL_MASS_GEV, 'unit': 'GeV', 'uncertainty': 5e-6},
    
    # Mass Differences (GeV)
    'neutron_proton_diff': {'value': NEUTRON_MASS_GEV - PROTON_MASS_GEV, 'unit': 'GeV'},
    'pion_mass_diff': {'value': PION_CHARGED_MASS_GEV - PION_NEUTRAL_MASS_GEV, 'unit': 'GeV'},
    
    # Electroweak Ratios
    'W_Z_ratio': {'value': W_MASS_GEV / Z_MASS_GEV, 'unit': ''},
    'H_Z_ratio': {'value': HIGGS_MASS_GEV / Z_MASS_GEV, 'unit': ''},
}


class TestTier(Enum):
    """Classification of test importance."""
    ESSENTIAL = 1  # Must match to validate theory
    STRONG = 2     # Should match to build confidence
    NOVEL = 3      # Predictions that distinguish from SM


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    name: str
    tier: TestTier
    computed: float
    expected: float
    uncertainty: float
    error_percent: float
    passed: bool
    message: str


class TestbenchValidator:
    """
    Validator for comparing SFT results with testbench values.
    
    Provides comprehensive validation against experimental data
    organized by tier of importance.
    
    Attributes:
        results: List of validation results.
        tolerance: Default tolerance for pass/fail.
    """
    
    def __init__(self, tolerance: float = 0.10):
        """
        Initialize the validator.
        
        Args:
            tolerance: Default tolerance (fraction) for tests.
                      0.10 = 10% tolerance.
        """
        self.tolerance = tolerance
        self.results: List[ValidationResult] = []
    
    def validate_lepton_masses(
        self,
        masses: Dict[str, float],
        beta: Optional[float] = None
    ) -> List[ValidationResult]:
        """
        Validate lepton masses against experimental values.
        
        Args:
            masses: Dictionary with 'electron', 'muon', 'tau' masses in GeV.
            beta: The β parameter used (for reference).
            
        Returns:
            List of ValidationResult objects.
        """
        results = []
        
        expected_masses = {
            'electron': ELECTRON_MASS_GEV,
            'muon': MUON_MASS_GEV,
            'tau': TAU_MASS_GEV,
        }
        
        for name, expected in expected_masses.items():
            if name in masses:
                computed = masses[name]
                error = abs(computed - expected) / expected
                error_percent = error * 100
                passed = error < self.tolerance
                
                result = ValidationResult(
                    name=f"{name}_mass",
                    tier=TestTier.ESSENTIAL,
                    computed=computed,
                    expected=expected,
                    uncertainty=expected * 0.001,  # 0.1% assumed
                    error_percent=error_percent,
                    passed=passed,
                    message=f"{name} mass: {computed:.6e} GeV (expected {expected:.6e}, error {error_percent:.1f}%)"
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def validate_mass_ratios(
        self,
        ratios: Dict[str, float]
    ) -> List[ValidationResult]:
        """
        Validate mass ratios (critical test for SFT).
        
        Args:
            ratios: Dictionary with ratio values.
                   Keys: 'muon/electron', 'tau/muon', 'tau/electron'
            
        Returns:
            List of ValidationResult objects.
        """
        results = []
        
        expected_ratios = {
            'muon/electron': MUON_ELECTRON_RATIO,
            'tau/muon': TAU_MUON_RATIO,
            'tau/electron': TAU_ELECTRON_RATIO,
        }
        
        for name, expected in expected_ratios.items():
            if name in ratios:
                computed = ratios[name]
                error = abs(computed - expected) / expected
                error_percent = error * 100
                passed = error < self.tolerance
                
                result = ValidationResult(
                    name=f"ratio_{name.replace('/', '_')}",
                    tier=TestTier.ESSENTIAL,
                    computed=computed,
                    expected=expected,
                    uncertainty=expected * 0.001,
                    error_percent=error_percent,
                    passed=passed,
                    message=f"m_{name}: {computed:.2f} (expected {expected:.2f}, error {error_percent:.1f}%)"
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def validate_beautiful_equation(
        self,
        beta: float,
        L0: float
    ) -> ValidationResult:
        """
        Validate the Beautiful Equation β L₀ c = ℏ.
        
        Args:
            beta: Mass coupling in GeV.
            L0: Subspace radius in meters.
            
        Returns:
            ValidationResult object.
        """
        # Convert β to J: β (GeV) × 1.602e-10 (J/GeV)
        GeV_to_J = 1.602176634e-10
        beta_J = beta * GeV_to_J
        
        # Compute β L₀ c / ℏ (should equal 1)
        # Note: β has units of energy, L₀ of length
        # β L₀ c has units of energy × length × velocity = J·m·(m/s) = J·m²/s
        # This doesn't match ℏ = J·s
        # The correct form is: β × L₀ / (ℏ c) = 1 (dimensionless)
        # Or equivalently: β = ℏ c / L₀
        
        # Let's use: ratio = β L₀ / (ℏ c)
        ratio = (beta_J * L0) / (HBAR * C)
        
        error = abs(ratio - 1.0)
        passed = error < 1e-6  # Should be exactly 1
        
        result = ValidationResult(
            name="beautiful_equation",
            tier=TestTier.ESSENTIAL,
            computed=ratio,
            expected=1.0,
            uncertainty=1e-10,
            error_percent=error * 100,
            passed=passed,
            message=f"β L₀ / (ℏc) = {ratio:.10f} (should be 1.0)"
        )
        
        self.results.append(result)
        return result
    
    def validate_charge_quantization(
        self,
        winding_numbers: Dict[str, int]
    ) -> List[ValidationResult]:
        """
        Validate that winding numbers give correct charges.
        
        Q = ±e/k
        k=1: leptons (Q=±e)
        k=3: down quarks (Q=∓e/3)
        k=5: up quarks (Q=±2e/3)
        
        Args:
            winding_numbers: Dict mapping particle names to k values.
            
        Returns:
            List of ValidationResult objects.
        """
        results = []
        
        expected_windings = {
            'electron': 1,
            'muon': 1,
            'tau': 1,
            'up_quark': 5,
            'down_quark': 3,
        }
        
        for name, expected in expected_windings.items():
            if name in winding_numbers:
                computed = winding_numbers[name]
                passed = computed == expected
                
                result = ValidationResult(
                    name=f"winding_{name}",
                    tier=TestTier.ESSENTIAL,
                    computed=float(computed),
                    expected=float(expected),
                    uncertainty=0.0,
                    error_percent=0.0 if passed else 100.0,
                    passed=passed,
                    message=f"{name} k={computed} (expected {expected})"
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def summary(self) -> str:
        """
        Generate a summary of all validation results.
        
        Returns:
            Formatted summary string.
        """
        lines = ["=" * 70]
        lines.append("SFT Testbench Validation Summary")
        lines.append("=" * 70)
        
        # Group by tier
        by_tier: Dict[TestTier, List[ValidationResult]] = {
            tier: [] for tier in TestTier
        }
        for result in self.results:
            by_tier[result.tier].append(result)
        
        for tier in TestTier:
            tier_results = by_tier[tier]
            if not tier_results:
                continue
            
            passed = sum(1 for r in tier_results if r.passed)
            total = len(tier_results)
            
            lines.append(f"\n{tier.name} Tests ({passed}/{total} passed):")
            lines.append("-" * 50)
            
            for result in tier_results:
                status = "✓" if result.passed else "✗"
                lines.append(f"  {status} {result.message}")
        
        # Overall summary
        total_passed = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Overall: {total_passed}/{total_tests} tests passed")
        
        essential = by_tier[TestTier.ESSENTIAL]
        if essential:
            essential_passed = sum(1 for r in essential if r.passed)
            if essential_passed == len(essential):
                lines.append("✓ All essential tests passed - theory validated!")
            else:
                lines.append(f"✗ {len(essential) - essential_passed} essential tests failed")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def passed_essential(self) -> bool:
        """
        Check if all essential tests passed.
        
        Returns:
            True if all essential tests passed.
        """
        essential = [r for r in self.results if r.tier == TestTier.ESSENTIAL]
        return all(r.passed for r in essential)
    
    def to_dict(self) -> Dict:
        """
        Export validation results as dictionary.
        
        Returns:
            Dictionary with all results.
        """
        return {
            'tolerance': self.tolerance,
            'results': [
                {
                    'name': r.name,
                    'tier': r.tier.name,
                    'computed': r.computed,
                    'expected': r.expected,
                    'error_percent': r.error_percent,
                    'passed': r.passed,
                    'message': r.message,
                }
                for r in self.results
            ],
            'summary': {
                'total': len(self.results),
                'passed': sum(1 for r in self.results if r.passed),
                'essential_passed': self.passed_essential(),
            }
        }

