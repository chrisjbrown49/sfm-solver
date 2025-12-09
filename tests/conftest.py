"""
Pytest configuration for SFM Solver test suite.

Provides automatic results file generation after each test run.
"""

import pytest
import time
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sfm_solver.reporting.results_reporter import (
    ResultsReporter, 
    TestResult, 
    PredictionResult,
    get_reporter,
    reset_reporter,
)
from sfm_solver.reporting.results_viewer import generate_html_results


# Store timing information
_test_start_times = {}


def pytest_configure(config):
    """Called after command line options have been parsed."""
    # Reset and start the reporter
    reset_reporter()
    reporter = get_reporter()
    reporter.start_run()
    
    # Add markers
    config.addinivalue_line(
        "markers", "tier1: marks tests as Tier 1 (single-particle eigenstates)"
    )
    config.addinivalue_line(
        "markers", "tier1b: marks tests as Tier 1b (electromagnetic forces)"
    )
    config.addinivalue_line(
        "markers", "tier2: marks tests as Tier 2 (multi-quark bound states)"
    )
    config.addinivalue_line(
        "markers", "tier3: marks tests as Tier 3 (weak decay tunneling)"
    )
    config.addinivalue_line(
        "markers", "lepton: marks tests for physics-based lepton solver"
    )
    config.addinivalue_line(
        "markers", "mass_hierarchy: marks tests for mass hierarchy validation"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    # Tier 2 specific markers
    config.addinivalue_line(
        "markers", "baryon: marks tests for baryon (three-quark) solver"
    )
    config.addinivalue_line(
        "markers", "meson: marks tests for meson (quark-antiquark) solver"
    )
    config.addinivalue_line(
        "markers", "color: marks tests for color phase verification"
    )
    config.addinivalue_line(
        "markers", "confinement: marks tests for quark confinement"
    )
    config.addinivalue_line(
        "markers", "binding: marks tests for binding energy calculations"
    )
    # Tier 2b specific markers (quarkonia radial excitations)
    config.addinivalue_line(
        "markers", "tier2b: marks tests as Tier 2b (quarkonia radial excitations)"
    )
    config.addinivalue_line(
        "markers", "charmonium: marks tests for charmonium (cc-bar) family"
    )
    config.addinivalue_line(
        "markers", "bottomonium: marks tests for bottomonium (bb-bar) family"
    )
    config.addinivalue_line(
        "markers", "radial_excitation: marks tests for radial excitation physics"
    )


def pytest_runtest_setup(item):
    """Called before each test."""
    _test_start_times[item.nodeid] = time.time()


def pytest_runtest_makereport(item, call):
    """Called to create a test report for each test phase."""
    if call.when == 'call':
        # Get timing
        start_time = _test_start_times.get(item.nodeid, time.time())
        duration = time.time() - start_time
        
        # Determine category from test path
        category = _get_test_category(item)
        
        # Create test result
        passed = call.excinfo is None
        message = ""
        details = {}
        
        if not passed and call.excinfo:
            message = str(call.excinfo.value)[:200]
            details['exception_type'] = call.excinfo.typename
        
        result = TestResult(
            name=item.name,
            passed=passed,
            duration=duration,
            category=category,
            message=message,
            details=details,
        )
        
        # Add to reporter
        reporter = get_reporter()
        reporter.add_test_result(result)


def _get_test_category(item) -> str:
    """Determine test category from test item."""
    # Check markers first
    for marker in item.iter_markers():
        if marker.name == 'tier1':
            return 'Tier 1: Eigenstates'
        elif marker.name == 'tier1b':
            return 'Tier 1b: EM Forces'
        elif marker.name == 'tier2':
            return 'Tier 2: Multi-quark'
        elif marker.name == 'tier2b':
            return 'Tier 2b: Quarkonia'
        elif marker.name == 'tier3':
            return 'Tier 3: Weak Decay'
        elif marker.name == 'lepton':
            return 'Tier 1: Lepton Solver'
        elif marker.name == 'mass_hierarchy':
            return 'Physics: Mass Hierarchy'
    
    # Infer from test file/class name
    nodeid = item.nodeid.lower()
    
    # Check file name first for more accurate categorization
    if 'test_tier1_lepton' in nodeid or 'lepton_solver' in nodeid:
        if 'mass_ratio' in nodeid or 'hierarchy' in nodeid:
            return 'Tier 1: Lepton Mass Hierarchy'
        elif 'beta' in nodeid or 'beautiful' in nodeid:
            return 'Tier 1: Global Constants'
        else:
            return 'Tier 1: Lepton Solver'
    elif 'test_tier1b' in nodeid or 'electromagnetic' in nodeid:
        # Further categorize Tier 1b tests
        if 'charge' in nodeid or 'quantization' in nodeid:
            return 'Tier 1b: Charge Quantization'
        elif 'circulation' in nodeid:
            return 'Tier 1b: Circulation Integral'
        elif 'coulomb' in nodeid or 'scaling' in nodeid:
            return 'Tier 1b: Coulomb Scaling'
        elif 'fine_structure' in nodeid or 'alpha' in nodeid:
            return 'Tier 1b: Fine Structure'
        elif 'multi' in nodeid or 'particle' in nodeid:
            return 'Tier 1b: Multi-Particle'
        elif 'asymmetry' in nodeid or 'envelope' in nodeid:
            return 'Tier 1b: Envelope Asymmetry'
        elif 'analysis' in nodeid:
            return 'Tier 1b: Two-Particle Analysis'
        elif 'consistency' in nodeid or 'physical' in nodeid:
            return 'Tier 1b: Physical Consistency'
        else:
            return 'Tier 1b: EM Forces'
    elif 'test_tier1' in nodeid:
        return 'Tier 1: Eigenstates'
    elif 'test_tier2' in nodeid or 'multiquark' in nodeid or 'hadron' in nodeid or 'baryon' in nodeid or 'meson' in nodeid:
        # Further categorize Tier 2 tests
        if 'color' in nodeid:
            if 'emergence' in nodeid:
                return 'Tier 2: Color Emergence'
            elif 'neutrality' in nodeid:
                return 'Tier 2: Color Neutrality'
            elif 'phase' in nodeid:
                return 'Tier 2: Color Phases'
            else:
                return 'Tier 2: Color Verification'
        elif 'baryon' in nodeid:
            if 'mass' in nodeid:
                return 'Tier 2: Baryon Mass'
            elif 'binding' in nodeid:
                return 'Tier 2: Binding Energy'
            elif 'confine' in nodeid:
                return 'Tier 2: Confinement'
            elif 'amplitude' in nodeid:
                return 'Tier 2: Baryon Amplitudes'
            else:
                return 'Tier 2: Baryon Solver'
        elif 'meson' in nodeid:
            if 'pion' in nodeid:
                return 'Tier 2: Pion Structure'
            else:
                return 'Tier 2: Meson Solver'
        else:
            return 'Tier 2: Multi-quark'
    elif 'test_tier3' in nodeid or 'weak_decay' in nodeid:
        return 'Tier 3: Weak Decay'
    elif 'test_grid' in nodeid:
        return 'Infrastructure: Grid'
    elif 'test_potential' in nodeid:
        return 'Infrastructure: Potentials'
    elif 'test_eigensolver' in nodeid:
        return 'Infrastructure: Eigensolvers'
    elif 'test_physics' in nodeid:
        return 'Physics Validation'
    else:
        return 'General'


def pytest_sessionfinish(session, exitstatus):
    """Called after all tests have been run."""
    reporter = get_reporter()
    reporter.end_run()
    
    # Add notes about the run
    if exitstatus == 0:
        reporter.add_note("All tests passed successfully.")
    else:
        reporter.add_note(f"Test session finished with exit status {exitstatus}.")
    
    # Check for known issues
    _check_for_known_issues(reporter)
    
    # Generate the markdown report
    report_path = reporter.generate_report()
    
    # Generate the HTML report (overwrites results.html each time)
    html_path = generate_html_results(reporter)
    
    # Print location to terminal
    print(f"\n{'='*60}")
    print(f"SFM Solver Results Report Generated")
    print(f"{'='*60}")
    print(f"Markdown: {report_path}")
    print(f"JSON:     {report_path.replace('.md', '.json')}")
    print(f"HTML:     {html_path}")
    print(f"{'='*60}\n")


def _check_for_known_issues(reporter: ResultsReporter):
    """Check for and document known issues based on test results."""
    
    # Get test statistics
    failed_tests = [t for t in reporter.test_results if not t.passed]
    passed_tests = [t for t in reporter.test_results if t.passed]
    
    # Check lepton solver tests (physics-based)
    lepton_tests = [t for t in reporter.test_results 
                   if 'lepton' in t.name.lower() or 'tier1_lepton' in t.category.lower()]
    if lepton_tests:
        lepton_passed = sum(1 for t in lepton_tests if t.passed)
        lepton_failed = sum(1 for t in lepton_tests if not t.passed)
        if lepton_failed == 0:
            reporter.add_note(
                f"Physics-based lepton solver: {lepton_passed} tests passing. "
                "Mass hierarchy emerges from four-term energy functional E = E_σ + E_x + E_coupling + E_curv."
            )
        else:
            reporter.add_issue(
                f"Lepton solver: {lepton_failed} tests failing. "
                "Check energy functional minimization parameters."
            )
    
    # Check mass ratio issue
    mass_ratio_tests = [t for t in reporter.test_results 
                       if 'mass_ratio' in t.name.lower()]
    if mass_ratio_tests:
        reporter.add_note(
            "Mass ratio tests: The physics-based lepton solver uses "
            "the four-term energy functional to achieve emergent m_μ/m_e ratio. "
            "Tau mass prediction provides theory validation."
        )
    
    # Check nonlinear convergence
    nonlinear_tests = [t for t in reporter.test_results 
                      if 'nonlinear' in t.name.lower()]
    for test in nonlinear_tests:
        if 'converge' in test.name.lower() and not test.passed:
            reporter.add_issue(
                "Nonlinear solver convergence: Simple mixing algorithm may "
                "oscillate for g1 > 0.01. DIIS and Anderson mixing now available."
            )
            break
    
    # Check tier completion
    tier1_tests = [t for t in reporter.test_results 
                  if 'tier1' in t.name.lower() and 'tier1b' not in t.name.lower()]
    tier1_passed = all(t.passed for t in tier1_tests) if tier1_tests else False
    
    if tier1_tests and tier1_passed:
        reporter.add_note("Tier 1 infrastructure tests: All passing.")
    elif tier1_tests and not tier1_passed:
        failed = sum(1 for t in tier1_tests if not t.passed)
        reporter.add_issue(f"Tier 1 tests: {failed} failures detected.")
    
    # Check Tier 2 tests (baryons, mesons, color)
    tier2_tests = [t for t in reporter.test_results 
                  if 'tier2' in t.name.lower() or 'tier 2' in t.category.lower()
                  or 'baryon' in t.name.lower() or 'meson' in t.name.lower()
                  or 'color' in t.name.lower()]
    if tier2_tests:
        tier2_passed = sum(1 for t in tier2_tests if t.passed)
        tier2_total = len(tier2_tests)
        tier2_failed = tier2_total - tier2_passed
        
        if tier2_failed == 0:
            reporter.add_note(
                f"Tier 2 multi-quark tests: {tier2_passed}/{tier2_total} passing. "
                "Color phase emergence validated, baryon structure confirmed."
            )
        else:
            reporter.add_note(
                f"Tier 2 multi-quark tests: {tier2_passed}/{tier2_total} passing, "
                f"{tier2_failed} failing."
            )
        
        # Check color emergence specifically
        color_tests = [t for t in tier2_tests if 'color' in t.name.lower()]
        if color_tests:
            color_passed = sum(1 for t in color_tests if t.passed)
            if color_passed == len(color_tests):
                reporter.add_note(
                    "Color phase emergence: All tests passing. "
                    "Three-phase structure {0, 2π/3, 4π/3} emerges from dynamics."
                )
            else:
                reporter.add_issue(
                    f"Color phase tests: {len(color_tests) - color_passed} failures. "
                    "Color emergence mechanism may need refinement."
                )


# Fixtures for test convenience

@pytest.fixture
def results_reporter():
    """Provide access to the results reporter in tests."""
    return get_reporter()


@pytest.fixture
def add_prediction(results_reporter):
    """Fixture to add prediction comparisons from tests."""
    def _add_prediction(
        parameter: str,
        predicted: float,
        experimental: float,
        uncertainty: float = 0.0,
        unit: str = "",
        target_accuracy: float = 0.10,
        notes: str = ""
    ):
        if experimental != 0:
            percent_error = abs(predicted - experimental) / experimental * 100
            within_target = percent_error <= target_accuracy * 100
        else:
            # For zero targets (like color sum), use absolute error
            # and check if predicted is "close enough" to zero
            absolute_error = abs(predicted - experimental)
            percent_error = 0.0 if absolute_error < 0.01 else absolute_error * 100
            within_target = absolute_error < 0.01  # Within 0.01 of target
        
        prediction = PredictionResult(
            parameter=parameter,
            predicted=predicted,
            experimental=experimental,
            uncertainty=uncertainty,
            unit=unit,
            percent_error=percent_error,
            within_target=within_target,
            target_accuracy=target_accuracy,
            notes=notes,
        )
        results_reporter.add_prediction(prediction)
        return prediction
    
    return _add_prediction


@pytest.fixture
def add_solver_parameter(results_reporter):
    """Fixture to record solver parameters from tests."""
    def _add_parameter(name: str, value):
        results_reporter.add_parameter(name, value)
    return _add_parameter

