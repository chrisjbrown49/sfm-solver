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
        "markers", "coupled: marks tests for coupled subspace-spacetime solver"
    )
    config.addinivalue_line(
        "markers", "mass_hierarchy: marks tests for mass hierarchy validation"
    )
    config.addinivalue_line(
        "markers", "amplitude: marks tests for amplitude quantization solver"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
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
        elif marker.name == 'tier3':
            return 'Tier 3: Weak Decay'
        elif marker.name == 'coupled':
            return 'Coupled Solver: Mass Hierarchy'
        elif marker.name == 'mass_hierarchy':
            return 'Physics: Mass Hierarchy'
        elif marker.name == 'amplitude':
            return 'Amplitude Solver: Quantization'
    
    # Infer from test file/class name
    nodeid = item.nodeid.lower()
    
    # Check file name first for more accurate categorization
    if 'test_amplitude' in nodeid or 'amplitude_solver' in nodeid:
        if 'ite' in nodeid:
            return 'Amplitude Solver: ITE'
        elif 'branch' in nodeid or 'continuation' in nodeid:
            return 'Amplitude Solver: Branch Continuation'
        else:
            return 'Amplitude Solver: Quantization'
    elif 'test_nonlinear_coupled' in nodeid:
        return 'Amplitude Solver: Nonlinear Coupled'
    elif 'test_coupled' in nodeid:
        # Further categorize coupled tests
        if 'radial' in nodeid:
            return 'Coupled Solver: Radial Grid'
        elif 'hamiltonian' in nodeid:
            return 'Coupled Solver: Hamiltonian'
        elif 'mass' in nodeid or 'hierarchy' in nodeid or 'fitter' in nodeid:
            return 'Coupled Solver: Mass Hierarchy'
        elif 'physics' in nodeid:
            return 'Physics: Mass Predictions'
        else:
            return 'Coupled Solver: Eigenvalue Problem'
    elif 'test_tier1b' in nodeid or 'electromagnetic' in nodeid:
        return 'Tier 1b: EM Forces'
    elif 'test_tier1' in nodeid:
        return 'Tier 1: Eigenstates'
    elif 'test_tier2' in nodeid or 'multiquark' in nodeid or 'hadron' in nodeid:
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
    
    # Check coupled solver tests
    coupled_tests = [t for t in reporter.test_results 
                    if 'coupled' in t.name.lower() or 'coupled' in t.category.lower()]
    if coupled_tests:
        coupled_passed = sum(1 for t in coupled_tests if t.passed)
        coupled_failed = sum(1 for t in coupled_tests if not t.passed)
        if coupled_failed == 0:
            reporter.add_note(
                f"Coupled subspace-spacetime solver: {coupled_passed} tests passing. "
                "This solver includes the H_coupling = -alpha(d^2/dr/dsigma) term "
                "that creates the mass hierarchy between particles with same winding number."
            )
        else:
            reporter.add_issue(
                f"Coupled solver: {coupled_failed} tests failing. "
                "The spacetime-subspace coupling mechanism may need adjustment."
            )
    
    # Check mass hierarchy/fitting tests
    hierarchy_tests = [t for t in reporter.test_results 
                      if 'hierarchy' in t.name.lower() or 'fitter' in t.name.lower() 
                      or 'alpha' in t.name.lower()]
    if hierarchy_tests:
        passed = all(t.passed for t in hierarchy_tests)
        if passed:
            reporter.add_note(
                "Mass hierarchy fitting: Tests verify that the coupling constant alpha "
                "can be fitted to reproduce m_mu/m_e ratio, and tau mass emerges as prediction."
            )
    
    # Check mass ratio issue
    mass_ratio_tests = [t for t in reporter.test_results 
                       if 'mass_ratio' in t.name.lower()]
    if mass_ratio_tests:
        reporter.add_note(
            "Mass ratio tests: The coupled solver with alpha fitting "
            "aims to achieve m_mu/m_e ratio from first principles. "
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
        percent_error = abs(predicted - experimental) / experimental * 100 if experimental != 0 else float('inf')
        within_target = percent_error <= target_accuracy * 100
        
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

