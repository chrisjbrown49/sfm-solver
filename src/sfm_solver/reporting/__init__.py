"""
Reporting module for SFM Solver.

Provides automated results file generation with:
- Run summaries and timing information
- Test pass/fail status
- Parameter solutions and predictions
- Comparison with experimental values
- Conclusions and issue identification

Output formats:
- Markdown (.md) - for documentation and git tracking
- JSON (.json) - for programmatic access
- HTML (results.html) - for browser viewing (overwritten each run)
"""

from sfm_solver.reporting.results_reporter import ResultsReporter, TestResult, RunSummary
from sfm_solver.reporting.results_viewer import HTMLResultsViewer, generate_html_results

__all__ = [
    'ResultsReporter', 
    'TestResult', 
    'RunSummary',
    'HTMLResultsViewer',
    'generate_html_results',
]

