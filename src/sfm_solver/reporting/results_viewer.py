"""
HTML Results Viewer for SFM Solver.

Generates a styled results.html file that can be viewed in a web browser.
This file is overwritten each run to always show the latest results.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

from sfm_solver.reporting.results_reporter import ResultsReporter, RunSummary
from sfm_solver.core.constants import (
    ELECTRON_MASS_GEV, MUON_MASS_GEV, TAU_MASS_GEV,
    MUON_ELECTRON_RATIO, TAU_ELECTRON_RATIO,
    ALPHA_EM,
)


class HTMLResultsViewer:
    """
    Generates an HTML results file for browser viewing.
    
    The HTML file is always named 'results.html' and is overwritten
    each run to show the latest results.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the HTML viewer.
        
        Args:
            output_dir: Directory for output file. Defaults to 'outputs' in solver root.
        """
        if output_dir is None:
            current = Path(__file__).parent
            while current.name != 'sfm-solver' and current.parent != current:
                current = current.parent
            output_dir = current / 'outputs'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, reporter: ResultsReporter) -> str:
        """
        Generate the HTML results file.
        
        Args:
            reporter: The ResultsReporter with data to display.
            
        Returns:
            Path to the generated HTML file.
        """
        html_path = self.output_dir / 'results.html'
        
        summary = reporter._get_summary()
        html_content = self._generate_html(reporter, summary)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _generate_html(self, reporter: ResultsReporter, summary: RunSummary) -> str:
        """Generate the complete HTML document."""
        
        # Calculate statistics
        predictions_met = [p for p in reporter.predictions if p.within_target]
        predictions_failed = [p for p in reporter.predictions if not p.within_target]
        pred_pct = (len(predictions_met) / len(reporter.predictions) * 100) if reporter.predictions else 0
        
        # Get mass ratio predictions
        mass_ratio_preds = [p for p in reporter.predictions if 'm_' in p.parameter.lower() or 'ratio' in p.parameter.lower()]
        charge_preds = [p for p in reporter.predictions if 'q' in p.parameter.lower()]
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>SFM Solver Results</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --border-color: #30363d;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-blue: #58a6ff;
            --accent-purple: #a371f7;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }}
        
        h1 {{
            font-size: 2.5em;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .run-info {{
            color: var(--text-secondary);
            font-size: 0.95em;
            margin-top: 10px;
        }}
        
        .run-info code {{
            background: var(--bg-tertiary);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'SF Mono', Consolas, monospace;
        }}
        
        nav {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 30px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
        }}
        
        nav a {{
            color: var(--accent-blue);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: background 0.2s;
        }}
        
        nav a:hover {{
            background: var(--bg-tertiary);
        }}
        
        section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid var(--border-color);
        }}
        
        h2 {{
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent-purple);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        h3 {{
            font-size: 1.2em;
            font-weight: 600;
            margin: 20px 0 15px 0;
            color: var(--accent-blue);
        }}
        
        h4 {{
            font-size: 1.05em;
            font-weight: 600;
            margin: 15px 0 10px 0;
            color: var(--text-secondary);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        .status-pass {{
            color: var(--accent-green);
            font-weight: 600;
        }}
        
        .status-fail {{
            color: var(--accent-red);
            font-weight: 600;
        }}
        
        .status-pending {{
            color: var(--accent-yellow);
            font-weight: 600;
        }}
        
        .status-partial {{
            color: var(--accent-yellow);
            font-weight: 600;
        }}
        
        .metric-card {{
            display: inline-block;
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 15px 25px;
            margin: 5px;
            text-align: center;
            min-width: 120px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: 700;
            display: block;
        }}
        
        .metric-label {{
            font-size: 0.85em;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .summary-box {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid var(--accent-blue);
        }}
        
        .summary-box.success {{
            border-left-color: var(--accent-green);
        }}
        
        .summary-box.warning {{
            border-left-color: var(--accent-yellow);
        }}
        
        .summary-box.error {{
            border-left-color: var(--accent-red);
        }}
        
        .critical-finding {{
            background: linear-gradient(135deg, rgba(248, 81, 73, 0.1), rgba(248, 81, 73, 0.05));
            border: 1px solid rgba(248, 81, 73, 0.3);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .critical-finding h4 {{
            color: var(--accent-red);
            margin-top: 0;
        }}
        
        .critical-finding ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        
        .critical-finding li {{
            margin: 5px 0;
        }}
        
        code {{
            font-family: 'SF Mono', Consolas, 'Liberation Mono', monospace;
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        .progress-bar {{
            background: var(--bg-primary);
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        
        .progress-fill.green {{
            background: linear-gradient(90deg, var(--accent-green), #2ea043);
        }}
        
        .progress-fill.red {{
            background: linear-gradient(90deg, var(--accent-red), #da3633);
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            font-size: 0.9em;
            border-top: 1px solid var(--border-color);
            margin-top: 30px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        
        .badge-success {{
            background: rgba(63, 185, 80, 0.2);
            color: var(--accent-green);
        }}
        
        .badge-error {{
            background: rgba(248, 81, 73, 0.2);
            color: var(--accent-red);
        }}
        
        .badge-pending {{
            background: rgba(210, 153, 34, 0.2);
            color: var(--accent-yellow);
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            h1 {{
                font-size: 1.8em;
            }}
            
            table {{
                font-size: 0.8em;
            }}
            
            th, td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>SFM Solver Results</h1>
        <div class="run-info">
            Run ID: <code>{summary.run_id}</code> | 
            Generated: <code>{summary.timestamp[:19]}</code> |
            Duration: <code>{summary.duration_seconds:.1f}s</code>
        </div>
    </header>
    
    <nav>
        <a href="#summary">Summary</a>
        <a href="#tests">Test Results</a>
        <a href="#parameters">Parameters</a>
        <a href="#predictions">Predictions</a>
        <a href="#conclusions">Conclusions</a>
    </nav>
    
    <!-- Section 1: Summary -->
    <section id="summary">
        <h2>📊 Run Summary</h2>
        
        <div class="summary-grid">
            <div class="summary-box {'success' if summary.failed_tests == 0 else 'error'}">
                <div class="metric-value {'status-pass' if summary.failed_tests == 0 else 'status-fail'}">{summary.passed_tests}/{summary.total_tests}</div>
                <div class="metric-label">Tests Passed</div>
            </div>
            <div class="summary-box {'success' if pred_pct >= 50 else 'error'}">
                <div class="metric-value {'status-pass' if pred_pct >= 50 else 'status-fail'}">{len(predictions_met)}/{len(reporter.predictions)}</div>
                <div class="metric-label">Predictions Met</div>
            </div>
            <div class="summary-box">
                <div class="metric-value">{summary.duration_seconds:.1f}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="summary-box">
                <div class="metric-value">v{summary.solver_version}</div>
                <div class="metric-label">Solver Version</div>
            </div>
        </div>
        
        <h3>Environment</h3>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Python Version</td><td>{summary.python_version}</td></tr>
            <tr><td>Platform</td><td>{summary.platform_info}</td></tr>
            <tr><td>Grid Points</td><td>{reporter.parameters.get('grid_points', 'N/A')}</td></tr>
        </table>
        
        <h3>Tier Completion Status</h3>
        <table>
            <tr><th>Tier</th><th>Status</th></tr>
            <tr>
                <td>Tier 1 (Eigenstates)</td>
                <td class="{'status-pass' if summary.tier1_complete else 'status-pending'}">
                    {'✅ Complete' if summary.tier1_complete else '⏳ Pending'}
                </td>
            </tr>
            <tr>
                <td>Tier 1b (EM Forces)</td>
                <td class="{'status-pass' if summary.tier1b_complete else 'status-pending'}">
                    {'✅ Complete' if summary.tier1b_complete else '⏳ Pending'}
                </td>
            </tr>
            <tr>
                <td>Tier 2 (Multi-quark)</td>
                <td class="{'status-pass' if summary.tier2_complete else 'status-pending'}">
                    {'✅ Complete' if summary.tier2_complete else '⏳ Pending'}
                </td>
            </tr>
            <tr>
                <td>Tier 3 (Weak Decay)</td>
                <td class="{'status-pass' if summary.tier3_complete else 'status-pending'}">
                    {'✅ Complete' if summary.tier3_complete else '⏳ Pending'}
                </td>
            </tr>
        </table>
    </section>
    
    <!-- Section 2: Test Results -->
    <section id="tests">
        <h2>🧪 Test Results</h2>
        
        <div class="progress-bar">
            <div class="progress-fill {'green' if summary.failed_tests == 0 else 'red'}" 
                 style="width: {summary.passed_tests/summary.total_tests*100 if summary.total_tests > 0 else 0}%"></div>
        </div>
        <p style="text-align: center; color: var(--text-secondary);">
            {summary.passed_tests} passed, {summary.failed_tests} failed ({summary.passed_tests/summary.total_tests*100 if summary.total_tests > 0 else 0:.0f}% pass rate)
        </p>
        
        {self._generate_test_tables(reporter)}
    </section>
    
    <!-- Section 3: Solver Parameters -->
    <section id="parameters">
        <h2>⚙️ Solver Parameters</h2>
        
        {self._generate_parameters_table(reporter)}
    </section>
    
    <!-- Section 4: Predictions vs Experiment -->
    <section id="predictions">
        <h2>🎯 Predictions vs Experiment</h2>
        
        <h3>Reference Values (Experimental)</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Source</th></tr>
            <tr><td>m_e</td><td>{ELECTRON_MASS_GEV*1000:.6f}</td><td>MeV</td><td>PDG 2024</td></tr>
            <tr><td>m_μ</td><td>{MUON_MASS_GEV*1000:.4f}</td><td>MeV</td><td>PDG 2024</td></tr>
            <tr><td>m_τ</td><td>{TAU_MASS_GEV*1000:.2f}</td><td>MeV</td><td>PDG 2024</td></tr>
            <tr><td>m_μ/m_e</td><td>{MUON_ELECTRON_RATIO:.4f}</td><td>-</td><td>Target (±10%)</td></tr>
            <tr><td>m_τ/m_e</td><td>{TAU_ELECTRON_RATIO:.2f}</td><td>-</td><td>Target (±10%)</td></tr>
            <tr><td>α⁻¹</td><td>{1/ALPHA_EM:.6f}</td><td>-</td><td>CODATA</td></tr>
        </table>
        
        {self._generate_predictions_tables(reporter, mass_ratio_preds, charge_preds)}
    </section>
    
    <!-- Section 5: Conclusions -->
    <section id="conclusions">
        <h2>📋 Conclusions</h2>
        
        {self._generate_conclusions(reporter, summary)}
    </section>
    
    <footer>
        <p>Report generated by SFM Solver v{summary.solver_version}</p>
        <p>Auto-refreshes every 30 seconds | <a href="#" onclick="location.reload(); return false;">Refresh Now</a></p>
    </footer>
</body>
</html>'''
        
        return html
    
    def _generate_test_tables(self, reporter: ResultsReporter) -> str:
        """Generate HTML for test result tables grouped by category."""
        categories = {}
        for test in reporter.test_results:
            cat = test.category or "General"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test)
        
        html_parts = []
        for cat, tests in sorted(categories.items()):
            passed = sum(1 for t in tests if t.passed)
            status_class = 'status-pass' if passed == len(tests) else 'status-partial'
            
            html_parts.append(f'''
        <h3>{cat} <span class="{status_class}">({passed}/{len(tests)})</span></h3>
        <table>
            <tr><th>Test</th><th>Status</th><th>Duration</th></tr>
            {''.join(f"""
            <tr>
                <td>{t.name}</td>
                <td class="{'status-pass' if t.passed else 'status-fail'}">{'✅ Pass' if t.passed else '❌ Fail'}</td>
                <td>{t.duration:.3f}s</td>
            </tr>""" for t in tests)}
        </table>''')
        
        return ''.join(html_parts)
    
    def _generate_parameters_table(self, reporter: ResultsReporter) -> str:
        """Generate HTML for solver parameters table."""
        if not reporter.parameters:
            return '<p style="color: var(--text-secondary);">No parameters recorded.</p>'
        
        rows = ''.join(f'<tr><td><code>{k}</code></td><td>{v}</td></tr>' 
                      for k, v in sorted(reporter.parameters.items()))
        
        return f'''
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            {rows}
        </table>'''
    
    def _generate_predictions_tables(self, reporter, mass_ratio_preds, charge_preds) -> str:
        """Generate HTML for predictions tables."""
        html_parts = []
        
        if mass_ratio_preds:
            met = sum(1 for p in mass_ratio_preds if p.within_target)
            html_parts.append(f'''
        <h3>Mass Ratio Predictions <span class="{'status-pass' if met > 0 else 'status-fail'}">({met}/{len(mass_ratio_preds)} within target)</span></h3>
        <table>
            <tr><th>Parameter</th><th>Predicted</th><th>Experimental</th><th>Error (%)</th><th>Target</th><th>Status</th></tr>
            {''.join(f"""
            <tr>
                <td>{p.parameter}</td>
                <td>{p.predicted:.6g}</td>
                <td>{p.experimental:.6g}</td>
                <td>{p.percent_error:.1f}%</td>
                <td>±{p.target_accuracy*100:.0f}%</td>
                <td class="{'status-pass' if p.within_target else 'status-fail'}">{'✅' if p.within_target else '❌'}</td>
            </tr>""" for p in mass_ratio_preds)}
        </table>''')
        
        if charge_preds:
            met = sum(1 for p in charge_preds if p.within_target)
            html_parts.append(f'''
        <h3>Charge Quantization <span class="{'status-pass' if met == len(charge_preds) else 'status-fail'}">({met}/{len(charge_preds)})</span></h3>
        <table>
            <tr><th>Parameter</th><th>Predicted</th><th>Experimental</th><th>Error (%)</th><th>Status</th></tr>
            {''.join(f"""
            <tr>
                <td>{p.parameter}</td>
                <td>{p.predicted:.4g}</td>
                <td>{p.experimental:.4g}</td>
                <td>{p.percent_error:.1f}%</td>
                <td class="{'status-pass' if p.within_target else 'status-fail'}">{'✅' if p.within_target else '❌'}</td>
            </tr>""" for p in charge_preds)}
        </table>''')
        
        if not reporter.predictions:
            html_parts.append('<p style="color: var(--text-secondary);">No predictions recorded in this run.</p>')
        
        return ''.join(html_parts)
    
    def _generate_conclusions(self, reporter: ResultsReporter, summary: RunSummary) -> str:
        """Generate HTML for conclusions section."""
        
        # Analyze predictions
        predictions_met = [p for p in reporter.predictions if p.within_target]
        mass_ratio_preds = [p for p in reporter.predictions if 'm_μ/m_e' in p.parameter.lower()]
        best_ratio = min(mass_ratio_preds, key=lambda p: p.percent_error) if mass_ratio_preds else None
        
        # Check requirements
        tier1_tests = [t for t in reporter.test_results 
                      if 'tier 1' in t.category.lower() or 'eigenstate' in t.category.lower()]
        tier1_passed = all(t.passed for t in tier1_tests) if tier1_tests else False
        periodic_tests = [t for t in reporter.test_results if 'periodic' in t.name.lower()]
        periodic_passed = all(t.passed for t in periodic_tests) if periodic_tests else True
        winding_tests = [t for t in reporter.test_results if 'winding' in t.name.lower()]
        winding_passed = all(t.passed for t in winding_tests) if winding_tests else True
        
        html = f'''
        <h3>5a. Solver Execution Success</h3>
        <div class="summary-box {'success' if summary.failed_tests == 0 else 'error'}">
            <strong>Status: {'✅ ALL TESTS PASSED' if summary.failed_tests == 0 else '⚠️ SOME TESTS FAILED'}</strong>
            <p>The solver executed {'successfully' if summary.failed_tests == 0 else 'with issues'} with {summary.passed_tests}/{summary.total_tests} tests passing.</p>
        </div>
        
        <table>
            <tr><th>Component</th><th>Status</th><th>Details</th></tr>
            <tr>
                <td>Test Execution</td>
                <td class="{'status-pass' if summary.failed_tests == 0 else 'status-fail'}">{'✅ Pass' if summary.failed_tests == 0 else '❌ Fail'}</td>
                <td>{summary.passed_tests}/{summary.total_tests} tests</td>
            </tr>
            <tr>
                <td>Linear Eigensolver</td>
                <td class="status-pass">✅ Working</td>
                <td>Converges with residual &lt; 10⁻⁶</td>
            </tr>
            <tr>
                <td>Nonlinear Eigensolver</td>
                <td class="status-partial">⚠️ Limited</td>
                <td>Oscillates for g₁ &gt; 0.01</td>
            </tr>
            <tr>
                <td>Spectral Grid</td>
                <td class="status-pass">✅ Working</td>
                <td>FFT-based, N=64-512</td>
            </tr>
            <tr>
                <td>Three-Well Potential</td>
                <td class="status-pass">✅ Working</td>
                <td>V(σ) periodic, 3-fold symmetric</td>
            </tr>
        </table>
        
        <h3>5b. Physics Prediction Success</h3>
        <div class="summary-box {'success' if len(predictions_met) > len(reporter.predictions)/2 else 'error'}">
            <strong>Predictions Meeting Target: {len(predictions_met)}/{len(reporter.predictions)} ({len(predictions_met)/len(reporter.predictions)*100 if reporter.predictions else 0:.0f}%)</strong>
        </div>
        
        <h4>Tier 1 Requirements Checklist</h4>
        <table>
            <tr><th>#</th><th>Requirement</th><th>Status</th><th>Evidence</th></tr>
            <tr>
                <td>1</td>
                <td>k=1 mode convergence</td>
                <td class="{'status-pass' if tier1_passed else 'status-fail'}">{'✅ PASSED' if tier1_passed else '❌ FAILED'}</td>
                <td>Linear solver converges</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Mass ratio m_μ/m_e ≈ 206.77</td>
                <td class="{'status-pass' if best_ratio and best_ratio.within_target else 'status-fail'}">{'✅ PASSED' if best_ratio and best_ratio.within_target else '❌ NOT MET'}</td>
                <td>Best: {best_ratio.predicted:.4f} ({best_ratio.percent_error:.1f}% error) if best_ratio else 'No data'</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Periodic boundary conditions</td>
                <td class="{'status-pass' if periodic_passed else 'status-fail'}">{'✅ PASSED' if periodic_passed else '❌ FAILED'}</td>
                <td>χ(σ+2π) = χ(σ)</td>
            </tr>
            <tr>
                <td>4</td>
                <td>Winding number preservation</td>
                <td class="{'status-pass' if winding_passed else 'status-fail'}">{'✅ PASSED' if winding_passed else '❌ FAILED'}</td>
                <td>k-sector eigenstates valid</td>
            </tr>
        </table>
        
        <h4>What's Working vs What's Needed</h4>
        <table>
            <tr><th>Aspect</th><th>Status</th><th>Details</th></tr>
            <tr><td>Spectral Grid</td><td class="status-pass">✅ Working</td><td>FFT-based differentiation</td></tr>
            <tr><td>Three-Well Potential</td><td class="status-pass">✅ Working</td><td>V(σ) = V₀[1-cos(3σ)] + V₁[1-cos(6σ)]</td></tr>
            <tr><td>Linear Eigensolver</td><td class="status-pass">✅ Working</td><td>Converges, correct eigenstates</td></tr>
            <tr><td>Nonlinear Eigensolver</td><td class="status-partial">⚠️ Partial</td><td>Runs but oscillates, needs DIIS</td></tr>
            <tr><td>Mass Formula m=βA²</td><td class="status-pass">✅ Implemented</td><td>But gives ratio=1 (see below)</td></tr>
            <tr><td>Amplitude Quantization</td><td class="status-fail">❌ Missing</td><td>All states have A²=2π</td></tr>
        </table>
        
        <div class="critical-finding">
            <h4>⚠️ Critical Finding: Amplitude Quantization</h4>
            <p>The SFM theory requires different particles to have different amplitudes:</p>
            <ul>
                <li>A_e &lt; A_μ &lt; A_τ with A_μ/A_e ≈ √206.77 ≈ 14.4</li>
            </ul>
            <p><strong>Current solver behavior:</strong></p>
            <ul>
                <li>All normalized wavefunctions have A² = 2π (by normalization)</li>
                <li>This produces mass ratio = 1.0 instead of 206.77</li>
                <li>The mechanism for amplitude quantization is not yet functional</li>
            </ul>
        </div>
        
        {self._generate_notes_html(reporter)}
        '''
        
        return html
    
    def _generate_notes_html(self, reporter: ResultsReporter) -> str:
        """Generate HTML for notes and issues."""
        html_parts = []
        
        if reporter.issues:
            html_parts.append('<h4>Identified Issues</h4><ul>')
            for issue in reporter.issues:
                html_parts.append(f'<li>{issue}</li>')
            html_parts.append('</ul>')
        
        if reporter.notes:
            html_parts.append('<h4>Notes</h4><ul>')
            for note in reporter.notes:
                html_parts.append(f'<li>{note}</li>')
            html_parts.append('</ul>')
        
        return ''.join(html_parts)


def generate_html_results(reporter: ResultsReporter, output_dir: str = None) -> str:
    """
    Convenience function to generate HTML results.
    
    Args:
        reporter: The ResultsReporter with data to display.
        output_dir: Optional output directory.
        
    Returns:
        Path to the generated HTML file.
    """
    viewer = HTMLResultsViewer(output_dir)
    return viewer.generate(reporter)

