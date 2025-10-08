"""
Report generation for evaluation results.
"""

from pathlib import Path
from typing import Optional

from .schemas import EvalReport, EvalResult
from .telemetry import write_csv, write_parquet


def save_report(
    report: EvalReport,
    out_dir: str | Path,
    run_id: Optional[str] = None,
) -> Path:
    """
    Save evaluation report to disk (CSV, Parquet, and Markdown summary).
    
    Args:
        report: EvalReport to save
        out_dir: Output directory
        run_id: Optional run ID for directory naming; uses dataset name if not provided
        
    Returns:
        Path to the created run directory
        
    Example:
        >>> from rag_papers.eval import save_report
        >>> save_report(report, "runs/", run_id="abc-123")
        PosixPath('runs/run_abc-123')
    """
    out_dir = Path(out_dir)
    
    # Create run-specific directory
    if run_id:
        run_dir = out_dir / f"run_{run_id}"
    else:
        # Use dataset name and try to find unique dir
        base_name = f"run_{report.dataset}"
        run_dir = out_dir / base_name
        counter = 1
        while run_dir.exists():
            run_dir = out_dir / f"{base_name}_{counter}"
            counter += 1
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to dicts for CSV/Parquet
    result_rows = []
    for r in report.results:
        result_rows.append({
            "id": r.id,
            "query": r.meta.get("query", ""),
            "intent": r.meta.get("intent", ""),
            "answer": r.answer,
            "accepted": r.accepted,
            "verify_score": r.verify_score,
            "repair_used": r.repair_used,
            "steps": "|".join(r.steps),
            "total_ms": sum(t.ms for t in r.timings),
            "retrieved_count": r.retrieved_count,
            "pruned_chars": r.pruned_chars,
            "context_chars": r.context_chars,
        })
    
    # Save CSV
    csv_path = run_dir / "results.csv"
    write_csv(csv_path, result_rows)
    
    # Save Parquet
    parquet_path = run_dir / "results.parquet"
    write_parquet(parquet_path, result_rows)
    
    # Save Markdown summary
    summary_path = run_dir / "summary.md"
    _write_markdown_summary(report, summary_path)
    
    return run_dir


def _write_markdown_summary(report: EvalReport, path: Path) -> None:
    """Write a Markdown summary of the evaluation report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report: {report.dataset}\n\n")
        
        # Metrics table
        f.write("## Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        metrics = report.metrics
        f.write(f"| Accept@1 | {metrics.get('accept_at_1', 0):.2%} |\n")
        f.write(f"| Avg Score | {metrics.get('avg_score', 0):.3f} |\n")
        f.write(f"| Repair Rate | {metrics.get('repair_rate', 0):.2%} |\n")
        f.write(f"| Total Queries | {int(metrics.get('total_queries', 0))} |\n")
        f.write(f"| Avg Retrieved | {metrics.get('avg_retrieved_count', 0):.1f} |\n")
        f.write(f"| Avg Context Chars | {metrics.get('avg_context_chars', 0):.0f} |\n")
        f.write(f"| Avg Pruned Chars | {metrics.get('avg_pruned_chars', 0):.0f} |\n")
        f.write("\n")
        
        # Latency table
        f.write("## Latency\n\n")
        f.write("| Statistic | Value (ms) |\n")
        f.write("|-----------|------------|\n")
        f.write(f"| P50 | {metrics.get('latency_p50_ms', 0):.1f} |\n")
        f.write(f"| P95 | {metrics.get('latency_p95_ms', 0):.1f} |\n")
        f.write(f"| Mean | {metrics.get('latency_mean_ms', 0):.1f} |\n")
        f.write(f"| Max | {metrics.get('latency_max_ms', 0):.1f} |\n")
        f.write("\n")
        
        # Worst cases (bottom 3 by verify score)
        worst = sorted(report.results, key=lambda r: r.verify_score)[:3]
        
        if worst:
            f.write("## Top 3 Worst Cases (by verify score)\n\n")
            for i, result in enumerate(worst, 1):
                f.write(f"### {i}. Query ID: {result.id}\n\n")
                f.write(f"**Query:** {result.meta.get('query', 'N/A')}\n\n")
                f.write(f"**Intent:** {result.meta.get('intent', 'N/A')}\n\n")
                f.write(f"**Verify Score:** {result.verify_score:.3f}\n\n")
                f.write(f"**Accepted:** {result.accepted}\n\n")
                f.write(f"**Repair Used:** {result.repair_used}\n\n")
                f.write(f"**Path:** {' → '.join(result.steps)}\n\n")
                f.write(f"**Answer:** {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}\n\n")
        
        # Best cases (top 3 by verify score)
        best = sorted(report.results, key=lambda r: r.verify_score, reverse=True)[:3]
        
        if best:
            f.write("## Top 3 Best Cases (by verify score)\n\n")
            for i, result in enumerate(best, 1):
                f.write(f"### {i}. Query ID: {result.id}\n\n")
                f.write(f"**Query:** {result.meta.get('query', 'N/A')}\n\n")
                f.write(f"**Intent:** {result.meta.get('intent', 'N/A')}\n\n")
                f.write(f"**Verify Score:** {result.verify_score:.3f}\n\n")
                f.write(f"**Accepted:** {result.accepted}\n\n")
                f.write(f"**Path:** {' → '.join(result.steps)}\n\n")


def save_html(
    report: EvalReport,
    path: str | Path,
) -> None:
    """
    Save evaluation report as HTML.
    
    Args:
        report: EvalReport to save
        path: Output HTML file path
        
    Note:
        This is a minimal HTML export without heavy dependencies.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head>\n")
        f.write(f"<title>Evaluation Report: {report.dataset}</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 40px; }\n")
        f.write("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("th { background-color: #4CAF50; color: white; }\n")
        f.write("tr:nth-child(even) { background-color: #f2f2f2; }\n")
        f.write(".metric-value { font-weight: bold; }\n")
        f.write(".worst-case { background-color: #ffebee; padding: 10px; margin: 10px 0; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        
        f.write(f"<h1>Evaluation Report: {report.dataset}</h1>\n")
        
        # Metrics table
        f.write("<h2>Metrics</h2>\n")
        f.write("<table>\n")
        f.write("<tr><th>Metric</th><th>Value</th></tr>\n")
        
        metrics = report.metrics
        f.write(f"<tr><td>Accept@1</td><td class='metric-value'>{metrics.get('accept_at_1', 0):.2%}</td></tr>\n")
        f.write(f"<tr><td>Avg Score</td><td class='metric-value'>{metrics.get('avg_score', 0):.3f}</td></tr>\n")
        f.write(f"<tr><td>Repair Rate</td><td class='metric-value'>{metrics.get('repair_rate', 0):.2%}</td></tr>\n")
        f.write(f"<tr><td>Total Queries</td><td class='metric-value'>{int(metrics.get('total_queries', 0))}</td></tr>\n")
        f.write(f"<tr><td>P50 Latency (ms)</td><td class='metric-value'>{metrics.get('latency_p50_ms', 0):.1f}</td></tr>\n")
        f.write(f"<tr><td>P95 Latency (ms)</td><td class='metric-value'>{metrics.get('latency_p95_ms', 0):.1f}</td></tr>\n")
        f.write("</table>\n")
        
        # Worst cases
        worst = sorted(report.results, key=lambda r: r.verify_score)[:3]
        
        if worst:
            f.write("<h2>Top 3 Worst Cases</h2>\n")
            for i, result in enumerate(worst, 1):
                f.write("<div class='worst-case'>\n")
                f.write(f"<h3>{i}. Query ID: {result.id}</h3>\n")
                f.write(f"<p><strong>Query:</strong> {result.meta.get('query', 'N/A')}</p>\n")
                f.write(f"<p><strong>Verify Score:</strong> {result.verify_score:.3f}</p>\n")
                f.write(f"<p><strong>Accepted:</strong> {result.accepted}</p>\n")
                f.write(f"<p><strong>Path:</strong> {' → '.join(result.steps)}</p>\n")
                f.write("</div>\n")
        
        f.write("</body>\n</html>\n")
