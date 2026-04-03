"""
Report rendering: Rich console output + JSON export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.rule import Rule

from .models import SecurityReport, EvaluatorResult, EvaluatorStatus, Severity, Finding


_SEVERITY_COLORS: dict[str, str] = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim white",
    Severity.UNKNOWN: "dim white",
}

_STATUS_ICONS: dict[EvaluatorStatus, str] = {
    EvaluatorStatus.PASS: "[green]PASS[/green]",
    EvaluatorStatus.FAIL: "[red]FAIL[/red]",
    EvaluatorStatus.WARNING: "[yellow]WARN[/yellow]",
    EvaluatorStatus.ERROR: "[red]ERROR[/red]",
    EvaluatorStatus.SKIPPED: "[dim]SKIP[/dim]",
}

_SEVERITY_BADGE: dict[Severity, str] = {
    Severity.CRITICAL: "[bold white on red] CRIT [/bold white on red]",
    Severity.HIGH:     "[bold white on dark_red] HIGH [/bold white on dark_red]",
    Severity.MEDIUM:   "[bold black on yellow] MED  [/bold black on yellow]",
    Severity.LOW:      "[bold black on cyan]  LOW  [/bold black on cyan]",
    Severity.INFO:     "[dim white on grey23]  INFO [/dim white on grey23]",
    Severity.UNKNOWN:  "[dim] UNK [/dim]",
}


def render_report(report: SecurityReport, console: Console | None = None, verbose: bool = False) -> None:
    if console is None:
        console = Console()

    # ─── Header ───────────────────────────────────────────────────────────────
    overall = report.overall_severity
    header_style = _SEVERITY_COLORS.get(overall, "white")
    console.print()
    console.print(Panel(
        f"[bold]Model Security Evaluation Report[/bold]\n"
        f"Model: [bold cyan]{report.model_id}[/bold cyan]  "
        f"Source: [bold]{report.model_source.value}[/bold]\n"
        f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Overall Risk: [{header_style}]{overall.value.upper()}[/{header_style}]",
        title="[bold]modelcheck[/bold]",
        border_style=header_style,
        expand=False,
    ))

    # ─── Summary table ────────────────────────────────────────────────────────
    counts = report.total_findings()
    summary_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    summary_table.add_column("Category", style="bold", no_wrap=True, min_width=28)
    summary_table.add_column("Status", justify="center", no_wrap=True, width=6)
    summary_table.add_column("CRIT", justify="right", style="bold red", width=5)
    summary_table.add_column("HIGH", justify="right", style="red", width=5)
    summary_table.add_column("MED", justify="right", style="yellow", width=5)
    summary_table.add_column("LOW", justify="right", style="cyan", width=5)
    summary_table.add_column("INFO", justify="right", style="dim", width=5)
    summary_table.add_column("Summary", overflow="fold", min_width=40)

    for result in [report.provenance, report.adversarial, report.privacy, report.file_security]:
        if result is None:
            continue
        summary_table.add_row(
            result.name,
            _STATUS_ICONS.get(result.status, result.status.value),
            str(result.critical_count) if result.critical_count else "[dim]0[/dim]",
            str(result.high_count) if result.high_count else "[dim]0[/dim]",
            str(result.medium_count) if result.medium_count else "[dim]0[/dim]",
            str(result.low_count) if result.low_count else "[dim]0[/dim]",
            str(sum(1 for f in result.findings if f.severity == Severity.INFO)),
            result.summary,
        )

    # Totals row
    summary_table.add_section()
    summary_table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"[bold red]{counts['critical']}[/bold red]" if counts["critical"] else "[dim]0[/dim]",
        f"[red]{counts['high']}[/red]" if counts["high"] else "[dim]0[/dim]",
        f"[yellow]{counts['medium']}[/yellow]" if counts["medium"] else "[dim]0[/dim]",
        f"[cyan]{counts['low']}[/cyan]" if counts["low"] else "[dim]0[/dim]",
        str(counts["info"]),
        f"[bold]{counts['total']} total finding(s)[/bold]",
    )

    console.print(summary_table)

    # ─── Detailed findings ────────────────────────────────────────────────────
    results = [r for r in [report.provenance, report.adversarial, report.privacy, report.file_security] if r]
    for result in results:
        if not result.findings and not verbose:
            continue
        _render_evaluator_section(console, result, verbose)

    # ─── Footer ───────────────────────────────────────────────────────────────
    console.print(Rule(style="dim"))
    if counts["critical"] + counts["high"] > 0:
        console.print(
            f"[bold red]ACTION REQUIRED:[/bold red] "
            f"{counts['critical']} critical and {counts['high']} high severity findings detected."
        )
    else:
        console.print("[green]No critical or high severity findings.[/green]")
    console.print()


def _render_evaluator_section(console: Console, result: EvaluatorResult, verbose: bool) -> None:
    status_icon = _STATUS_ICONS.get(result.status, result.status.value)
    console.print(Panel(
        f"{status_icon}  {result.summary}",
        title=f"[bold]{result.name}[/bold]",
        border_style="red" if result.status == EvaluatorStatus.FAIL else
                      "yellow" if result.status == EvaluatorStatus.WARNING else "green",
        expand=True,
    ))

    if not result.findings:
        if verbose:
            console.print("  [dim]No findings.[/dim]")
        return

    findings_table = Table(box=box.MINIMAL, show_header=True, header_style="bold dim", expand=True)
    findings_table.add_column("ID", style="dim", no_wrap=True, width=22)
    findings_table.add_column("Sev", no_wrap=True, width=8)
    findings_table.add_column("Title", overflow="fold")
    if verbose:
        findings_table.add_column("Remediation", overflow="fold")

    for finding in sorted(result.findings, key=lambda f: _severity_sort_key(f.severity)):
        badge = _SEVERITY_BADGE.get(finding.severity, finding.severity.value)
        if verbose:
            findings_table.add_row(
                finding.id,
                badge,
                f"[bold]{finding.title}[/bold]\n[dim]{finding.description[:300]}[/dim]",
                finding.remediation[:200] if finding.remediation else "",
            )
        else:
            findings_table.add_row(finding.id, badge, finding.title)

    console.print(findings_table)

    if verbose and result.metadata:
        _render_metadata(console, result.metadata)


def _render_metadata(console: Console, metadata: dict) -> None:
    if not metadata:
        return
    interesting = {
        k: v for k, v in metadata.items()
        if k not in ("model_id",) and v not in (None, [], {}, "")
    }
    if not interesting:
        return
    console.print("  [dim]Metadata:[/dim]")
    for k, v in interesting.items():
        if isinstance(v, (dict, list)):
            console.print(f"    [dim]{k}:[/dim] {json.dumps(v, default=str)[:200]}")
        else:
            console.print(f"    [dim]{k}:[/dim] {v}")


def _severity_sort_key(s: Severity) -> int:
    order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2,
             Severity.LOW: 3, Severity.INFO: 4, Severity.UNKNOWN: 5}
    return order.get(s, 99)


def export_json(report: SecurityReport, path: str) -> None:
    """Export the full report as JSON."""
    output = report.model_dump(mode="json")
    Path(path).write_text(json.dumps(output, indent=2, default=str))
