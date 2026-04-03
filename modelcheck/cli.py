"""
modelcheck CLI — ML Model Security Evaluation Tool
"""

from __future__ import annotations

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .models import ModelSource, SecurityReport
from .report import render_report, export_json


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _detect_source(model: str) -> ModelSource:
    """Determine if model is a local path or HuggingFace ID."""
    p = Path(model)
    if p.exists():
        return ModelSource.LOCAL
    return ModelSource.HUGGINGFACE


def _resolve_token(token: str | None) -> str | None:
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _scan_one(
    model: str,
    hf_token: str | None,
    skip: tuple[str, ...],
    download: bool,
    no_probes: bool,
    rule_config,
    progress: Progress | None = None,
    task_id=None,
) -> SecurityReport:
    """Scan a single model and return a SecurityReport."""
    source = _detect_source(model)
    hf_info: dict | None = None
    local_path: str | None = None

    def _upd(msg: str) -> None:
        if progress is not None and task_id is not None:
            progress.update(task_id, description=msg)

    _upd(f"[dim]{model}[/dim] — fetching metadata…")

    if source == ModelSource.HUGGINGFACE:
        from .utils.hf_utils import fetch_hf_model_info
        hf_info = fetch_hf_model_info(model, token=hf_token)
    else:
        local_path = model
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Path does not exist: {local_path}")

    report = SecurityReport(model_id=model, model_source=source)

    if "provenance" not in skip:
        _upd(f"[dim]{model}[/dim] — [1/4] Provenance…")
        from .evaluators.provenance import evaluate_provenance
        report.provenance = evaluate_provenance(model, source, hf_info, local_path)

    if "adversarial" not in skip:
        _upd(f"[dim]{model}[/dim] — [2/4] Adversarial…")
        from .evaluators.adversarial import evaluate_adversarial
        report.adversarial = evaluate_adversarial(
            model, source, hf_info, local_path, hf_token,
            run_inference_probes=not no_probes,
        )

    if "privacy" not in skip:
        _upd(f"[dim]{model}[/dim] — [3/4] Privacy…")
        from .evaluators.privacy import evaluate_privacy
        report.privacy = evaluate_privacy(model, source, hf_info, local_path)

    if "file-security" not in skip:
        _upd(f"[dim]{model}[/dim] — [4/4] File Security…")
        from .evaluators.file_security import evaluate_file_security
        report.file_security = evaluate_file_security(
            model, source, hf_info, local_path, download, hf_token,
        )

    if rule_config:
        for result in [report.provenance, report.adversarial,
                       report.privacy, report.file_security]:
            if result is not None:
                result.findings = rule_config.apply(result.findings)

    _upd(f"[green]{model}[/green] — done")
    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="modelcheck")
def main() -> None:
    """modelcheck — ML Model Security Evaluation Tool.

    Evaluates models (HuggingFace or local) across four security dimensions:

    \b
    1. Supply Chain & Provenance
    2. Adversarial Robustness
    3. Data & Privacy Risks
    4. Model File Security (static scanning)
    """


@main.command("scan")
@click.argument("models", nargs=-1, required=True, metavar="MODEL [MODEL ...]")
@click.option(
    "--token", "-t",
    envvar="HF_TOKEN",
    default=None,
    help=(
        "HuggingFace API token (or set HF_TOKEN env var). "
        "Create a fine-grained token with 'Read access to contents of all public gated repos "
        "you can access' at https://huggingface.co/settings/tokens/new?tokenType=fineGrained"
    ),
)
@click.option(
    "--output", "-o",
    default=None,
    metavar="FILE.json or DIR/",
    help=(
        "Export JSON report(s). For a single model: path to a .json file. "
        "For multiple models: path to a directory (created if absent)."
    ),
)
@click.option(
    "--html",
    "html_out",
    default=None,
    metavar="FILE.html",
    help=(
        "Write the HTML report to this path. "
        "Defaults to <model_id>_security_card.html (single) or "
        "modelcheck_report.html (multi)."
    ),
)
@click.option(
    "--no-html",
    is_flag=True,
    default=False,
    help="Suppress automatic HTML report generation.",
)
@click.option(
    "--parallelism", "-p",
    default=4,
    show_default=True,
    type=click.IntRange(1, 16),
    help="Maximum number of models to scan in parallel.",
)
@click.option(
    "--skip", "-s",
    multiple=True,
    type=click.Choice(["provenance", "adversarial", "privacy", "file-security"]),
    help="Skip one or more evaluation modules (can be repeated).",
)
@click.option(
    "--download/--no-download",
    default=False,
    help="Download HuggingFace model files for deep static analysis.",
)
@click.option(
    "--no-probes",
    is_flag=True,
    default=False,
    help="Disable live adversarial inference probes (faster, but less thorough).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Show full finding descriptions and metadata.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except JSON (useful for CI pipelines).",
)
@click.option(
    "--fail-on",
    default="high",
    type=click.Choice(["critical", "high", "medium", "low", "never"]),
    help="Exit with non-zero code if findings at this severity or above are found. Default: high.",
)
@click.option(
    "--rules", "-r",
    "rules_file",
    default=None,
    metavar="rules.yaml",
    help=(
        "Path to a rules.yaml file. Overrides severity, title, and remediation "
        "per rule ID; rules with 'enabled: false' are suppressed."
    ),
)
def scan(
    models: tuple[str, ...],
    token: Optional[str],
    output: Optional[str],
    html_out: Optional[str],
    no_html: bool,
    parallelism: int,
    skip: tuple[str, ...],
    download: bool,
    no_probes: bool,
    verbose: bool,
    quiet: bool,
    fail_on: str,
    rules_file: Optional[str],
) -> None:
    """Evaluate the security of one or more models.

    MODEL [MODEL ...] — one or more HuggingFace model IDs or local paths.

    \b
    Examples:
      modelcheck scan microsoft/phi-2
      modelcheck scan microsoft/phi-2 meta-llama/Llama-2-7b-hf --parallelism 2
      modelcheck scan ./my-local-model
    """
    console = Console(stderr=quiet)
    err_console = Console(stderr=True)

    hf_token = _resolve_token(token)

    # ── Load rules file if provided ───────────────────────────────────────────
    rule_config = None
    if rules_file:
        try:
            from .rule_config import RuleConfig
            rule_config = RuleConfig(rules_file)
            if not quiet:
                console.print(f"[dim]Rules:[/dim] {rule_config.summary()}")
        except Exception as exc:
            err_console.print(f"[yellow]Warning:[/yellow] Could not load rules file: {exc}")

    n = len(models)
    if not quiet:
        if n == 1:
            console.print(
                f"\n[bold]modelcheck[/bold] scanning [cyan]{models[0]}[/cyan]  "
                f"[dim](source: {_detect_source(models[0]).value})[/dim]\n"
            )
        else:
            console.print(
                f"\n[bold]modelcheck[/bold] scanning [bold]{n}[/bold] models  "
                f"[dim](parallelism={min(parallelism, n)})[/dim]\n"
            )

    # ── Scan models (parallel) ────────────────────────────────────────────────
    results: dict[str, SecurityReport | Exception] = {}

    # Suppress HuggingFace Hub's tqdm download bars — they corrupt Rich's live
    # display when multiple threads write to the terminal simultaneously.
    try:
        import huggingface_hub.utils as _hf_utils
        _hf_utils.disable_progress_bars()
    except Exception:
        pass
    try:
        from tqdm import tqdm as _tqdm
        _tqdm_original_init = _tqdm.__init__

        def _tqdm_disabled_init(self, *args, **kwargs):
            kwargs["disable"] = True
            _tqdm_original_init(self, *args, **kwargs)

        _tqdm.__init__ = _tqdm_disabled_init
    except Exception:
        pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=quiet,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        task_ids = {m: progress.add_task(f"[dim]{m}[/dim] — queued", total=None)
                    for m in models}

        with ThreadPoolExecutor(max_workers=min(parallelism, n)) as pool:
            futures = {
                pool.submit(
                    _scan_one, m, hf_token, skip, download, no_probes,
                    rule_config, progress, task_ids[m],
                ): m
                for m in models
            }
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    results[m] = fut.result()
                except Exception as exc:
                    results[m] = exc
                    progress.update(task_ids[m],
                                    description=f"[red]{m}[/red] — error: {exc}")

    # Preserve original ordering
    reports: list[SecurityReport] = []
    for m in models:
        r = results.get(m)
        if isinstance(r, SecurityReport):
            reports.append(r)
        else:
            err_console.print(f"[red]Failed to scan {m}:[/red] {r}")

    if not reports:
        err_console.print("[red]No models scanned successfully. Exiting.[/red]")
        sys.exit(1)

    # ── Render console report ─────────────────────────────────────────────────
    if not quiet:
        for rpt in reports:
            render_report(rpt, console=console, verbose=verbose)

    # ── Export JSON ───────────────────────────────────────────────────────────
    if output:
        if len(reports) == 1:
            export_json(reports[0], output)
            if not quiet:
                console.print(f"[dim]JSON report:[/dim] {output}")
        else:
            out_dir = Path(output)
            out_dir.mkdir(parents=True, exist_ok=True)
            for rpt in reports:
                safe = rpt.model_id.replace("/", "_").replace("\\", "_")
                p = out_dir / f"{safe}.json"
                export_json(rpt, str(p))
            if not quiet:
                console.print(f"[dim]JSON reports written to:[/dim] {out_dir}/")

    # ── Generate HTML report ──────────────────────────────────────────────────
    if not no_html:
        if len(reports) == 1:
            safe_name = reports[0].model_id.replace("/", "_").replace("\\", "_")
            html_path = html_out or f"{safe_name}_security_card.html"
        else:
            html_path = html_out or "modelcheck_report.html"

        try:
            if not quiet:
                with console.status("[dim]Generating HTML report…[/dim]"):
                    written = _write_html(reports, html_path)
            else:
                written = _write_html(reports, html_path)
            if not quiet:
                console.print(f"[dim]HTML report:[/dim] [bold]{written}[/bold]")
        except Exception as exc:
            err_console.print(f"[yellow]Warning:[/yellow] HTML generation failed: {exc}")

    # ── Exit code for CI ──────────────────────────────────────────────────────
    if fail_on != "never":
        threshold_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        threshold = threshold_map[fail_on]
        severity_order = ["critical", "high", "medium", "low"]
        for rpt in reports:
            counts = rpt.total_findings()
            for sev in severity_order[:threshold + 1]:
                if counts.get(sev, 0) > 0:
                    if not quiet:
                        console.print(
                            f"\n[red]Exiting with code 1[/red]: "
                            f"{rpt.model_id} has {counts[sev]} {sev} finding(s) "
                            f"(--fail-on {fail_on})."
                        )
                    sys.exit(1)


def _write_html(reports: list[SecurityReport], html_path: str) -> str:
    if len(reports) == 1:
        from .html_report import generate_html_report
        return generate_html_report(reports[0], html_path)
    else:
        from .html_report import generate_multi_html_report
        return generate_multi_html_report(reports, html_path)


@main.command("list-checks")
def list_checks() -> None:
    """List all available security checks and their IDs."""
    console = Console()
    from rich.table import Table
    from rich import box

    table = Table(title="Available Security Checks", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Module", style="bold")
    table.add_column("Description")

    checks = [
        # Provenance
        ("PROV-001", "Provenance", "HuggingFace security scanner flagged unsafe files"),
        ("PROV-002", "Provenance", "Unknown model author"),
        ("PROV-003", "Provenance", "Country of origin undetermined"),
        ("PROV-004", "Provenance", "No license specified"),
        ("PROV-005", "Provenance", "Restrictive license (commercial restrictions)"),
        ("PROV-006", "Provenance", "Low community adoption (<100 downloads)"),
        ("PROV-007", "Provenance", "Weight files may lack integrity verification"),
        ("PROV-008", "Provenance", "Missing or sparse model card"),
        ("PROV-009", "Provenance", "Gated model access"),
        ("PROV-L001-4", "Provenance", "Local model: missing card, unknown author/country"),
        # Adversarial
        ("ADV-000", "Adversarial", "Adversarial testing tools not installed"),
        ("ADV-MC-001", "Adversarial", "No adversarial robustness evaluation in model card"),
        ("ADV-MC-002", "Adversarial", "No bias/fairness evaluation disclosed"),
        ("ADV-ART-001", "Adversarial", "ART: FGSM/PGD attack success rate (requires ART + torch)"),
        ("ADV-TA-001", "Adversarial", "TextAttack: TextFooler attack success rate (requires textattack)"),
        ("ADV-LLM-001", "Adversarial", "LLM probe: DAN jailbreak"),
        ("ADV-LLM-002", "Adversarial", "LLM probe: Prompt injection via role play"),
        ("ADV-LLM-003", "Adversarial", "LLM probe: System prompt extraction"),
        ("ADV-LLM-004", "Adversarial", "LLM probe: Token smuggling (unicode homoglyphs)"),
        ("ADV-LLM-005", "Adversarial", "LLM probe: Indirect prompt injection"),
        # Privacy
        ("PRIV-PII-*", "Privacy", "PII detected (email, SSN, credit card, API key, etc.)"),
        ("PRIV-010", "Privacy", "High-risk training datasets (web-scraped)"),
        ("PRIV-011", "Privacy", "Training data with consent concerns"),
        ("PRIV-015", "Privacy", "No privacy disclosures in model card"),
        ("PRIV-016", "Privacy", "Elevated membership inference attack risk"),
        ("PRIV-018", "Privacy", "Web-scraped data without regulatory compliance disclosure"),
        ("PRIV-020", "Privacy", "Training data not disclosed"),
        # File Security
        ("FSEC-HF-001", "File Security", "HuggingFace security scanner: unsafe files"),
        ("FSEC-001", "File Security", "Pickle-based weight files present"),
        ("FSEC-002", "File Security", "No SafeTensors alternative available"),
        ("FSEC-MA-*", "File Security", "modelaudit findings"),
        ("FSEC-PKL-001", "File Security", "Unsafe pickle content (dangerous opcodes/imports)"),
        ("FSEC-PKL-002", "File Security", "Non-standard pickle imports"),
        ("FSEC-MS-*", "File Security", "modelscan findings (requires Python <3.13)"),
        ("FSEC-FMT-*", "File Security", "Suspicious file types (.exe, .sh, .dll, etc.)"),
        ("FSEC-EMBED-001", "File Security", "Suspicious patterns in config JSON files"),
    ]

    for check_id, module, desc in checks:
        table.add_row(check_id, module, desc)

    console.print(table)


@main.command("info")
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("--token", "-t", envvar="HF_TOKEN", default=None)
def info(model: str, token: Optional[str]) -> None:
    """Show basic information about a model without running full security scans."""
    console = Console()
    hf_token = _resolve_token(token)
    source = _detect_source(model)

    if source == ModelSource.HUGGINGFACE:
        from .utils.hf_utils import fetch_hf_model_info
        with console.status("Fetching model info..."):
            info_data = fetch_hf_model_info(model, token=hf_token)

        from rich.table import Table
        from rich import box

        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("Field", style="bold cyan", no_wrap=True)
        t.add_column("Value", overflow="fold")

        fields = [
            ("Model ID", model),
            ("Author", info_data.get("author", "Unknown")),
            ("Country of Origin", info_data.get("country_of_origin", "Unknown")),
            ("License", info_data.get("license", "Unknown")),
            ("Pipeline", info_data.get("pipeline_tag", "Unknown")),
            ("Downloads", str(info_data.get("downloads", 0))),
            ("Likes", str(info_data.get("likes", 0))),
            ("Gated", str(info_data.get("gated", False))),
            ("Private", str(info_data.get("private", False))),
            ("Created", info_data.get("created_at", "Unknown")),
            ("Last Modified", info_data.get("last_modified", "Unknown")),
            ("Files", str(len(info_data.get("files", [])))),
            ("Datasets", ", ".join(info_data.get("datasets", [])) or "Not documented"),
        ]
        for field, value in fields:
            t.add_row(field, value)

        console.print(t)
    else:
        console.print(f"[bold]Local model path:[/bold] {model}")
        p = Path(model)
        if p.is_dir():
            files = list(p.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            console.print(f"  Files: {file_count}")
            console.print(f"  Total size: {total_size / 1e9:.2f} GB")
