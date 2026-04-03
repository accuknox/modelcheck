"""
Evaluator 4 — Model File Security

Static scanning using:
  - modelaudit (installed): comprehensive ML file security scanner
  - Built-in pickle/serialization scanner (modelscan-equivalent for Python 3.13+)
  - Additional checks: file integrity, suspicious content, format validation
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..models import EvaluatorResult, EvaluatorStatus, Finding, PassedCheck, ModelSource, Severity
from ..utils.pickle_scanner import scan_path, PickleScanResult


def evaluate_file_security(
    model_id: str,
    source: ModelSource,
    hf_info: dict[str, Any] | None = None,
    local_path: str | None = None,
    download_for_scan: bool = False,
    hf_token: str | None = None,
) -> EvaluatorResult:
    findings: list[Finding] = []
    passed: list[PassedCheck] = []
    metadata: dict[str, Any] = {"model_id": model_id, "scanners_used": []}

    scan_path_str: str | None = local_path

    # For HF models we can scan without downloading if we have file listing
    if source == ModelSource.HUGGINGFACE and hf_info and not scan_path_str:
        hf_meta = _evaluate_hf_file_metadata(hf_info, findings, passed)
        metadata.update(hf_meta)

        if download_for_scan:
            scan_path_str = _download_hf_model(model_id, hf_token, findings)
        else:
            _add_no_local_scan_finding(findings)

    if scan_path_str and Path(scan_path_str).exists():
        metadata["scan_path"] = scan_path_str

        # --- modelaudit scan ---
        pre_count = len(findings)
        modelaudit_meta = _run_modelaudit(scan_path_str, findings)
        metadata["modelaudit"] = modelaudit_meta
        metadata["scanners_used"].append("modelaudit")
        if len(findings) == pre_count:
            passed.append(PassedCheck(id="FSEC-MA", title="modelaudit: No Issues Found",
                                      detail="modelaudit static scan reported no security issues"))

        # --- Pickle / serialization scanner ---
        pre_count = len(findings)
        pickle_meta = _run_pickle_scanner(scan_path_str, findings)
        metadata["pickle_scanner"] = pickle_meta
        metadata["scanners_used"].append("pickle_scanner")
        if len(findings) == pre_count:
            passed.append(PassedCheck(id="FSEC-PKL", title="Pickle Scanner: No Unsafe Content",
                                      detail="Built-in pickle scanner found no dangerous opcodes"))

        # --- modelscan (via subprocess if available) ---
        modelscan_meta = _run_modelscan_subprocess(scan_path_str, findings)
        if modelscan_meta:
            metadata["modelscan"] = modelscan_meta
            metadata["scanners_used"].append("modelscan")

        # --- Additional static checks ---
        pre_count = len(findings)
        _check_file_formats(scan_path_str, hf_info, findings, metadata)
        if len(findings) == pre_count:
            passed.append(PassedCheck(id="FSEC-FMT", title="File Formats: No Suspicious Types",
                                      detail="No executables, scripts, or suspicious file types found"))

        pre_count = len(findings)
        _check_for_embedded_scripts(scan_path_str, findings)
        if len(findings) == pre_count:
            passed.append(PassedCheck(id="FSEC-EMBED", title="Config Files: No Embedded Scripts",
                                      detail="No suspicious patterns detected in JSON config files"))

    if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in findings):
        status = EvaluatorStatus.FAIL
    elif any(f.severity == Severity.MEDIUM for f in findings):
        status = EvaluatorStatus.WARNING
    elif findings:
        status = EvaluatorStatus.WARNING
    else:
        status = EvaluatorStatus.PASS

    scanners = ", ".join(metadata["scanners_used"]) or "metadata-only"
    critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    high = sum(1 for f in findings if f.severity == Severity.HIGH)
    summary = (
        f"{len(findings)} finding(s): {critical} critical, {high} high. "
        f"Scanners: {scanners}."
    )

    return EvaluatorResult(
        name="Model File Security",
        status=status,
        summary=summary,
        findings=findings,
        passed_checks=passed,
        metadata=metadata,
    )


def _evaluate_hf_file_metadata(
    hf_info: dict[str, Any],
    findings: list[Finding],
    passed: list[PassedCheck],
) -> dict[str, Any]:
    """Evaluate file metadata from HuggingFace without downloading."""
    meta: dict[str, Any] = {}
    files = hf_info.get("files", [])

    if not files:
        return meta

    # Check HF security scan result
    hf_sec = hf_info.get("hf_security_status") or {}
    if hf_sec.get("has_unsafe_files"):
        findings.append(Finding(
            id="FSEC-HF-001",
            title="HuggingFace Security Scanner: Unsafe Files Detected",
            severity=Severity.CRITICAL,
            description=(
                "HuggingFace's built-in security scanner has flagged this repository. "
                "One or more files contain potentially malicious content (e.g., unsafe pickle opcodes)."
            ),
            remediation="Do not load or use this model. Report to HuggingFace Trust & Safety.",
            references=["https://huggingface.co/docs/hub/security-pickle"],
        ))
    elif hf_sec:
        passed.append(PassedCheck(id="FSEC-HF-001", title="HuggingFace Security Scan: Clean",
                                  detail="HuggingFace scanner found no unsafe files"))

    # File format analysis
    pickle_files = []
    safetensor_files = []
    gguf_files = []
    unknown_files = []

    ext_map = {
        ".safetensors": safetensor_files,
        ".gguf": gguf_files,
        ".ggml": gguf_files,
    }
    pickle_exts = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".ckpt", ".joblib"}

    for f in files:
        fname = f["filename"]
        suffix = Path(fname).suffix.lower()
        if suffix in pickle_exts:
            pickle_files.append(fname)
        elif suffix in ext_map:
            ext_map[suffix].append(fname)
        elif suffix not in {".json", ".txt", ".md", ".yaml", ".yml", ".py", ".h5", ".npz", ".npy"}:
            unknown_files.append(fname)

    meta["file_formats"] = {
        "pickle_based": len(pickle_files),
        "safetensors": len(safetensor_files),
        "gguf": len(gguf_files),
        "other": len(unknown_files),
    }

    if pickle_files:
        findings.append(Finding(
            id="FSEC-001",
            title=f"Pickle-Based Weight Files Present ({len(pickle_files)} file(s))",
            severity=Severity.MEDIUM,
            description=(
                f"The repository contains {len(pickle_files)} pickle-based weight file(s): "
                f"{', '.join(pickle_files[:5])}{'...' if len(pickle_files) > 5 else ''}.\n\n"
                "Pickle deserialization can execute arbitrary code. Even without explicit malice, "
                "pickle files present an attack surface that safetensors files do not."
            ),
            remediation=(
                "Prefer repositories that provide .safetensors format. "
                "If using pickle files, scan with modelscan before loading."
            ),
            details={"pickle_files": pickle_files[:10]},
            references=[
                "https://huggingface.co/docs/safetensors/index",
                "https://github.com/protectai/modelscan",
            ],
        ))
    else:
        passed.append(PassedCheck(id="FSEC-001", title="No Pickle-Based Weight Files",
                                  detail="Repository does not contain .pkl/.pt/.bin pickle-format weights"))

    if not safetensor_files and not gguf_files and pickle_files:
        findings.append(Finding(
            id="FSEC-002",
            title="No SafeTensors Alternative Available",
            severity=Severity.LOW,
            description=(
                "The repository provides only pickle-based weights with no SafeTensors alternative. "
                "SafeTensors is a safer format that does not allow arbitrary code execution during loading."
            ),
            remediation="Request or convert to SafeTensors format before use.",
            references=["https://github.com/huggingface/safetensors"],
        ))
    elif safetensor_files or gguf_files:
        fmt = "SafeTensors" if safetensor_files else "GGUF"
        passed.append(PassedCheck(id="FSEC-002", title=f"Safe Format Available ({fmt})",
                                  detail=f"{len(safetensor_files + gguf_files)} safe-format file(s) present"))

    return meta


def _add_no_local_scan_finding(findings: list[Finding]) -> None:
    findings.append(Finding(
        id="FSEC-INFO-001",
        title="Deep File Scan Skipped (No Local Copy)",
        severity=Severity.INFO,
        description=(
            "The model files were not downloaded for deep scanning. "
            "Only HuggingFace metadata was analyzed."
        ),
        remediation="Re-run with --download flag to perform full static analysis on model files.",
    ))


def _run_modelaudit(scan_path_str: str, findings: list[Finding]) -> dict[str, Any]:
    """Run modelaudit static scan using its native Python API."""
    meta: dict[str, Any] = {"ran": False}
    try:
        from modelaudit.core import scan_model_directory_or_file
        from modelaudit.models import ModelAuditResultModel
        from modelaudit.scanners.base import Issue, Check, CheckStatus, IssueSeverity

        result: ModelAuditResultModel = scan_model_directory_or_file(scan_path_str)
        meta["ran"] = True
        meta["files_scanned"] = result.files_scanned
        meta["bytes_scanned"] = result.bytes_scanned
        meta["total_checks"] = result.total_checks
        meta["passed_checks"] = result.passed_checks
        meta["failed_checks"] = result.failed_checks
        meta["has_errors"] = result.has_errors
        meta["scanners_used"] = result.scanner_names
        meta["duration_s"] = round(result.duration, 2)
        meta["assets"] = [
            {"path": a.path, "type": a.type, "size": a.size}
            for a in (result.assets or [])
        ]

        # modelaudit IssueSeverity → our Severity
        # CRITICAL → CRITICAL, WARNING → HIGH, INFO → MEDIUM, DEBUG → INFO
        _sev = {
            IssueSeverity.CRITICAL: Severity.CRITICAL,
            IssueSeverity.WARNING:  Severity.HIGH,
            IssueSeverity.INFO:     Severity.MEDIUM,
            IssueSeverity.DEBUG:    Severity.INFO,
        }

        # ── Issues (positive security findings) ───────────────────────────
        for issue in (result.issues or []):
            severity = _sev.get(issue.severity, Severity.MEDIUM)
            # Skip pure debug/format-detection noise unless it's a real issue
            if issue.severity == IssueSeverity.DEBUG and not issue.why:
                continue

            loc = issue.location or ""
            why = issue.why or ""
            desc_parts = [issue.message]
            if loc:
                desc_parts.append(f"Location: {loc}")
            if why:
                desc_parts.append(f"Why: {why}")
            if issue.details:
                relevant = {k: v for k, v in issue.details.items()
                            if k not in ("raw",) and v not in (None, "", [], {})}
                if relevant:
                    desc_parts.append(f"Details: {relevant}")

            issue_type = (issue.type or "UNK").upper().replace(" ", "_")[:30]
            findings.append(Finding(
                id=f"FSEC-MA-{issue_type}",
                title=f"[modelaudit] {issue.message[:100]}",
                severity=severity,
                description="\n".join(desc_parts),
                remediation="Review modelaudit documentation for remediation guidance.",
                details={
                    "scanner": "modelaudit",
                    "location": loc,
                    "type": issue.type,
                    "raw_details": issue.details,
                },
                references=["https://github.com/accuknox/modelaudit"],
            ))

        # ── Failed checks (structural/policy violations) ──────────────────
        seen_check_names: set[str] = set()
        for check in (result.checks or []):
            if check.status != CheckStatus.FAILED:
                continue
            # Skip pure debug-severity format detection noise
            if check.severity == IssueSeverity.DEBUG and "unknown" in check.message.lower():
                continue
            # Deduplicate repeated check names (same check failing across many files)
            if check.name in seen_check_names:
                continue
            seen_check_names.add(check.name)

            severity = _sev.get(check.severity, Severity.LOW) if check.severity else Severity.LOW
            loc = check.location or ""
            why = check.why or ""
            desc_parts = [check.message]
            if loc:
                desc_parts.append(f"Location: {loc}")
            if why:
                desc_parts.append(f"Why: {why}")
            if check.details:
                relevant = {k: v for k, v in check.details.items()
                            if k not in ("findings",) and v not in (None, "", [], {})}
                if relevant:
                    desc_parts.append(f"Details: {relevant}")

            check_id = check.name.upper().replace(" ", "_")[:30]
            findings.append(Finding(
                id=f"FSEC-MA-CHK-{check_id}",
                title=f"[modelaudit] {check.name} failed",
                severity=severity,
                description="\n".join(desc_parts),
                details={"scanner": "modelaudit", "check": check.name, "location": loc},
                references=["https://github.com/accuknox/modelaudit"],
            ))

    except ImportError:
        findings.append(Finding(
            id="FSEC-MA-MISSING",
            title="modelaudit Not Available",
            severity=Severity.INFO,
            description="modelaudit package is not installed. Skipping modelaudit scan.",
            remediation="Install with: pip install modelaudit",
            references=["https://github.com/accuknox/modelaudit"],
        ))
    except Exception as exc:
        meta["error"] = str(exc)
        findings.append(Finding(
            id="FSEC-MA-ERR",
            title="modelaudit Scan Error",
            severity=Severity.INFO,
            description=f"modelaudit scan failed: {exc}",
        ))

    return meta


def _run_pickle_scanner(scan_path_str: str, findings: list[Finding]) -> dict[str, Any]:
    """Run native pickle scanner."""
    meta: dict[str, Any] = {"ran": True}
    try:
        results: list[PickleScanResult] = scan_path(scan_path_str)
        meta["files_scanned"] = len(results)
        meta["unsafe_files"] = sum(1 for r in results if not r.is_safe)

        for r in results:
            if not r.is_safe or r.suspicious_imports:
                sev = Severity.CRITICAL if r.suspicious_imports else Severity.HIGH
                findings.append(Finding(
                    id="FSEC-PKL-001",
                    title=f"Unsafe Pickle Content: {Path(r.filepath).name}",
                    severity=sev,
                    description=(
                        f"File: {r.filepath}\n"
                        f"Format: {r.file_format}\n"
                        f"Dangerous opcodes: {', '.join(r.dangerous_opcodes)}\n"
                        f"Suspicious imports: {r.suspicious_imports}\n"
                        f"All globals imported: {r.global_imports[:10]}"
                        + (f"\nError: {r.error}" if r.error else "")
                    ),
                    remediation=(
                        "Do not load this file. Convert to SafeTensors format if possible. "
                        "If the model is from HuggingFace, report it as potentially malicious."
                    ),
                    details={
                        "filepath": r.filepath,
                        "dangerous_opcodes": r.dangerous_opcodes,
                        "suspicious_imports": [list(x) for x in r.suspicious_imports],
                        "global_imports": [list(x) for x in r.global_imports[:10]],
                    },
                    references=[
                        "https://github.com/protectai/modelscan",
                        "https://docs.python.org/3/library/pickle.html#restricting-globals",
                    ],
                ))
            elif r.global_imports:
                # Flag non-suspicious but non-trivial global imports for awareness
                non_trivial = [
                    g for g in r.global_imports
                    if g[0] not in (
                        "torch", "_codecs", "collections", "numpy", "storage",
                        "torch.storage", "torch._utils", "collections.abc",
                        "_torch_saved_objects", "torch.nn.modules",
                    )
                ]
                if non_trivial:
                    findings.append(Finding(
                        id="FSEC-PKL-002",
                        title=f"Non-Standard Pickle Imports: {Path(r.filepath).name}",
                        severity=Severity.LOW,
                        description=(
                            f"File contains pickle GLOBAL opcodes importing non-standard modules:\n"
                            f"{non_trivial[:10]}"
                        ),
                        remediation="Review these imports to confirm they are expected for this model type.",
                        details={"non_trivial_imports": [list(x) for x in non_trivial[:10]]},
                    ))

    except Exception as exc:
        meta["error"] = str(exc)
        meta["ran"] = False

    return meta


def _run_modelscan_subprocess(scan_path_str: str, findings: list[Finding]) -> dict[str, Any] | None:
    """
    Try to run modelscan via subprocess (if installed in a different Python environment).
    modelscan does not support Python 3.13+ so we look for it in common venv locations.
    """
    # Look for modelscan in PATH or common venv locations
    import shutil
    modelscan_bin = shutil.which("modelscan")
    if not modelscan_bin:
        # Try common virtual environment paths
        for candidate in [
            "/usr/local/bin/modelscan",
            os.path.expanduser("~/.local/bin/modelscan"),
            os.path.expanduser("~/venv/bin/modelscan"),
            "/opt/homebrew/bin/modelscan",
        ]:
            if os.path.exists(candidate):
                modelscan_bin = candidate
                break

    if not modelscan_bin:
        findings.append(Finding(
            id="FSEC-MS-MISSING",
            title="modelscan Not Available",
            severity=Severity.INFO,
            description=(
                "modelscan (ProtectAI) is not installed or not in PATH. "
                "Note: modelscan requires Python <3.13."
            ),
            remediation=(
                "Install modelscan in a Python 3.10/3.11/3.12 environment:\n"
                "  python3.12 -m pip install modelscan\n"
                "  modelscan scan -p <model_path>\n"
                "Or use: pipx install --python python3.12 modelscan"
            ),
            references=["https://github.com/protectai/modelscan"],
        ))
        return None

    meta: dict[str, Any] = {"binary": modelscan_bin, "ran": False}
    try:
        result = subprocess.run(
            [modelscan_bin, "scan", "-p", scan_path_str, "--show-skipped", "-r", "json"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        meta["ran"] = True
        meta["returncode"] = result.returncode

        # Parse JSON output
        try:
            output = json.loads(result.stdout)
            meta["summary"] = output.get("summary", {})
            issues = output.get("issues", [])
            meta["issue_count"] = len(issues)

            for issue in issues:
                sev_str = str(issue.get("severity", "medium")).lower()
                sev_map = {"critical": Severity.CRITICAL, "high": Severity.HIGH,
                           "medium": Severity.MEDIUM, "low": Severity.LOW}
                findings.append(Finding(
                    id=f"FSEC-MS-{issue.get('code', 'UNK')}",
                    title=f"[modelscan] {issue.get('name', 'Issue')}",
                    severity=sev_map.get(sev_str, Severity.MEDIUM),
                    description=issue.get("description", str(issue)),
                    remediation=issue.get("remediation", ""),
                    details={"scanner": "modelscan", "raw": issue},
                    references=["https://github.com/protectai/modelscan"],
                ))
        except json.JSONDecodeError:
            # Fallback: parse text output
            meta["raw_output"] = result.stdout[:2000]
            if "CRITICAL" in result.stdout or "unsafe" in result.stdout.lower():
                findings.append(Finding(
                    id="FSEC-MS-TEXT",
                    title="[modelscan] Issues Detected (text output)",
                    severity=Severity.HIGH,
                    description=f"modelscan reported issues:\n{result.stdout[:1000]}",
                    references=["https://github.com/protectai/modelscan"],
                ))

    except subprocess.TimeoutExpired:
        meta["error"] = "modelscan timed out (300s)"
    except Exception as exc:
        meta["error"] = str(exc)

    return meta


def _check_file_formats(
    scan_path_str: str,
    hf_info: dict[str, Any] | None,
    findings: list[Finding],
    metadata: dict[str, Any],
) -> None:
    """Analyze file format distribution and suspicious extensions."""
    p = Path(scan_path_str)
    if not p.is_dir():
        return

    file_types: dict[str, list[str]] = {}
    suspicious_exts = {".exe", ".dll", ".so", ".dylib", ".sh", ".bat", ".ps1", ".vbs", ".js"}

    for f in p.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            file_types.setdefault(ext, []).append(f.name)
            if ext in suspicious_exts:
                findings.append(Finding(
                    id=f"FSEC-FMT-{ext[1:].upper()}",
                    title=f"Suspicious File Type in Model Repository: {f.name}",
                    severity=Severity.HIGH,
                    description=(
                        f"Found a file with suspicious extension '{ext}' in the model directory: "
                        f"{f.relative_to(p)}. Executable or script files should not be "
                        "present in model repositories."
                    ),
                    remediation="Remove unexpected executable files before loading the model.",
                    details={"file": str(f.relative_to(p)), "extension": ext},
                ))

    metadata["file_type_distribution"] = {k: len(v) for k, v in file_types.items()}


def _check_for_embedded_scripts(scan_path_str: str, findings: list[Finding]) -> None:
    """Look for embedded shell commands or suspicious strings in JSON configs."""
    p = Path(scan_path_str)
    if not p.is_dir():
        return

    shell_patterns = [
        b"os.system(", b"subprocess.call(", b"subprocess.Popen(",
        b"eval(", b"exec(", b"__import__(",
        b"import socket", b"import base64",
        b"curl ", b"wget ", b"/bin/sh", b"/bin/bash",
        b"powershell", b"cmd.exe",
    ]

    for json_file in p.rglob("*.json"):
        if json_file.stat().st_size > 5_000_000:
            continue
        try:
            content = json_file.read_bytes()
            matched = [p.decode() for p in shell_patterns if p in content]
            if matched:
                findings.append(Finding(
                    id="FSEC-EMBED-001",
                    title=f"Suspicious Patterns in Config File: {json_file.name}",
                    severity=Severity.HIGH,
                    description=(
                        f"Found potentially malicious patterns in {json_file.name}:\n"
                        f"Patterns: {', '.join(matched)}"
                    ),
                    remediation="Review this file carefully before loading the model.",
                    details={"file": str(json_file.name), "patterns": matched},
                ))
        except Exception:
            continue


def _download_hf_model(
    model_id: str,
    hf_token: str | None,
    findings: list[Finding],
) -> str | None:
    """Download model to a temp directory for scanning."""
    try:
        from ..utils.hf_utils import download_model_locally
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="modelcheck_scan_")
        path = download_model_locally(model_id, tmp_dir, hf_token)
        return path
    except Exception as exc:
        findings.append(Finding(
            id="FSEC-DL-ERR",
            title="Model Download Failed",
            severity=Severity.INFO,
            description=f"Could not download model for scanning: {exc}",
            remediation="Check your internet connection and HuggingFace token.",
        ))
        return None
