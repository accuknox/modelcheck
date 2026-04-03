"""
Evaluator 1 — Supply Chain & Provenance

Assesses model origins, authorship, licensing, and supply chain risks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..models import EvaluatorResult, EvaluatorStatus, Finding, PassedCheck, ModelSource, Severity


# Licenses considered permissive / research-safe
_PERMISSIVE_LICENSES = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause",
    "cc-by-4.0", "cc-by-sa-4.0", "openrail", "openrail++",
    "llama2", "llama3", "gemma", "llama3.1", "llama3.2",
}

# Licenses with commercial restrictions (not necessarily a security risk, but worth flagging)
_RESTRICTED_LICENSES = {
    "cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0",
    "gpl-2.0", "gpl-3.0", "agpl-3.0",
    "other", "unknown",
}


def _check_unsigned_weights(files: list[dict[str, Any]]) -> bool:
    """Returns True if no SHA256 checksums / LFS hashes are available for weight files."""
    weight_exts = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".pkl"}
    weight_files = [f for f in files if Path(f["filename"]).suffix.lower() in weight_exts]
    if not weight_files:
        return False
    # If any weight file lacks LFS tracking it may be unsigned
    unsigned = [f for f in weight_files if not f.get("lfs") and not f.get("blob_id")]
    return len(unsigned) > 0


def evaluate_provenance(
    model_id: str,
    source: ModelSource,
    hf_info: dict[str, Any] | None = None,
    local_path: str | None = None,
) -> EvaluatorResult:
    findings: list[Finding] = []
    passed: list[PassedCheck] = []
    metadata: dict[str, Any] = {}

    if source == ModelSource.HUGGINGFACE and hf_info:
        metadata = _evaluate_hf_provenance(model_id, hf_info, findings, passed)
    elif source == ModelSource.LOCAL:
        metadata = _evaluate_local_provenance(model_id, local_path or "", findings, passed)
    else:
        return EvaluatorResult(
            name="Supply Chain & Provenance",
            status=EvaluatorStatus.ERROR,
            summary="No model information available.",
            error="Neither HuggingFace info nor local path provided.",
        )

    if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in findings):
        status = EvaluatorStatus.FAIL
    elif findings:
        status = EvaluatorStatus.WARNING
    else:
        status = EvaluatorStatus.PASS

    critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    high = sum(1 for f in findings if f.severity == Severity.HIGH)
    summary = (
        f"{len(findings)} finding(s): {critical} critical, {high} high. "
        f"Author: {metadata.get('author', 'Unknown')} | "
        f"Country: {metadata.get('country_of_origin', 'Unknown')} | "
        f"License: {metadata.get('license', 'Unknown')}"
    )

    return EvaluatorResult(
        name="Supply Chain & Provenance",
        status=status,
        summary=summary,
        findings=findings,
        passed_checks=passed,
        metadata=metadata,
    )


def _evaluate_hf_provenance(
    model_id: str,
    info: dict[str, Any],
    findings: list[Finding],
    passed: list[PassedCheck],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "author": info.get("author", "Unknown"),
        "country_of_origin": info.get("country_of_origin", "Unknown"),
        "license": info.get("license", "Unknown"),
        "pipeline_tag": info.get("pipeline_tag", "Unknown"),
        "downloads": info.get("downloads", 0),
        "likes": info.get("likes", 0),
        "tags": info.get("tags", []),
        "created_at": info.get("created_at", "Unknown"),
        "last_modified": info.get("last_modified", "Unknown"),
        "datasets": info.get("datasets", []),
        "gated": info.get("gated", "false"),
        "private": info.get("private", False),
        "hf_security_status": info.get("hf_security_status", {}),
    }

    # --- HuggingFace built-in security scan ---
    hf_sec = info.get("hf_security_status") or {}
    if hf_sec.get("has_unsafe_files"):
        findings.append(Finding(
            id="PROV-001",
            title="HuggingFace Security Scan: Unsafe Files Detected",
            severity=Severity.CRITICAL,
            description=(
                "HuggingFace's own security scanner has flagged this repository as containing "
                "unsafe files. This indicates high-confidence malicious content."
            ),
            remediation="Do not load this model. Report the repository to HuggingFace.",
            references=["https://huggingface.co/docs/hub/security-pickle"],
        ))
    elif hf_sec:
        passed.append(PassedCheck(id="PROV-001", title="HuggingFace Security Scan",
                                  detail="No unsafe files detected by HuggingFace scanner"))

    # --- Unknown / anonymous author ---
    author = info.get("author", "Unknown")
    if not author or author.lower() in ("unknown", ""):
        findings.append(Finding(
            id="PROV-002",
            title="Unknown Model Author",
            severity=Severity.MEDIUM,
            description="The model has no identifiable author or organization on HuggingFace Hub.",
            remediation="Verify the model's origin before use. Prefer models from known organizations.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-002", title="Model Author Identified",
                                  detail=f"Author: {author}"))

    # --- Country of origin ---
    country = info.get("country_of_origin", "Unknown")
    if country == "Unknown":
        findings.append(Finding(
            id="PROV-003",
            title="Country of Origin Undetermined",
            severity=Severity.LOW,
            description="Could not determine the country of origin for this model.",
            remediation="Manually investigate the author/organization's background.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-003", title="Country of Origin Determined",
                                  detail=f"Country: {country}"))

    # --- License ---
    license_val = (info.get("license") or "unknown").lower()
    if license_val in ("unknown", ""):
        findings.append(Finding(
            id="PROV-004",
            title="No License Specified",
            severity=Severity.MEDIUM,
            description=(
                "No license is declared for this model. Legal and compliance risks are unclear. "
                "The model may not be suitable for commercial or production use."
            ),
            remediation="Contact the author to clarify licensing terms.",
        ))
    elif license_val in _RESTRICTED_LICENSES:
        findings.append(Finding(
            id="PROV-005",
            title=f"Restrictive License: {license_val}",
            severity=Severity.INFO,
            description=f"The model uses license '{license_val}' which may restrict commercial use.",
            remediation="Review license terms before deployment.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-004", title="License Declared",
                                  detail=f"License: {license_val}"))
        passed.append(PassedCheck(id="PROV-005", title="Permissive License",
                                  detail=f"{license_val} permits commercial and research use"))

    # --- Low download / new model with no community validation ---
    downloads = info.get("downloads", 0) or 0
    likes = info.get("likes", 0) or 0
    if downloads < 100 and likes < 5:
        findings.append(Finding(
            id="PROV-006",
            title="Low Community Adoption",
            severity=Severity.LOW,
            description=(
                f"Model has only {downloads} downloads and {likes} likes. "
                "Unvetted models from low-profile accounts carry higher supply chain risk."
            ),
            remediation="Prefer models with substantial community vetting and usage history.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-006", title="Adequate Community Adoption",
                                  detail=f"{downloads:,} downloads · {likes} likes"))

    # --- Unsigned / unverified weights ---
    files = info.get("files", [])
    if files and _check_unsigned_weights(files):
        findings.append(Finding(
            id="PROV-007",
            title="Weight Files May Lack Integrity Verification",
            severity=Severity.LOW,
            description=(
                "Some model weight files do not appear to be tracked via LFS with verifiable "
                "checksums, making it harder to verify file integrity."
            ),
            remediation="Verify file hashes against published checksums if available.",
        ))
    elif files:
        passed.append(PassedCheck(id="PROV-007", title="Weight File Integrity Verifiable",
                                  detail="All weight files are LFS-tracked with checksums"))

    # --- No model card ---
    card = info.get("model_card", "")
    if not card or len(card.strip()) < 100:
        findings.append(Finding(
            id="PROV-008",
            title="Missing or Sparse Model Card",
            severity=Severity.LOW,
            description=(
                "The model has no meaningful model card. This makes it impossible to assess "
                "training methodology, intended use, limitations, and biases."
            ),
            remediation="Request the author to provide a comprehensive model card.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-008", title="Model Card Present",
                                  detail=f"Model card with {len(card.strip()):,} characters"))

    # --- Gated model that is now public ---
    gated = str(info.get("gated", "false")).lower()
    if gated not in ("false", ""):
        findings.append(Finding(
            id="PROV-009",
            title=f"Gated Model Access (gated={gated})",
            severity=Severity.INFO,
            description="This model uses HuggingFace gated access, requiring agreement to usage terms.",
            remediation="Review and accept the model's usage terms before deployment.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-009", title="Publicly Accessible",
                                  detail="Model has no gating restrictions"))

    # --- Organizational backing / author credibility ---
    _check_author_credibility(info, findings, passed, metadata)

    return metadata


def _check_author_credibility(
    info: dict[str, Any],
    findings: list[Finding],
    passed: list[PassedCheck],
    metadata: dict[str, Any],
) -> None:
    """
    PROV-010 — flags models from low-profile individuals or unvetted organisations.

    Thresholds for individual users (by HF follower count):
      < 100   → HIGH   (unknown individual, minimal community vetting)
      100–999 → MEDIUM (known but niche individual)
      1 000+  → INFO   (well-established individual)

    Unknown organisations (not in WELL_KNOWN_ORGS) → MEDIUM.
    Well-known organisations → no finding.
    """
    profile = info.get("author_profile") or {}
    author_type   = profile.get("author_type", "unknown")
    num_followers = profile.get("num_followers", 0) or 0
    num_models    = profile.get("num_models", 0) or 0
    author        = info.get("author", "Unknown")

    # Store in metadata for the HTML report
    metadata["author_type"]   = author_type
    metadata["author_followers"] = num_followers

    if author_type == "organization_known":
        passed.append(PassedCheck(id="PROV-010", title="Author Credibility: Trusted Organisation",
                                  detail=f"{author} is a well-known AI organisation"))
        return

    if author_type == "individual":
        if num_followers < 100:
            findings.append(Finding(
                id="PROV-010",
                title=f"Low-Profile Individual Author ({num_followers} followers)",
                severity=Severity.HIGH,
                description=(
                    f"The model is published by an individual HuggingFace user "
                    f"'{author}' with only {num_followers} followers and {num_models} "
                    f"public model(s). Low-profile individuals have not been vetted by "
                    f"the community and present a higher supply-chain risk."
                ),
                remediation=(
                    "Prefer models from well-known organisations or individuals with a "
                    "substantial community presence (1 000+ followers). "
                    "Inspect the model files carefully before use."
                ),
                details={
                    "author": author,
                    "author_type": "individual",
                    "num_followers": num_followers,
                    "num_models": num_models,
                },
                references=["https://huggingface.co/" + author],
            ))
        elif num_followers < 1_000:
            findings.append(Finding(
                id="PROV-010",
                title=f"Individual Author with Limited Following ({num_followers} followers)",
                severity=Severity.MEDIUM,
                description=(
                    f"'{author}' is an individual user with {num_followers} followers. "
                    f"While not anonymous, this author has not reached broad community "
                    f"recognition. Models from less-followed individuals carry moderate "
                    f"supply-chain risk."
                ),
                remediation=(
                    "Review the model card and file contents carefully. "
                    "Consider whether a well-known organisation publishes an equivalent model."
                ),
                details={
                    "author": author,
                    "author_type": "individual",
                    "num_followers": num_followers,
                    "num_models": num_models,
                },
                references=["https://huggingface.co/" + author],
            ))
        else:
            # Well-established individual — low risk, informational only
            findings.append(Finding(
                id="PROV-010",
                title=f"Individual Author (well-established, {num_followers:,} followers)",
                severity=Severity.INFO,
                description=(
                    f"'{author}' is an individual user with {num_followers:,} followers "
                    f"and {num_models} public model(s). This level of community presence "
                    f"suggests a degree of public vetting, but organisational backing "
                    f"is absent."
                ),
                remediation="No immediate action required. Standard model vetting applies.",
                details={
                    "author": author,
                    "author_type": "individual",
                    "num_followers": num_followers,
                    "num_models": num_models,
                },
                references=["https://huggingface.co/" + author],
            ))
        return

    if author_type == "organization_unknown":
        findings.append(Finding(
            id="PROV-010",
            title=f"Unvetted Organisation: '{author}'",
            severity=Severity.MEDIUM,
            description=(
                f"'{author}' is not among the curated set of well-known AI organisations. "
                f"Smaller or newer organisations may not have undergone independent security "
                f"audits and present a moderate supply-chain risk."
            ),
            remediation=(
                "Research the organisation's background, funding, and track record before "
                "deploying this model in production."
            ),
            details={"author": author, "author_type": "organization_unknown"},
            references=["https://huggingface.co/" + author],
        ))
        return

    # author_type == "unknown" — already handled by PROV-002


def _evaluate_local_provenance(
    model_id: str,
    local_path: str,
    findings: list[Finding],
    passed: list[PassedCheck],
) -> dict[str, Any]:
    p = Path(local_path)
    metadata: dict[str, Any] = {
        "author": "Unknown",
        "country_of_origin": "Unknown",
        "license": "Unknown",
        "local_path": local_path,
    }

    if not p.exists():
        findings.append(Finding(
            id="PROV-L001",
            title="Model Path Does Not Exist",
            severity=Severity.CRITICAL,
            description=f"The specified path '{local_path}' does not exist.",
            remediation="Verify the model path and try again.",
        ))
        return metadata

    # Try to read a model card if present
    card_paths = [p / "README.md", p / "MODEL_CARD.md", p / "model_card.md"]
    card_content = ""
    for cp in card_paths:
        if cp.exists():
            card_content = cp.read_text(errors="replace")
            break

    # Check for config.json to extract model metadata
    config_path = p / "config.json"
    if config_path.exists():
        import json
        try:
            config = json.loads(config_path.read_text())
            arch = config.get("architectures", [])
            model_type = config.get("model_type", "Unknown")
            metadata["model_type"] = model_type
            metadata["architectures"] = arch
        except Exception:
            pass

    if not card_content:
        findings.append(Finding(
            id="PROV-L002",
            title="No Model Card Found Locally",
            severity=Severity.MEDIUM,
            description="No README.md or model card found in the model directory.",
            remediation="Add a model card documenting the model's origin, training data, and intended use.",
        ))
    else:
        metadata["author"] = _extract_author_from_card(card_content) or "Unknown"
        metadata["country_of_origin"] = _infer_country_from_card(card_content)
        metadata["license"] = _extract_license_from_card(card_content) or "Unknown"
        passed.append(PassedCheck(id="PROV-L002", title="Model Card Present",
                                  detail="README / model card found locally"))

    # Always flag unknown provenance for local models
    if metadata["author"] == "Unknown":
        findings.append(Finding(
            id="PROV-L003",
            title="Unknown Local Model Author",
            severity=Severity.MEDIUM,
            description="Cannot determine the author of this locally stored model.",
            remediation="Document the model's provenance in a README.md or model card.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-L003", title="Local Model Author Identified",
                                  detail=f"Author: {metadata['author']}"))

    if metadata["country_of_origin"] == "Unknown":
        findings.append(Finding(
            id="PROV-L004",
            title="Country of Origin Undetermined",
            severity=Severity.LOW,
            description="Cannot determine the country of origin for this local model.",
            remediation="Document the model's origin in a model card.",
        ))
    else:
        passed.append(PassedCheck(id="PROV-L004", title="Country of Origin Determined",
                                  detail=f"Country: {metadata['country_of_origin']}"))

    return metadata


def _extract_author_from_card(text: str) -> str:
    patterns = [
        r"(?:author|developed by|created by|by)[:\s]+([^\n]+)",
        r"(?:organization|org)[:\s]+([^\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("*_`")
    return ""


def _extract_license_from_card(text: str) -> str:
    m = re.search(r"license[:\s]+([^\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("*_`")
    return ""


def _infer_country_from_card(text: str) -> str:
    from ..utils.hf_utils import _infer_country
    return _infer_country("", text)
