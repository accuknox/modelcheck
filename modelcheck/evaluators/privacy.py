"""
Evaluator 3 — Data & Privacy Risks

Assesses training data privacy, PII exposure risk, membership inference
susceptibility, and regulatory compliance (GDPR, CCPA).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..models import EvaluatorResult, EvaluatorStatus, Finding, PassedCheck, ModelSource, Severity


# Regex patterns for PII that might appear in model cards, configs, or outputs
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1\s?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
    "api_key": re.compile(r"(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token)['\"]?\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "ip_address": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
}

# High-risk training datasets (known to contain PII or sensitive data)
_HIGH_RISK_DATASETS: set[str] = {
    "cc100", "c4", "the_pile", "pile", "books3", "openwebtext",
    "common_crawl", "laion", "laion-2b", "laion-5b",
    "redpajama", "dolma", "refinedweb", "oscar", "roots",
}

# Datasets with consent / privacy concerns
_CONSENT_CONCERN_DATASETS: set[str] = {
    "laion-2b", "laion-5b", "books3", "pile-of-law",
    "github-code", "stack", "thepile",
}


def evaluate_privacy(
    model_id: str,
    source: ModelSource,
    hf_info: dict[str, Any] | None = None,
    local_path: str | None = None,
) -> EvaluatorResult:
    findings: list[Finding] = []
    passed: list[PassedCheck] = []
    metadata: dict[str, Any] = {"model_id": model_id}

    card_text = ""
    datasets_used: list[str] = []
    pipeline_tag = "unknown"

    if hf_info:
        card_text = hf_info.get("model_card", "")
        datasets_used = hf_info.get("datasets", []) or []
        pipeline_tag = hf_info.get("pipeline_tag", "unknown") or "unknown"
        card_meta = hf_info.get("model_card_metadata", {}) or {}
        for d in (card_meta.get("datasets") or []):
            if isinstance(d, str) and d not in datasets_used:
                datasets_used.append(d)

    elif local_path:
        card_text = _read_local_card(local_path)
        datasets_used = _extract_datasets_from_card(card_text)

    metadata["pipeline_tag"] = pipeline_tag
    metadata["datasets_mentioned"] = datasets_used

    # --- PII scan across card text and config files ---
    pii_findings = _scan_for_pii(card_text, "model_card")
    findings.extend(pii_findings)
    if not pii_findings and card_text:
        passed.append(PassedCheck(id="PRIV-PII-CARD", title="No PII in Model Card",
                                  detail="No personal information detected in model card text"))

    if local_path:
        local_pii = _scan_local_files_for_pii(local_path)
        findings.extend(local_pii)
        if not local_pii:
            passed.append(PassedCheck(id="PRIV-PII-LOCAL", title="No PII in Local Files",
                                      detail="No personal information detected in local model files"))

    # --- Training data privacy analysis ---
    pre_count = len(findings)
    _analyze_training_data(datasets_used, findings, metadata)
    if len(findings) == pre_count and datasets_used:
        passed.append(PassedCheck(id="PRIV-010", title="No High-Risk Training Datasets",
                                  detail=f"Datasets used: {', '.join(datasets_used[:5])}"))
        passed.append(PassedCheck(id="PRIV-011", title="No Consent-Concern Datasets",
                                  detail="Training datasets have no known consent issues"))

    # --- Privacy disclosures in model card ---
    pre_count = len(findings)
    _check_privacy_disclosures(card_text, findings)
    if len(findings) == pre_count and card_text:
        passed.append(PassedCheck(id="PRIV-015", title="Privacy Disclosures Present",
                                  detail="Model card includes privacy/data protection information"))

    # --- Membership inference risk ---
    pre_count = len(findings)
    _assess_membership_inference_risk(pipeline_tag, card_text, datasets_used, findings, metadata)
    if len(findings) == pre_count:
        passed.append(PassedCheck(id="PRIV-016", title="Low Membership Inference Risk",
                                  detail=f"Risk score: {metadata.get('membership_inference_risk_score', 0)}/6"))

    # --- Regulatory compliance ---
    pre_count = len(findings)
    _check_regulatory_compliance(card_text, datasets_used, findings)
    if len(findings) == pre_count and datasets_used:
        passed.append(PassedCheck(id="PRIV-018", title="Regulatory Compliance Addressed",
                                  detail="No web-scraped data compliance gaps detected"))

    # --- Data provenance ---
    if not datasets_used:
        findings.append(Finding(
            id="PRIV-020",
            title="Training Data Not Disclosed",
            severity=Severity.MEDIUM,
            description=(
                "No training datasets are documented. Without knowing training data provenance, "
                "it is impossible to assess data privacy compliance, consent, or bias."
            ),
            remediation="Document all training datasets in the model card.",
            references=["https://arxiv.org/abs/1803.09010"],
        ))
    else:
        passed.append(PassedCheck(id="PRIV-020", title="Training Data Documented",
                                  detail=f"{len(datasets_used)} dataset(s) declared"))

    if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in findings):
        status = EvaluatorStatus.FAIL
    elif any(f.severity == Severity.MEDIUM for f in findings):
        status = EvaluatorStatus.WARNING
    elif findings:
        status = EvaluatorStatus.WARNING
    else:
        status = EvaluatorStatus.PASS

    critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    high = sum(1 for f in findings if f.severity == Severity.HIGH)
    summary = (
        f"{len(findings)} finding(s): {critical} critical, {high} high. "
        f"Datasets: {len(datasets_used)}. "
        f"Pipeline: {pipeline_tag}."
    )

    return EvaluatorResult(
        name="Data & Privacy Risks",
        status=status,
        summary=summary,
        findings=findings,
        passed_checks=passed,
        metadata=metadata,
    )


def _scan_for_pii(text: str, source_label: str) -> list[Finding]:
    findings: list[Finding] = []
    if not text:
        return findings

    for pii_type, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Credentials always high; email in model_card is low (contact info);
            # everything else in model_card is medium; local files follow same logic
            if pii_type in ("api_key", "aws_key", "private_key", "ssn", "credit_card"):
                sev = Severity.HIGH
            elif pii_type == "email" and source_label == "model_card":
                sev = Severity.LOW
            else:
                sev = Severity.MEDIUM
            # Redact matches in the finding
            redacted = [m[:4] + "****" if len(m) > 8 else "****" for m in matches[:5]]
            findings.append(Finding(
                id=f"PRIV-PII-{pii_type.upper()}",
                title=f"PII Detected in {source_label}: {pii_type.replace('_', ' ').title()}",
                severity=sev,
                description=(
                    f"Found {len(matches)} instance(s) of {pii_type} in {source_label}. "
                    f"Samples (redacted): {', '.join(redacted)}"
                ),
                remediation=f"Remove {pii_type} data from {source_label} before publishing.",
                details={"pii_type": pii_type, "count": len(matches)},
            ))

    return findings


def _scan_local_files_for_pii(local_path: str) -> list[Finding]:
    """Scan text files in local model directory for PII."""
    findings: list[Finding] = []
    text_exts = {".json", ".txt", ".md", ".yaml", ".yml", ".cfg", ".ini"}
    p = Path(local_path)
    if not p.is_dir():
        return findings

    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in text_exts and f.stat().st_size < 10_000_000:
            try:
                content = f.read_text(errors="replace")
                file_findings = _scan_for_pii(content, str(f.relative_to(p)))
                findings.extend(file_findings)
            except Exception:
                continue

    return findings


def _analyze_training_data(
    datasets: list[str],
    findings: list[Finding],
    metadata: dict[str, Any],
) -> None:
    high_risk = [d for d in datasets if any(hr in d.lower() for hr in _HIGH_RISK_DATASETS)]
    consent_concerns = [d for d in datasets if any(cc in d.lower() for cc in _CONSENT_CONCERN_DATASETS)]

    metadata["high_risk_datasets"] = high_risk
    metadata["consent_concern_datasets"] = consent_concerns

    if high_risk:
        findings.append(Finding(
            id="PRIV-010",
            title=f"High-Risk Training Datasets: {', '.join(high_risk)}",
            severity=Severity.MEDIUM,
            description=(
                f"This model was trained on datasets known to contain web-scraped content "
                f"({', '.join(high_risk)}) which may include personal information without "
                "explicit consent. The model may have memorized PII from training data."
            ),
            remediation=(
                "Evaluate the model for training data memorization. "
                "Implement differential privacy or data deduplication."
            ),
            details={"datasets": high_risk},
            references=[
                "https://arxiv.org/abs/2012.07805",  # Memorization in large LMs
                "https://arxiv.org/abs/2202.07646",  # Extracting training data from LLMs
            ],
        ))

    if consent_concerns:
        findings.append(Finding(
            id="PRIV-011",
            title=f"Training Data with Consent Concerns: {', '.join(consent_concerns)}",
            severity=Severity.MEDIUM,
            description=(
                f"Dataset(s) {', '.join(consent_concerns)} have known copyright or consent issues. "
                "Using models trained on these may create legal exposure."
            ),
            remediation="Consult legal counsel about permissible use and consider re-training on consented data.",
            details={"datasets": consent_concerns},
            references=["https://www.law.uchicago.edu/news/scraping-web-legal-or-not"],
        ))


def _check_privacy_disclosures(card_text: str, findings: list[Finding]) -> None:
    if not card_text:
        return

    card_lower = card_text.lower()
    privacy_keywords = ["privacy", "gdpr", "ccpa", "personal data", "pii", "anonymi", "differenti"]
    has_privacy_section = any(kw in card_lower for kw in privacy_keywords)

    if not has_privacy_section:
        findings.append(Finding(
            id="PRIV-015",
            title="No Privacy Disclosures in Model Card",
            severity=Severity.LOW,
            description=(
                "The model card does not address privacy considerations, "
                "data anonymization procedures, or regulatory compliance."
            ),
            remediation=(
                "Document privacy measures taken during training data collection and processing. "
                "Include GDPR/CCPA compliance status if applicable."
            ),
        ))


def _assess_membership_inference_risk(
    pipeline_tag: str,
    card_text: str,
    datasets: list[str],
    findings: list[Finding],
    metadata: dict[str, Any],
) -> None:
    # Models trained on personal data with low privacy measures are at higher risk
    high_risk_pipelines = {
        "text-generation", "fill-mask", "text2text-generation",  # LLMs memorize training data
        "feature-extraction",  # embeddings can leak training info
    }

    card_lower = (card_text or "").lower()
    has_differential_privacy = any(kw in card_lower for kw in ["differential privacy", "dp-sgd", "epsilon"])
    has_deduplication = any(kw in card_lower for kw in ["deduplication", "deduplicated", "dedup"])

    risk_score = 0
    risk_factors = []

    if pipeline_tag.lower() in high_risk_pipelines:
        risk_score += 2
        risk_factors.append(f"high-risk pipeline type ({pipeline_tag})")

    high_risk_training_data = [d for d in datasets if any(hr in d.lower() for hr in _HIGH_RISK_DATASETS)]
    if high_risk_training_data:
        risk_score += 2
        risk_factors.append(f"web-scraped training data ({', '.join(high_risk_training_data[:2])})")

    if not has_differential_privacy:
        risk_score += 1
        risk_factors.append("no differential privacy documented")

    if not has_deduplication:
        risk_score += 1
        risk_factors.append("no data deduplication documented")

    metadata["membership_inference_risk_score"] = risk_score
    metadata["membership_inference_risk_factors"] = risk_factors

    if risk_score >= 4:
        severity = Severity.HIGH
    elif risk_score >= 2:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    if risk_score >= 2:
        findings.append(Finding(
            id="PRIV-016",
            title=f"Elevated Membership Inference Attack Risk (score={risk_score}/6)",
            severity=severity,
            description=(
                f"This model has an elevated risk of membership inference attacks. "
                f"Risk factors: {'; '.join(risk_factors)}.\n\n"
                "Membership inference attacks allow an adversary to determine whether a specific "
                "individual's data was included in the training set."
            ),
            remediation=(
                "Mitigate with: (1) differential privacy during training, "
                "(2) training data deduplication, (3) minimum prediction confidence thresholding."
            ),
            details={"risk_score": risk_score, "risk_factors": risk_factors},
            references=[
                "https://arxiv.org/abs/1610.05820",  # Membership inference attacks
                "https://arxiv.org/abs/2112.03570",  # Measuring forgetting
            ],
        ))


def _check_regulatory_compliance(
    card_text: str,
    datasets: list[str],
    findings: list[Finding],
) -> None:
    card_lower = (card_text or "").lower()

    gdpr_claim = "gdpr" in card_lower
    ccpa_claim = "ccpa" in card_lower

    # Flag if using web-scraped data but no compliance mention
    uses_web_data = any(d for d in datasets if any(hr in d.lower() for hr in {"c4", "cc100", "common_crawl", "oscar"}))
    if uses_web_data and not gdpr_claim and not ccpa_claim:
        findings.append(Finding(
            id="PRIV-018",
            title="Web-Scraped Training Data Without Regulatory Compliance Disclosure",
            severity=Severity.MEDIUM,
            description=(
                "The model uses web-scraped training data but the model card does not "
                "mention GDPR or CCPA compliance. If any EU/CA residents' data was "
                "scraped, this may constitute a compliance violation."
            ),
            remediation=(
                "Review scraping sources for EU/CA personal data. "
                "Implement a data subject rights process (right to erasure). "
                "Document GDPR/CCPA compliance status in the model card."
            ),
            references=[
                "https://gdpr.eu/",
                "https://oag.ca.gov/privacy/ccpa",
            ],
        ))


def _read_local_card(local_path: str) -> str:
    p = Path(local_path)
    for name in ("README.md", "MODEL_CARD.md", "model_card.md"):
        cp = p / name
        if cp.exists():
            return cp.read_text(errors="replace")
    return ""


def _extract_datasets_from_card(text: str) -> list[str]:
    datasets: list[str] = []
    # Look for YAML front matter datasets field
    m = re.search(r"^datasets:\s*\n((?:\s*-\s*.+\n)+)", text, re.MULTILINE)
    if m:
        for line in m.group(1).splitlines():
            ds = line.strip().lstrip("-").strip()
            if ds:
                datasets.append(ds)
    # Also look for inline mentions
    for m2 in re.finditer(r"trained on\s+([A-Za-z0-9_\-/,\s]+?)(?:\.|,|\n)", text, re.IGNORECASE):
        for ds in re.split(r"[,\s]+", m2.group(1).strip()):
            if ds and ds not in datasets:
                datasets.append(ds)
    return datasets
