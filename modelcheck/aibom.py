"""
AI Bill of Materials (AIBOM) generator.

Produces a CycloneDX 1.5-compliant AI BOM document from a modelcheck
SecurityReport.  The output is a plain Python dict; call json.dumps() to
serialise it.

Spec:         https://cyclonedx.org/specification/overview/
AI/ML profile: https://cyclonedx.org/capabilities/mlbom/
"""

from __future__ import annotations

import uuid
from datetime import timezone
from typing import Any

from .models import SecurityReport, ModelSource, Severity


# Rough CVSS-like base score for each severity level (used in vulnerability ratings)
_SEV_SCORE: dict[Severity, float] = {
    Severity.CRITICAL: 9.0,
    Severity.HIGH:     7.0,
    Severity.MEDIUM:   5.0,
    Severity.LOW:      2.0,
    Severity.INFO:     0.0,
    Severity.UNKNOWN:  0.0,
}


def generate_aibom(report: SecurityReport) -> dict[str, Any]:
    """
    Build a CycloneDX 1.5 AI BOM from a modelcheck SecurityReport.

    Returns a plain Python dict.  The caller is responsible for JSON
    serialisation (e.g. ``json.dumps(bom, indent=2)``).
    """
    prov_meta = (report.provenance  and report.provenance.metadata)  or {}
    file_meta = (report.file_security and report.file_security.metadata) or {}

    bom_ref = "model-main"
    serial  = f"urn:uuid:{uuid.uuid4()}"

    # ── metadata ──────────────────────────────────────────────────────────────
    ts = report.generated_at.replace(tzinfo=timezone.utc).isoformat()

    bom_metadata: dict[str, Any] = {
        "timestamp": ts,
        "tools": [
            {
                "vendor":  "AccuKnox",
                "name":    "modelcheck",
                "version": report.tool_version,
                "externalReferences": [
                    {"type": "website", "url": "https://accuknox.com"}
                ],
            }
        ],
        "component": {
            "type":    "application",
            "name":    "modelcheck",
            "version": report.tool_version,
        },
    }

    # ── collect provenance fields ─────────────────────────────────────────────
    author         = prov_meta.get("author",            "Unknown")
    license_       = prov_meta.get("license",           "Unknown")
    pipeline       = prov_meta.get("pipeline_tag",      "Unknown")
    country        = prov_meta.get("country_of_origin", "Unknown")
    datasets       = prov_meta.get("datasets",          [])
    tags           = prov_meta.get("tags",              [])
    created        = prov_meta.get("created_at",        "Unknown")
    modified       = prov_meta.get("last_modified",     "Unknown")
    downloads      = prov_meta.get("downloads",         0)
    likes          = prov_meta.get("likes",             0)
    gated          = str(prov_meta.get("gated",  "false")).lower()
    private_       = prov_meta.get("private",           False)
    arch_list      = prov_meta.get("architectures",     [])
    model_type     = prov_meta.get("model_type",        "Unknown")
    author_type    = prov_meta.get("author_type",       "unknown")
    author_followers = prov_meta.get("author_followers", 0)
    hf_sec         = prov_meta.get("hf_security_status", {}) or {}
    model_card_txt = prov_meta.get("model_card",        "")

    file_formats   = file_meta.get("file_formats", {})
    scanners_used  = file_meta.get("scanners_used", [])

    # ── component ─────────────────────────────────────────────────────────────
    component: dict[str, Any] = {
        "type":    "machine-learning-model",
        "bom-ref": bom_ref,
        "name":    report.model_id,
    }

    if author and author not in ("Unknown", ""):
        component["supplier"] = {"name": author}
        component["author"]   = author

    if license_ and license_ not in ("Unknown", "unknown", ""):
        component["licenses"] = [{"expression": license_}]

    # External references (HuggingFace link)
    if report.model_source == ModelSource.HUGGINGFACE:
        hf_url = f"https://huggingface.co/{report.model_id}"
        component["externalReferences"] = [
            {"type": "website",      "url": hf_url},
            {"type": "distribution", "url": hf_url},
        ]

    # ── properties (CycloneDX AI/ML namespace) ────────────────────────────────
    props: list[dict[str, str]] = []

    def _p(name: str, value: Any) -> None:
        if value not in (None, "", "Unknown", "unknown", [], {}):
            props.append({"name": name, "value": str(value)})

    _p("cdx:ai:model:source",        report.model_source.value)
    _p("cdx:ai:model:task",          pipeline)
    _p("cdx:ai:model:type",          model_type)
    if arch_list:
        _p("cdx:ai:model:architectures", ", ".join(arch_list))
    _p("cdx:ai:model:countryOfOrigin", country)
    _p("cdx:ai:model:createdAt",     created)
    _p("cdx:ai:model:lastModified",  modified)
    _p("cdx:ai:model:downloads",     str(downloads))
    _p("cdx:ai:model:likes",         str(likes))
    _p("cdx:ai:model:gated",         gated)
    _p("cdx:ai:model:private",       str(private_).lower())
    _p("cdx:ai:author:type",         author_type)
    _p("cdx:ai:author:followers",    str(author_followers))
    if tags:
        _p("cdx:ai:model:tags", ", ".join(str(t) for t in tags[:20]))
    if file_formats:
        _p("cdx:ai:files:safetensors", str(file_formats.get("safetensors", 0)))
        _p("cdx:ai:files:pickle",      str(file_formats.get("pickle_based", 0)))
        _p("cdx:ai:files:gguf",        str(file_formats.get("gguf", 0)))
    if scanners_used:
        _p("cdx:tool:scanners", ", ".join(scanners_used))

    if props:
        component["properties"] = props

    # ── modelCard sub-object ──────────────────────────────────────────────────
    model_card: dict[str, Any] = {}

    # Model parameters
    mc_params: dict[str, Any] = {}
    if pipeline not in ("Unknown", ""):
        mc_params["task"] = pipeline
    if arch_list:
        mc_params["architectureFamily"] = arch_list[0] if len(arch_list) == 1 else arch_list
    if datasets:
        mc_params["datasets"] = [
            {"name": d, "type": "dataset"} for d in datasets
        ]
    if mc_params:
        model_card["modelParameters"] = mc_params

    # HF security scan result
    if hf_sec:
        model_card["considerations"] = {
            "securityScan": {
                "provider":       "HuggingFace",
                "hasUnsafeFiles": bool(hf_sec.get("has_unsafe_files", False)),
            }
        }

    # Quantitative analysis — scan summary
    all_results = [
        r for r in [
            report.provenance, report.adversarial,
            report.privacy, report.file_security,
        ] if r
    ]
    counts = report.total_findings()
    model_card["quantitativeAnalysis"] = {
        "performanceMetrics": [
            {"type": "modelcheck:findings:total",    "value": str(counts.get("total",    0))},
            {"type": "modelcheck:findings:critical", "value": str(counts.get("critical", 0))},
            {"type": "modelcheck:findings:high",     "value": str(counts.get("high",     0))},
            {"type": "modelcheck:findings:medium",   "value": str(counts.get("medium",   0))},
            {"type": "modelcheck:findings:low",      "value": str(counts.get("low",      0))},
        ]
    }

    if model_card:
        component["modelCard"] = model_card

    # ── vulnerabilities (one per Finding) ────────────────────────────────────
    vulnerabilities: list[dict[str, Any]] = []
    seen: set[str] = set()

    for result in all_results:
        for finding in result.findings:
            # Deduplicate by ID to avoid repeated entries
            dup_key = f"{finding.id}::{finding.title}"
            if dup_key in seen:
                continue
            seen.add(dup_key)

            score = _SEV_SCORE.get(finding.severity, 0.0)
            vuln: dict[str, Any] = {
                "bom-ref": f"finding-{finding.id.replace(' ', '-')}",
                "id":      finding.id,
                "source":  {"name": "modelcheck", "url": "https://accuknox.com"},
                "ratings": [
                    {
                        "source":   {"name": "modelcheck"},
                        "severity": finding.severity.value,
                        "score":    {"base": score, "source": "modelcheck"},
                    }
                ],
                "description": finding.description,
                "affects": [{"ref": bom_ref}],
            }
            if finding.remediation:
                vuln["recommendation"] = finding.remediation
            if finding.references:
                vuln["advisories"] = [{"url": r} for r in finding.references]
            vulnerabilities.append(vuln)

    # ── assemble BOM ──────────────────────────────────────────────────────────
    bom: dict[str, Any] = {
        "bomFormat":    "CycloneDX",
        "specVersion":  "1.5",
        "version":      1,
        "serialNumber": serial,
        "metadata":     bom_metadata,
        "components":   [component],
    }
    if vulnerabilities:
        bom["vulnerabilities"] = vulnerabilities

    return bom
