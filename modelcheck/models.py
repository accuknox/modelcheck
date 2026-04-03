"""Shared Pydantic models for the security report."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


class Finding(BaseModel):
    id: str
    title: str
    severity: Severity
    description: str
    details: dict[str, Any] = Field(default_factory=dict)
    remediation: str = ""
    references: list[str] = Field(default_factory=list)


class EvaluatorStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class PassedCheck(BaseModel):
    id: str
    title: str
    detail: str = ""


class EvaluatorResult(BaseModel):
    name: str
    status: EvaluatorStatus
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    passed_checks: list[PassedCheck] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.LOW)


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class SecurityReport(BaseModel):
    model_id: str
    model_source: ModelSource
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tool_version: str = "0.1.0"

    provenance: EvaluatorResult | None = None
    adversarial: EvaluatorResult | None = None
    privacy: EvaluatorResult | None = None
    file_security: EvaluatorResult | None = None

    @property
    def overall_severity(self) -> Severity:
        results = [r for r in [self.provenance, self.adversarial, self.privacy, self.file_security] if r]
        all_findings = [f for r in results for f in r.findings]
        if not all_findings:
            return Severity.INFO
        if any(f.severity == Severity.CRITICAL for f in all_findings):
            return Severity.CRITICAL
        if any(f.severity == Severity.HIGH for f in all_findings):
            return Severity.HIGH
        if any(f.severity == Severity.MEDIUM for f in all_findings):
            return Severity.MEDIUM
        if any(f.severity == Severity.LOW for f in all_findings):
            return Severity.LOW
        return Severity.INFO

    def total_findings(self) -> dict[str, int]:
        results = [r for r in [self.provenance, self.adversarial, self.privacy, self.file_security] if r]
        all_findings = [f for r in results for f in r.findings]
        counts: dict[str, int] = {s.value: 0 for s in Severity}
        for f in all_findings:
            counts[f.severity.value] += 1
        counts["total"] = len(all_findings)
        return counts
