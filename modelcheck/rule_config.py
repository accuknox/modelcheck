"""
Runtime rule configuration loaded from a rules.yaml file.

Supports per-rule overrides of:
  - enabled   (bool, default true)  — set false to suppress a finding
  - severity  (str)                 — override the emitted severity
  - title     (str)                 — override the finding title
  - remediation (str)               — override the remediation text

Rule IDs support a trailing '*' wildcard to match dynamic IDs, e.g.:
  PRIV-PII-*  matches  PRIV-PII-EMAIL, PRIV-PII-SSN, …
  FSEC-MA-*   matches  FSEC-MA-DANGEROUS_PATTERN, FSEC-MA-CHK-*, …
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from .models import Finding, Severity


class RuleConfig:
    """Flat index of rules loaded from a rules.yaml file."""

    def __init__(self, rules_file: str | Path) -> None:
        self._path = Path(rules_file)
        self._index: dict[str, dict[str, Any]] = {}   # id → rule dict
        self._patterns: list[tuple[str, dict[str, Any]]] = []  # wildcard patterns
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "PyYAML is required to load rules files. "
                "Install with: pip install pyyaml"
            )

        data = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid rules file: {self._path}")

        # Collect rules from all module_* sections
        for key, section in data.items():
            if not key.startswith("module_"):
                continue
            for rule in section.get("rules", []):
                rule_id: str = rule.get("id", "")
                if not rule_id:
                    continue
                if rule_id.endswith("*"):
                    self._patterns.append((rule_id, rule))
                else:
                    self._index[rule_id] = rule

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def _find(self, rule_id: str) -> dict[str, Any] | None:
        """Return the rule dict for an ID, or None if not configured."""
        if rule_id in self._index:
            return self._index[rule_id]
        for pattern, rule in self._patterns:
            if fnmatch.fnmatch(rule_id, pattern):
                return rule
        return None

    def is_enabled(self, rule_id: str) -> bool:
        """Return False only if the rule is explicitly disabled."""
        rule = self._find(rule_id)
        if rule is None:
            return True   # unknown rules are allowed through
        return bool(rule.get("enabled", True))

    # ── Finding post-processing ───────────────────────────────────────────────

    def apply(self, findings: list[Finding]) -> list[Finding]:
        """
        Apply rule overrides to a list of findings.

        - Findings for disabled rules are removed.
        - severity / title / remediation are overridden when the yaml provides them.
        """
        out: list[Finding] = []
        sev_map = {
            "critical": Severity.CRITICAL,
            "high":     Severity.HIGH,
            "medium":   Severity.MEDIUM,
            "low":      Severity.LOW,
            "info":     Severity.INFO,
        }

        for finding in findings:
            rule = self._find(finding.id)

            # Drop disabled rules
            if rule is not None and not rule.get("enabled", True):
                continue

            if rule is None:
                out.append(finding)
                continue

            # Apply overrides — only when the yaml explicitly provides them
            overrides: dict[str, Any] = {}

            sev_str = str(rule.get("severity", "")).strip().lower()
            if sev_str and sev_str in sev_map:
                overrides["severity"] = sev_map[sev_str]

            title = rule.get("title", "")
            # Only override title for non-dynamic rules (dynamic rules build their
            # own titles at runtime that carry useful context)
            if title and not rule.get("dynamic", False):
                overrides["title"] = title

            remediation = rule.get("remediation", "")
            if remediation:
                overrides["remediation"] = remediation.strip()

            if overrides:
                out.append(finding.model_copy(update=overrides))
            else:
                out.append(finding)

        return out

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        total   = len(self._index) + len(self._patterns)
        disabled = sum(
            1 for r in self._index.values() if not r.get("enabled", True)
        ) + sum(
            1 for _, r in self._patterns if not r.get("enabled", True)
        )
        return (
            f"{self._path.name}: {total} rules loaded, {disabled} disabled"
        )
