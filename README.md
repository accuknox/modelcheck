# modelcheck

**modelcheck** is an open-source CLI tool from [AccuKnox](https://accuknox.com) that performs security evaluations of machine learning models — whether hosted on [HuggingFace Hub](https://huggingface.co) or stored locally.

It covers four security dimensions and produces a self-contained HTML report, a JSON report, and a [CycloneDX AI BOM](https://cyclonedx.org/).

---

## Installation

**Requirements:** Python 3.10 or later.

### From the latest GitHub release (recommended)

```bash
pip install https://github.com/accuknox/modelcheck/releases/latest/download/modelcheck-latest-py3-none-any.whl
```

### From source

```bash
git clone https://github.com/accuknox/modelcheck.git
cd modelcheck
pip install -e .
```

### With full adversarial testing support (ART + TextAttack)

```bash
pip install "modelcheck[adversarial]"
```

---

## Quick Start

```bash
# Scan a model on HuggingFace Hub
modelcheck scan google-bert/bert-base-uncased

# Scan multiple models and generate a comparison report
modelcheck scan google-bert/bert-base-uncased distilbert/distilbert-base-uncased

# Scan a local model directory
modelcheck scan ./my-local-model

# Use a HuggingFace token for gated models
export HF_TOKEN=hf_...
modelcheck scan meta-llama/Llama-2-7b-hf
```

The tool writes an HTML report to the current directory by default:
- Single model → `<model-name>_security_card.html`
- Multiple models → `modelcheck_report.html`

### All options

```
modelcheck scan [OPTIONS] MODEL [MODEL ...]

Options:
  -t, --token TEXT          HuggingFace API token (or set HF_TOKEN env var)
  --html FILE               Custom HTML output path
  -o, --output PATH         Export JSON report (file for single, dir for multi)
  -p, --parallelism INT     Max parallel scans, 1–16 (default: 4)
  -s, --skip MODULE         Skip a module: provenance | adversarial | privacy | file-security
  --download                Download model files for deep static analysis
  --no-probes               Disable live LLM adversarial inference probes
  --fail-on LEVEL           Exit non-zero on findings at this severity (default: high)
  -r, --rules FILE          Custom rules.yaml to override severities or suppress checks
  -v, --verbose             Show full finding descriptions
  -q, --quiet               Suppress terminal output (CI mode)
```

```bash
# List all available check IDs
modelcheck list-checks

# Quick model info without a full scan
modelcheck info google-bert/bert-base-uncased
```

---

## Security Checks

modelcheck evaluates models across four dimensions.

### 1 · Supply Chain & Provenance

| Check ID | Severity | Description |
|---|---|---|
| `PROV-001` | Critical | HuggingFace built-in scanner flagged unsafe files |
| `PROV-002` | Medium | Unknown or anonymous model author |
| `PROV-003` | Low | Country of origin cannot be determined |
| `PROV-004` | Medium | No license declared |
| `PROV-005` | Info | Restrictive license (commercial restrictions) |
| `PROV-006` | Low | Low community adoption (< 100 downloads, < 5 likes) |
| `PROV-007` | Low | Weight files may lack LFS integrity checksums |
| `PROV-008` | Low | Missing or sparse model card (< 100 chars) |
| `PROV-009` | Info | Model uses gated access |
| `PROV-010` | High / Medium / Info | Author credibility: individual follower count or unvetted org |
| `PROV-L00x` | Critical–Medium | Local model: missing path, no model card, unknown author/country |

### 2 · Adversarial Robustness

| Check ID | Severity | Description |
|---|---|---|
| `ADV-000` | Info | Adversarial testing libraries not installed |
| `ADV-MC-001` | Medium | Model card has no adversarial robustness disclosure |
| `ADV-MC-002` | Low | Model card has no bias or fairness evaluation |
| `ADV-ART-001` | High / Medium / Low | ART FGSM + PGD attack success rate (image models) |
| `ADV-TA-001` | High / Medium / Low | TextAttack TextFooler attack success rate (NLP models) |
| `ADV-LLM-001` | High | LLM probe: DAN jailbreak triggered |
| `ADV-LLM-002` | High | LLM probe: Prompt injection via role play triggered |
| `ADV-LLM-003` | High | LLM probe: System prompt extraction triggered |
| `ADV-LLM-004` | High | LLM probe: Token smuggling (Unicode homoglyphs) triggered |
| `ADV-LLM-005` | High | LLM probe: Indirect prompt injection triggered |

> LLM probes run against the HuggingFace Inference API. Pass `--no-probes` to skip them.  
> ART and TextAttack require `pip install modelcheck[adversarial]`.

### 3 · Data & Privacy Risks

| Check ID | Severity | Description |
|---|---|---|
| `PRIV-PII-EMAIL` | Low | Email address found in model card |
| `PRIV-PII-API_KEY` | High | API key or access token found in artefacts |
| `PRIV-PII-AWS_KEY` | High | AWS access key found in artefacts |
| `PRIV-PII-PRIVATE_KEY` | High | Private key (RSA/EC/SSH) found in artefacts |
| `PRIV-PII-SSN` | High | Social Security Number pattern detected |
| `PRIV-PII-CREDIT_CARD` | High | Credit card number pattern detected |
| `PRIV-010` | Medium | High-risk web-scraped training datasets (C4, Common Crawl, LAION, etc.) |
| `PRIV-011` | Medium | Training datasets with copyright or consent concerns |
| `PRIV-015` | Low | No privacy disclosures in model card |
| `PRIV-016` | High / Medium | Elevated membership inference attack risk |
| `PRIV-018` | Medium | Web-scraped data without GDPR / CCPA disclosure |
| `PRIV-020` | Medium | Training data not documented |

### 4 · Model File Security

| Check ID | Severity | Description |
|---|---|---|
| `FSEC-HF-001` | Critical | HuggingFace security scanner flagged unsafe files |
| `FSEC-001` | Medium | Pickle-based weight files present (.pkl, .pt, .bin, .ckpt) |
| `FSEC-002` | Low | No SafeTensors alternative available |
| `FSEC-MA-*` | Varies | [modelaudit](https://github.com/protectai/modelaudit) static scan findings |
| `FSEC-PKL-001` | Critical | Dangerous pickle opcodes or unsafe imports detected |
| `FSEC-PKL-002` | Medium | Non-standard / suspicious pickle imports |
| `FSEC-FMT-*` | Medium | Suspicious file types (.exe, .sh, .dll) bundled in repo |
| `FSEC-EMBED-001` | Medium | Suspicious patterns in config JSON files |

---

## Compliance & Framework Mapping

Findings are automatically mapped to industry frameworks:

| Framework | Mapping |
|---|---|
| **OWASP LLM Top 10** | Prompt injection, insecure output handling, supply chain |
| **MITRE ATLAS** | LLM prompt injection, model backdoor, supply chain |
| **MITRE ATT&CK** | Supply chain compromise, data collection, exfiltration |
| **NIST AI RMF** | Govern, Map, Measure, Manage functions |
| **EU AI Act** | High-risk AI transparency and data governance requirements |
| **CWE** | CWE-20 (Improper Input Validation), CWE-502 (Deserialization), etc. |
| **ISO/IEC 42001** | AI management system controls |

---

## Sample Report

[**→ View a sample HTML security report**](docs/HFModelReportExt.html)

The report includes:
- **Security dashboard** — overall score (0–100), dimension radar chart, finding distribution donut
- **Module scores** — per-dimension score cards with severity breakdown
- **Compliance dashboard** — framework violation grid (OWASP LLM Top 10, MITRE ATLAS, etc.)
- **Detailed findings** — expandable cards for each issue with remediation steps and references
- **Passed checks** — full list of checks that were run and passed
- **AI BOM** — CycloneDX 1.5 machine-readable bill of materials (downloadable JSON)
- **Recommendations** — prioritised action plan based on triggered findings
- **Multi-model comparison** — side-by-side scores, compliance table, and automated recommendation on which model is safest to use

---

## Output Files

| File | Description |
|---|---|
| `<model>_security_card.html` | Self-contained HTML report (single model) |
| `modelcheck_report.html` | Multi-model comparison report |
| `<model>.json` | Machine-readable JSON report |
| `<model>_aibom.json` | CycloneDX 1.5 AI Bill of Materials |

---

## CI / CD Integration

```bash
# Fail the pipeline if any high-severity findings are detected
modelcheck scan my-org/my-model --fail-on high --quiet

# Fail only on critical findings
modelcheck scan my-org/my-model --fail-on critical --no-html --quiet
```

Exit code `0` = no findings at or above the threshold.  
Exit code `1` = findings found, or scan error.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [AccuKnox](https://accuknox.com).
