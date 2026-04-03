"""HuggingFace Hub utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelCard, hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError


# Rough mapping of known HF organisation → country
_ORG_COUNTRY_MAP: dict[str, str] = {
    # USA
    "meta-llama": "United States",
    "facebook": "United States",
    "meta": "United States",
    "openai": "United States",
    "microsoft": "United States",
    "google": "United States",
    "google-deepmind": "United States",
    "deepmind": "United Kingdom",
    "anthropic": "United States",
    "mistralai": "France",
    "huggingface": "United States",
    "stabilityai": "United Kingdom",
    "tiiuae": "United Arab Emirates",
    "allenai": "United States",
    "bigscience": "International (HuggingFace BigScience)",
    "eleutherai": "United States",
    "cohere": "Canada",
    "ai21-labs": "Israel",
    "ai21": "Israel",
    "baidu": "China",
    "alibaba": "China",
    "alibaba-cloud": "China",
    "qwen": "China",
    "01-ai": "China",
    "deepseek-ai": "China",
    "openbmb": "China",
    "zhipu-ai": "China",
    "thudm": "China",
    "baichuan-inc": "China",
    "internlm": "China",
    "01ai": "China",
    "tigerresearch": "China",
    "nousresearch": "United States",
    "teknium": "United States",
    "unsloth": "Australia",
    "lmsys": "United States",
    "berkeley-nest": "United States",
    "togethercomputer": "United States",
    "together": "United States",
    "mosaicml": "United States",
    "cerebras": "United States",
    "adept": "United States",
    "xai": "United States",
    "samsung": "South Korea",
    "kakao": "South Korea",
    "kakaobrain": "South Korea",
    "upstage": "South Korea",
    "yanolja": "South Korea",
    "lgresearch": "South Korea",
    "rinna": "Japan",
    "cyberagent": "Japan",
    "pfnet": "Japan",
    "ricerca-ai": "Japan",
    "elyza": "Japan",
    "inceptioniai": "United Arab Emirates",
    "meituan": "China",
    "sensetime": "China",
    "iflytek": "China",
    "moonshot-ai": "China",
    "minimax": "China",
    "nvidiaai": "United States",
    "nvidia": "United States",
    "writer": "United States",
    "amazon": "United States",
    "aws": "United States",
    "ibm": "United States",
    "ibm-granite": "United States",
    "lightgbm": "International",
    "scikit-learn": "International",
    "explosion": "Germany",
    "dbmdz": "Germany",
    "deepset": "Germany",
    "occiglot": "Germany",
    "malteos": "Germany",
    "ai-forever": "Russia",
    "sber": "Russia",
    "sberbank-ai": "Russia",
    "yandex": "Russia",
}


# Curated set of well-known, institutionally backed organisations on HuggingFace.
# Presence here means the author is considered trusted for provenance purposes.
WELL_KNOWN_ORGS: frozenset[str] = frozenset({
    # Big Tech / Cloud
    "google", "google-deepmind", "deepmind", "microsoft", "meta", "meta-llama",
    "facebook", "amazon", "aws", "apple", "nvidia", "nvidiaai", "ibm", "ibm-granite",
    "openai", "anthropic", "xai",
    # AI Labs & Research
    "mistralai", "huggingface", "stabilityai", "allenai", "bigscience",
    "eleutherai", "tiiuae", "cohere", "ai21-labs", "ai21",
    "cerebras", "mosaicml", "togethercomputer", "together", "adept",
    "lmsys", "berkeley-nest", "nousresearch",
    # Asian AI / Tech
    "qwen", "deepseek-ai", "baidu", "alibaba", "alibaba-cloud",
    "openbmb", "zhipu-ai", "thudm", "baichuan-inc", "internlm",
    "01-ai", "01ai", "tigerresearch", "moonshot-ai", "minimax",
    "meituan", "sensetime", "iflytek",
    "kakaobrain", "kakao", "samsung", "upstage", "lgresearch",
    "rinna", "cyberagent", "elyza", "pfnet", "ricerca-ai",
    # European AI
    "explosion", "deepset", "dbmdz", "occiglot",
    "sberbank-ai", "ai-forever", "yandex", "sber",
    # Academic / Collaborative
    "sentence-transformers", "bert-base", "distilbert",
    "inceptioniai",
})


def fetch_author_profile(author: str, token: str | None = None) -> dict:
    """
    Fetch HuggingFace profile info for a model author.

    Returns a dict with:
      author_type  : "individual" | "organization_known" | "organization_unknown" | "unknown"
      num_followers: int (0 if unavailable)
      num_models   : int
      is_well_known: bool
    """
    result = {
        "author_type": "unknown",
        "num_followers": 0,
        "num_models": 0,
        "is_well_known": False,
    }

    # Normalize: strip sub-namespace (e.g. "meta-llama/Llama-2" → "meta-llama")
    namespace = author.split("/")[0].lower().strip()

    if namespace in WELL_KNOWN_ORGS:
        result["author_type"] = "organization_known"
        result["is_well_known"] = True
        return result

    # Try individual user lookup
    api = HfApi(token=token)
    try:
        profile = api.get_user_overview(author)
        result["author_type"] = "individual"
        result["num_followers"] = getattr(profile, "num_followers", 0) or 0
        result["num_models"]    = getattr(profile, "num_models", 0) or 0
    except Exception:
        # 404 → likely an org namespace not in our known list
        result["author_type"] = "organization_unknown"

    return result


def _infer_country(author: str, model_card_text: str) -> str:
    """Best-effort country inference from org name and model card content."""
    org = author.lower().split("/")[0] if "/" in author else author.lower()
    if org in _ORG_COUNTRY_MAP:
        return _ORG_COUNTRY_MAP[org]
    # Try prefix matches
    for key, country in _ORG_COUNTRY_MAP.items():
        if org.startswith(key) or key.startswith(org):
            return country
    # Scan model card for geographic clues
    card_lower = model_card_text.lower()
    geo_hints = [
        (["tsinghua", "peking university", "fudan", "shanghai", "beijing", "chinese academy"], "China"),
        (["tokyo", "kyoto", "osaka", "japanese", "japan"], "Japan"),
        (["seoul", "kaist", "postech", "korean", "korea"], "South Korea"),
        (["cambridge", "oxford", "imperial college", "ucl", "uk ", "united kingdom"], "United Kingdom"),
        (["paris", "inria", "sorbonne", "french", "france"], "France"),
        (["berlin", "munich", "german", "germany", "dfki", "tu berlin"], "Germany"),
        (["stanford", "mit ", "cmu ", "carnegie", "berkeley", "columbia", "american", "usa"], "United States"),
        (["toronto", "montreal", "mila", "vector institute", "canadian", "canada"], "Canada"),
        (["moscow", "yandex", "sber", "russian", "russia"], "Russia"),
        (["tel aviv", "weizmann", "technion", "israel", "israeli"], "Israel"),
        (["dubai", "abu dhabi", "uae", "emirat"], "United Arab Emirates"),
        (["singapore", "nus ", "ntu "], "Singapore"),
        (["sydney", "melbourne", "australian", "australia"], "Australia"),
    ]
    for hints, country in geo_hints:
        if any(h in card_lower for h in hints):
            return country
    return "Unknown"


def fetch_hf_model_info(model_id: str, token: str | None = None) -> dict[str, Any]:
    """Fetch comprehensive metadata for a HuggingFace model."""
    api = HfApi(token=token)
    info: dict[str, Any] = {"model_id": model_id}

    try:
        model_info = api.model_info(model_id, securityStatus=True, files_metadata=True)
        info["author"] = model_info.author or "Unknown"
        info["created_at"] = str(model_info.created_at) if model_info.created_at else "Unknown"
        info["last_modified"] = str(model_info.last_modified) if model_info.last_modified else "Unknown"
        info["downloads"] = getattr(model_info, "downloads", 0) or 0
        info["likes"] = getattr(model_info, "likes", 0) or 0
        info["tags"] = list(model_info.tags or [])
        info["pipeline_tag"] = model_info.pipeline_tag or "Unknown"
        info["private"] = model_info.private or False
        info["gated"] = str(model_info.gated) if model_info.gated else "false"
        info["license"] = _extract_license(model_info.tags or [])

        # Security status from HF
        sec = getattr(model_info, "security_status", None) or getattr(model_info, "securityStatus", None)
        if sec:
            info["hf_security_status"] = {
                "has_unsafe_files": getattr(sec, "has_unsafe_file", None),
                "scanners": getattr(sec, "scanner_results", None),
            }

        # Files
        siblings = model_info.siblings or []
        info["files"] = [
            {
                "filename": s.rfilename,
                "size": getattr(s, "size", None),
                "blob_id": getattr(s, "blob_id", None),
                "lfs": getattr(s, "lfs", None) is not None,
            }
            for s in siblings
        ]

        # Datasets used
        info["datasets"] = [
            d.dataset_name if hasattr(d, "dataset_name") else str(d)
            for d in (getattr(model_info, "card_data", None) and
                      getattr(model_info.card_data, "datasets", None) or [])
        ]

    except RepositoryNotFoundError:
        info["error"] = f"Model '{model_id}' not found on HuggingFace Hub"
        return info
    except Exception as exc:
        info["error"] = str(exc)

    # Fetch and parse model card
    try:
        card = ModelCard.load(model_id, token=token)
        info["model_card"] = card.content
        info["model_card_metadata"] = card.data.to_dict() if card.data else {}
    except Exception:
        info["model_card"] = ""
        info["model_card_metadata"] = {}

    # Infer country
    info["country_of_origin"] = _infer_country(
        info.get("author", ""), info.get("model_card", "")
    )

    # Author profile (org vs individual, follower count)
    author = info.get("author", "")
    if author and author.lower() != "unknown":
        info["author_profile"] = fetch_author_profile(author, token)
    else:
        info["author_profile"] = {"author_type": "unknown", "num_followers": 0,
                                  "num_models": 0, "is_well_known": False}

    return info


def _extract_license(tags: list[str]) -> str:
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return "Unknown"


def download_model_locally(model_id: str, local_dir: str, token: str | None = None) -> str:
    """Download a HuggingFace model to a local directory and return the path."""
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot"],
    )
    return path
