"""Load and filter WildChat queries by category."""

import logging
import re
from typing import Optional

from datasets import load_dataset

from config import PipelineConfig

logger = logging.getLogger(__name__)

# Keyword regex patterns used to assign a coarse category to each query.
CATEGORY_PATTERNS: dict[str, list[str]] = {
    "url": [
        r"\burl\b",
        r"\blink(s)?\b",
        r"\bwebsite\b",
        r"\bwebpage\b",
        r"\bhomepage\b",
        r"\bweb\s*address\b",
        r"\bgive\s+me\s+(a|the)\s+link\b",
    ],
    "citation": [
        r"\bcit(e|ation|ations)\b",
        r"\breference(s)?\b",
        r"\bsources?\b",
        r"\bbibliograph",
        r"\bpapers?\b.*\brecommend",
        r"\bacademic\b.*\breference\b",
    ],
    "coding": [
        r"\bcode\b",
        r"\bprogram(ming)?\b",
        r"\bfunction\b",
        r"\bscript\b",
        r"\bimplement\b",
        r"\balgorithm\b",
        r"\bpython\b",
        r"\bjavascript\b",
        r"\bjava\b",
        r"\bc\+\+\b",
        r"\bhtml\b",
        r"\bcss\b",
        r"\bsql\b",
        r"\bapi\b",
        r"\bdebug\b",
        r"\bclass\b.*\bmethod\b",
        r"\bwrite\b.*\b(a|the)\b.*\b(function|script|program)\b",
    ],
    "factual": [
        r"^(who|what|when|where|which|how\s+many|how\s+much|how\s+old|how\s+far|how\s+long)\b",
        r"\bis\s+it\s+true\b",
        r"\bcapital\s+of\b",
        r"\bpopulation\b",
        r"\bfounded\b",
        r"\binvented\b",
        r"\bdiscovered\b",
        r"\bhistory\s+of\b",
        r"\btell\s+me\s+(about|a\s+fact)\b",
    ],
}


def _classify_query(text: str, categories: list[str]) -> Optional[str]:
    """Return the first matching category for *text*, or ``None``."""
    text_lower = text.lower()
    for cat in categories:
        patterns = CATEGORY_PATTERNS.get(cat, [])
        for pat in patterns:
            if re.search(pat, text_lower):
                return cat
    return None


def load_wildchat_queries(cfg: PipelineConfig) -> list[dict]:
    """Download WildChat-1M and return filtered query dicts.

    Each returned dict has keys:
        conversation_hash, query_text, category
    """
    logger.info(
        "Loading dataset %s (split=%s) …", cfg.dataset_name, cfg.dataset_split
    )
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    queries: list[dict] = []
    seen_hashes: set[str] = set()

    for row in ds:
        if len(queries) >= cfg.max_queries:
            break

        # Language filter
        lang = row.get("language", "")
        if cfg.language_filter and lang != cfg.language_filter:
            continue

        # Deduplicate
        conv_hash = row.get("conversation_hash", "")
        if conv_hash in seen_hashes:
            continue
        seen_hashes.add(conv_hash)

        # Extract the first user message
        conversation = row.get("conversation", [])
        if not conversation:
            continue
        first_user_msg = None
        for msg in conversation:
            if msg.get("role") == "user":
                first_user_msg = msg["content"]
                break
        if not first_user_msg or len(first_user_msg.strip()) < 10:
            continue

        # Classify
        category = _classify_query(first_user_msg, cfg.categories)
        if category is None:
            continue

        queries.append(
            {
                "conversation_hash": conv_hash,
                "query_text": first_user_msg.strip(),
                "category": category,
            }
        )

    logger.info(
        "Loaded %d queries across categories: %s",
        len(queries),
        {
            cat: sum(1 for q in queries if q["category"] == cat)
            for cat in cfg.categories
        },
    )
    return queries
