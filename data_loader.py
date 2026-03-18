"""Load and filter WildChat queries by category."""

import logging
import re
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

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
        r"\bc\+\+",
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
    return "general"


def load_wildchat_queries(cfg: PipelineConfig) -> list[dict]:
    """Load queries from local file. Handles both JSON and Python-literal formats."""
    import json
    import ast
    from pathlib import Path
    
    if cfg.data_path:
        target_jsonl = Path(cfg.data_path)
    else:
        # Check standard locations
        target_jsonl = Path(__file__).parent / "5000_convhash_labels.jsonl"
        if not target_jsonl.exists():
            # Check Kaggle input directory common path
            kaggle_path = Path("/kaggle/input/my-dataset/5000_convhash_labels.jsonl")
            if kaggle_path.exists():
                target_jsonl = kaggle_path

    if not target_jsonl.exists():
        logger.error("File not found: %s", target_jsonl)
        return []

    logger.info("Loading queries from: %s", target_jsonl)
    queries = []
    seen_hashes = set()

    with open(target_jsonl, "r", encoding="utf-8") as f:
        # Some JSONL files might be formatted with multiple lines per record (though rare)
        # We'll try to handle standard JSONL (one record per line)
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if len(queries) >= cfg.max_queries:
                break
                
            record = None
            try:
                # Try standard JSON first
                record = json.loads(line)
            except json.JSONDecodeError:
                try:
                    # Try Python literal (common in some exported datasets)
                    record = ast.literal_eval(line)
                except Exception as e:
                    logger.warning("Line %d: Failed to parse record: %s", line_num, e)
                    continue

            if not record:
                continue

            conv_hash = record.get("conversation_hash")
            if not conv_hash or conv_hash in seen_hashes:
                continue
            
            # Use 'query' (Kaggle format) or 'query_text' (current pipeline format)
            query_text = record.get("query") or record.get("query_text", "")
            
            # Handle cases where the query might be a list or other structure
            if isinstance(query_text, list) and query_text:
                query_text = query_text[0]
            
            if not query_text or len(str(query_text).strip()) < 10:
                continue
                
            query_str = str(query_text).strip()
            category = _classify_query(query_str, cfg.categories)
            
            queries.append({
                "conversation_hash": conv_hash,
                "query_text": query_str,
                "category": category,
            })
            seen_hashes.add(conv_hash)

    logger.info("Loaded %d queries from local file.", len(queries))
    return queries
