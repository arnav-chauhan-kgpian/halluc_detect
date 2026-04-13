from __future__ import annotations
"""Storage utilities – save responses as Parquet + hidden states as .pt files."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from config import PipelineConfig

logger = logging.getLogger(__name__)


class ResultStorage:
    """Accumulate generation results and flush to disk."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._records: list[dict] = []
        self._total_saved: int = 0

        # Ensure output dirs exist
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def save_sample(
        self,
        query_id: str,
        query_text: str,
        category: str,
        response_text: str,
        generated_token_ids: torch.Tensor,
        original_record: Optional[dict] = None,
    ) -> None:
        """Buffer metadata and response for the sample."""
        record = {
            "query_id": query_id,
            "query_text": query_text,
            "category": category,
            "response_text": response_text,
            "num_generated_tokens": len(generated_token_ids),
        }
        
        if original_record:
            for k, v in original_record.items():
                if k not in record:
                    record[k] = v
            
        self._records.append(record)
        self._total_saved += 1

    def flush_metadata(self) -> Path:
        """Write accumulated metadata to a Parquet file and returning its path."""
        parquet_path = self.cfg.output_dir / "results.parquet"
        jsonl_path = self.cfg.output_dir / "results.jsonl"
        
        if not self._records:
            return parquet_path
            
        df_new = pd.DataFrame(self._records)
        if parquet_path.exists():
            df_old = pd.read_parquet(parquet_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset="query_id", keep="last")
            df_combined.to_parquet(parquet_path, index=False)
            df_combined.to_json(jsonl_path, orient="records", lines=True)
            logger.info("Merged metadata for %d new samples \u2192 %s and %s (Total rows: %d)", len(self._records), parquet_path, jsonl_path, len(df_combined))
        else:
            df_new.to_parquet(parquet_path, index=False)
            df_new.to_json(jsonl_path, orient="records", lines=True)
            logger.info("Saved initial metadata for %d samples \u2192 %s and %s", len(self._records), parquet_path, jsonl_path)
            
        self._records = []
        return parquet_path

    @property
    def num_saved(self) -> int:
        return self._total_saved
