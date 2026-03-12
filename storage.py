"""Storage utilities – save responses as Parquet + hidden states as .pt files."""

import logging
from pathlib import Path

import pandas as pd
import torch

from config import PipelineConfig

logger = logging.getLogger(__name__)


class ResultStorage:
    """Accumulate generation results and flush to disk."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._records: list[dict] = []

        # Ensure output dirs exist
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.hidden_states_dir.mkdir(parents=True, exist_ok=True)

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
        hidden_states: torch.Tensor,
        query_hidden_states: torch.Tensor,
    ) -> None:
        """Persist one sample's hidden states and buffer metadata."""
        # Save hidden states + token ids as a .pt file
        pt_path = self.cfg.hidden_states_dir / f"{query_id}.pt"
        torch.save(
            {
                "query_id": query_id,
                "hidden_states": hidden_states,
                "query_hidden_states": query_hidden_states,
                "generated_token_ids": generated_token_ids,
            },
            pt_path,
        )

        self._records.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "category": category,
                "response_text": response_text,
                "num_generated_tokens": len(generated_token_ids),
                "num_prompt_tokens": query_hidden_states.shape[0],
                "hidden_states_path": str(pt_path),
            }
        )

    def flush_metadata(self) -> Path:
        """Write accumulated metadata to a Parquet file and return its path."""
        parquet_path = self.cfg.output_dir / "results.parquet"
        df = pd.DataFrame(self._records)
        df.to_parquet(parquet_path, index=False)
        logger.info(
            "Saved metadata for %d samples → %s", len(self._records), parquet_path
        )
        return parquet_path

    @property
    def num_saved(self) -> int:
        return len(self._records)
