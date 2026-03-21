"""Generation pipeline – ties data loading, model inference, and storage together."""

import logging
import re
import time
from typing import Dict, Any

from tqdm import tqdm

from config import PipelineConfig
from data_loader import load_wildchat_queries
from model_wrapper import Qwen3Wrapper
from storage import ResultStorage
from utils import set_seed

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """End-to-end: load WildChat → generate with Qwen3 → store results."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(self) -> None:
        # 1. Load & filter queries
        queries = load_wildchat_queries(self.cfg)
        if not queries:
            logger.warning("No queries matched the filters – nothing to do.")
            return

        # 2. Filter already processed queries (Resume support)
        parquet_path = self.cfg.output_dir / "results.parquet"
        processed_ids = set()
        if parquet_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                processed_ids = set(df["query_id"].astype(str).tolist())
            except Exception as e:
                logger.warning("Failed to load existing results for resume support: %s", e)

        if processed_ids:
            original_count = len(queries)
            queries = [q for q in queries if q["conversation_hash"] not in processed_ids]
            logger.info(
                "Resuming: found %d already processed samples in results.parquet. Remaining: %d / %d",
                len(processed_ids),
                len(queries),
                original_count,
            )
        
        if not queries:
            logger.info("All queries already processed. Nothing to do.")
            return

        # 3. Load model
        model = Qwen3Wrapper(self.cfg)

        # 3. Prepare storage
        storage = ResultStorage(self.cfg)

        # 4. Generate
        logger.info("Starting generation for %d queries …", len(queries))
        start = time.time()

        for idx, q in enumerate(tqdm(queries, desc="Generating")):
            query_id = q["conversation_hash"]
            query_text = q["query_text"]

            try:
                # ── Per-Sample Seeding ──
                set_seed(self.cfg.seed)

                # ── Generation ──
                output = model.generate(query_text)

            except Exception as e:
                logger.exception("Failed on query %s – skipping. (%s)", query_id, type(e).__name__)
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            storage.save_sample(
                query_id=query_id,
                query_text=query_text,
                category=q["category"],
                response_text=output.response_text,
            )

            if (idx + 1) % 100 == 0:
                storage.flush_metadata()
                logger.info(
                    "Progress: %d / %d  (%.1f samples/min)",
                    idx + 1,
                    len(queries),
                    (idx + 1) / ((time.time() - start) / 60),
                )

        # 5. Flush metadata
        storage.flush_metadata()

        elapsed = time.time() - start
        logger.info(
            "Done – %d samples generated in %.1f min (%.2f s/sample).",
            storage.num_saved,
            elapsed / 60,
            elapsed / max(storage.num_saved, 1),
        )
