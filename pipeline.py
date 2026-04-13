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
        batch_size = self.cfg.batch_size
        logger.info("Starting generation for %d queries (Batch size: %d) …", len(queries), batch_size)
        start = time.time()

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            query_texts = [q["query_text"] for q in batch_queries]
            
            try:
                # ── Per-Sample Seeding ──
                # For batched runs, we set the seed once per batch for simplicity, 
                # or use salted seeding per sample if needed. 
                # Here we use salted seeding for absolute consistency.
                set_seed(self.cfg.seed)

                # ── Generation ──
                batch_outputs = model.generate_batch(query_texts)

            except Exception as e:
                logger.exception("Failed on batch starting at index %d – skipping. (%s)", i, type(e).__name__)
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for idx_in_batch, output in enumerate(batch_outputs):
                q = batch_queries[idx_in_batch]
                storage.save_sample(
                    query_id=q["conversation_hash"],
                    query_text=q["query_text"],
                    category=q["category"],
                    response_text=output.response_text,
                    generated_token_ids=output.generated_token_ids,
                    original_record=q.get("original_record"),
                )

            current_total = i + len(batch_queries)
            if current_total % 100 == 0 or current_total == len(queries):
                storage.flush_metadata()
                elapsed_min = (time.time() - start) / 60
                logger.info(
                    "Progress: %d / %d  (%.1f samples/min)",
                    current_total,
                    len(queries),
                    current_total / max(elapsed_min, 0.001),
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
