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
from utils.similarity_analysis import run_similarity_analysis

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

            # Heuristic context extraction
            context = {}
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', query_text)
            if func_match:
                context["func_name"] = func_match.group(1)
            bin_match = re.search(r'BINARY_PATH\s*=\s*(?:r)?["\'](.*?)["\']', query_text)
            if bin_match:
                context["binary_path"] = bin_match.group(1)
            asm_match = re.search(r'```(?:asm)?\n(.*?)\n```', query_text, re.DOTALL)
            if asm_match:
                context["original_asm"] = asm_match.group(1)

            try:
                # ── Initial Generation ──
                output = model.generate(query_text)
                metrics = run_similarity_analysis(output.response_text, query_text, context=context)
                
                # ── Refinement Loop (Feedback) ──
                # If similarity is low, try to refine the output once
                if metrics.get("overall_score", 0) < 0.7:
                    logger.info("Similarity for %s is low (%.2f). Attempting refinement...", query_id, metrics["overall_score"])
                    
                    feedback_prompt = f"The generated C++ code has some discrepancies with the source logic (Similarity Score: {metrics['overall_score']:.2f}). "
                    feedback_prompt += "Please review the following observations and provide a more accurate C++ implementation:\n"
                    if metrics["text_similarity"] < 0.8:
                        feedback_prompt += "- Ensure strict structural compliance with the pseudo-code.\n"
                    if metrics["cfg_similarity"] < 0.9:
                        feedback_prompt += "- Double-check control flow branches and logic paths.\n"
                    
                    refined_query = f"{query_text}\n\n[FEEDBACK]\n{feedback_prompt}"
                    
                    refined_output = model.generate(refined_query)
                    refined_metrics = run_similarity_analysis(refined_output.response_text, query_text, context=context)
                    
                    # If refinement improved the result, use it
                    if refined_metrics["overall_score"] > metrics["overall_score"]:
                        logger.info("Refinement improved score from %.2f to %.2f", metrics["overall_score"], refined_metrics["overall_score"])
                        output = refined_output
                        metrics = refined_metrics

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
                metrics=metrics,
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
