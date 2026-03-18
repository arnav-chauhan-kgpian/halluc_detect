"""Entry point – run the hallucination-detection data-generation pipeline."""

import argparse
import logging
from pathlib import Path

from config import PipelineConfig
from pipeline import GenerationPipeline
from utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Qwen3-8B responses for WildChat queries."
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id")
    p.add_argument("--max-queries", type=int, default=5000, help="Max queries to process")
    p.add_argument("--max-new-tokens", type=int, default=500, help="Max generation length")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Where to write results")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--no-sample", action="store_true", help="Use greedy decoding")
    p.add_argument("--enable-thinking", action="store_true", help="Enable Qwen3 thinking mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--language", default="English", help="WildChat language filter")
    p.add_argument(
        "--categories",
        nargs="+",
        default=["url", "citation", "coding", "factual"],
        help="Query categories to include",
    )
    p.add_argument("--data-path", type=Path, help="Path to local JSONL dataset")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    set_seed(args.seed)

    cfg = PipelineConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.dtype,
        device_map="auto",
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        dataset_name="allenai/WildChat-4.8M-Full",
        dataset_split="train",
        max_queries=args.max_queries,
        language_filter=args.language,
        categories=args.categories,
        output_dir=args.output_dir,
        batch_size=1,
        seed=args.seed,
        data_path=args.data_path,
    )

    GenerationPipeline(cfg).run()


if __name__ == "__main__":
    main()
