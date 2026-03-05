"""
Configuration for the hallucination detection ingestion pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the Qwen3 8B Instruct model."""

    model_name: str = "Qwen/Qwen3-8B"
    max_new_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"
    # Capture hidden states from all layers
    output_hidden_states: bool = True
    output_attentions: bool = False
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """Configuration for the WildChat dataset."""

    dataset_name: str = "allenai/WildChat-1M"
    split: str = "train"
    # Number of queries to process (pilot: 1K-5K)
    num_queries: int = 2000
    # Seed for reproducibility when sampling
    seed: int = 42
    # Filter for English queries only
    language_filter: str = "English"
    # Category filters aligned with project goals
    # We focus on queries likely to produce: URLs, citations, code, factual answers
    category_keywords: list[str] = field(
        default_factory=lambda: [
            "url",
            "link",
            "website",
            "cite",
            "citation",
            "reference",
            "code",
            "program",
            "script",
            "function",
            "fact",
            "explain",
            "what is",
            "who is",
            "when did",
            "how does",
            "define",
            "describe",
        ]
    )
    # Skip multi-turn conversations; use only first user message
    first_turn_only: bool = True


@dataclass
class StorageConfig:
    """Configuration for output storage (PyTorch .pt + Parquet)."""

    output_dir: Path = Path("D:/Projects/hallucination_detection/outputs")
    # Directory for hidden state .pt files (one per sample)
    hidden_states_dir: str = "hidden_states"
    # Parquet file for text data (query, response, metadata)
    parquet_filename: str = "responses.parquet"
    # Save hidden states every N samples (for checkpoint/resume)
    checkpoint_interval: int = 50


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    # Batch size for generation (adjust based on GPU memory)
    batch_size: int = 1  # 1 is safest for capturing per-sample hidden states
    # Number of dataloader workers
    num_workers: int = 2
    # Resume from a previous checkpoint
    resume_from: Optional[int] = None
    # Logging
    log_level: str = "INFO"
