"""Pipeline configuration for hallucination detection data generation."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class PipelineConfig:
    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3-8B"
    max_prompt_tokens: int = 4096
    max_new_tokens: int = 500
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    enable_thinking: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # ── Dataset ────────────────────────────────────────────────────────
    dataset_name: str = "allenai/WildChat-4.8M-Full"
    dataset_split: str = "train"
    max_queries: int = 5000
    language_filter: str = "English"
    categories: list[str] = field(
        default_factory=lambda: ["url", "citation", "coding", "factual"]
    )

    # ── Storage ────────────────────────────────────────────────────────
    output_dir: Path = Path("output")
    save_dtype: str = "float16"  # dtype for saved hidden-state tensors

    # ── Processing ─────────────────────────────────────────────────────
    batch_size: int = 1  # keep at 1 – all-layer hidden states are memory-heavy
    seed: int = 42
    num_proc: int = 4

    # ── Derived (set in __post_init__) ─────────────────────────────────
    hidden_states_dir: Path = field(init=False)

    _DTYPE_MAP: dict = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.hidden_states_dir = self.output_dir / "hidden_states"
        self._DTYPE_MAP = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

    @property
    def model_torch_dtype(self) -> torch.dtype:
        return self._DTYPE_MAP.get(self.torch_dtype, torch.bfloat16)

    @property
    def save_torch_dtype(self) -> torch.dtype:
        return self._DTYPE_MAP.get(self.save_dtype, torch.float16)
