"""Wrapper around Qwen3-8B-Instruct that captures all-layer hidden states."""

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Container for a single generation result."""

    response_text: str
    generated_token_ids: torch.Tensor  # (num_tokens,)
    hidden_states: torch.Tensor  # (num_tokens, num_layers, hidden_dim)


class Qwen3Wrapper:
    """Load Qwen3-8B-Instruct and generate responses with hidden-state capture."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        logger.info("Loading tokenizer from %s …", cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )

        logger.info("Loading model from %s (dtype=%s) …", cfg.model_name, cfg.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.model_torch_dtype,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, query: str) -> GenerationOutput:
        """Generate a response and extract hidden states via two-pass approach.

        Pass 1 – Fast autoregressive generation (full KV-cache, no hidden-state overhead).
        Pass 2 – Single forward pass on the complete sequence to capture all-layer hidden states.
        """
        input_ids, attention_mask = self._prepare_input(query)
        prompt_len = input_ids.shape[1]

        # ── Pass 1: generate (fast) ───────────────────────────────────
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        full_seq = outputs[0]  # (total_len,)
        gen_ids = full_seq[prompt_len:]
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # ── Pass 2: single forward pass for hidden states ─────────────
        hidden_states = self._extract_hidden_states(full_seq.unsqueeze(0), prompt_len)

        return GenerationOutput(
            response_text=response_text,
            generated_token_ids=gen_ids.cpu(),
            hidden_states=hidden_states,
        )

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _prepare_input(self, query: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Format the query with the chat template and tokenize."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
        except TypeError:
            # Fallback if tokenizer template doesn't accept enable_thinking
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        encoded = self.tokenizer(
            text, return_tensors="pt", padding=False, truncation=True
        )
        device = self.model.device
        return encoded.input_ids.to(device), encoded.attention_mask.to(device)

    def _extract_hidden_states(
        self, full_seq: torch.Tensor, prompt_len: int
    ) -> torch.Tensor:
        """Run a single forward pass on the complete sequence and return
        hidden states for the *generated* tokens only.

        Args:
            full_seq: (1, total_len) – prompt + generated token ids.
            prompt_len: number of prompt tokens to skip.

        Returns:
            Tensor of shape ``(num_generated_tokens, num_layers + 1, hidden_dim)``
            stored in ``cfg.save_torch_dtype`` on CPU.
        """
        fwd_out = self.model(
            input_ids=full_seq,
            output_hidden_states=True,
        )

        # fwd_out.hidden_states: tuple of (num_layers+1) tensors,
        # each of shape (1, total_len, hidden_dim)
        # Stack → (num_layers+1, num_gen_tokens, hidden_dim), then transpose
        all_layers = torch.stack(
            [layer[0, prompt_len:, :] for layer in fwd_out.hidden_states]
        )  # (num_layers+1, num_gen_tokens, hidden_dim)

        # Transpose to (num_gen_tokens, num_layers+1, hidden_dim)
        result = all_layers.transpose(0, 1).contiguous()

        return result.to(dtype=self.cfg.save_torch_dtype, device="cpu")
