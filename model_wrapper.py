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
        """Generate a response for *query* and return text + hidden states."""
        input_ids, attention_mask = self._prepare_input(query)
        prompt_len = input_ids.shape[1]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Decode only the newly generated tokens
        gen_ids = outputs.sequences[0, prompt_len:]
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract per-token hidden states across all layers
        hidden_states = self._extract_hidden_states(outputs.hidden_states)

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
        self, raw_hidden_states: tuple
    ) -> torch.Tensor:
        """Collapse the nested generate() hidden-state tuples into a single tensor.

        ``raw_hidden_states`` structure (from ``generate``):
        - Tuple of length ``num_generation_steps``.
        - Each element is a tuple of ``(num_layers + 1)`` tensors.
          * Step 0 (prefill):  each tensor shape ``(batch, prompt_len, hidden_dim)``
          * Step i > 0 (decode): each tensor shape ``(batch, 1, hidden_dim)``

        Returns:
            Tensor of shape ``(num_generated_tokens, num_layers + 1, hidden_dim)``
            stored in ``cfg.save_torch_dtype`` on CPU.
        """
        per_token: list[torch.Tensor] = []

        for step_states in raw_hidden_states:
            # step_states: tuple of (num_layers+1) tensors
            layers = []
            for layer_tensor in step_states:
                # Take last position along seq-len dim (works for both prefill & decode)
                token_vec = layer_tensor[0, -1, :]  # (hidden_dim,)
                layers.append(token_vec)
            # Stack across layers → (num_layers+1, hidden_dim)
            per_token.append(torch.stack(layers))

        # (num_generated_tokens, num_layers+1, hidden_dim)
        all_hidden = torch.stack(per_token)

        return all_hidden.to(dtype=self.cfg.save_torch_dtype, device="cpu")
