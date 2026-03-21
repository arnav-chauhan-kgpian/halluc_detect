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
    def generate_batch(self, queries: list[str]) -> list[GenerationOutput]:
        """Generate responses for a batch of queries (Response only)."""
        input_ids, attention_mask = self._prepare_batch_input(queries)
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        # Single pass generation without capturing hidden states
        gen_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            return_dict_in_generate=True,
        )

        results = []
        for i in range(batch_size):
            full_seq = gen_outputs.sequences[i]
            gen_ids = full_seq[prompt_len:]
            response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            results.append(GenerationOutput(
                response_text=response_text,
                generated_token_ids=gen_ids.cpu(),
            ))

        return results

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _prepare_batch_input(self, queries: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Format the queries and tokenize as a batch."""
        texts = []
        for query in queries:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=self.cfg.enable_thinking,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            texts.append(text)

        # Use left-padding for generation
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        )
        device = self.model.device
        return encoded.input_ids.to(device), encoded.attention_mask.to(device)

