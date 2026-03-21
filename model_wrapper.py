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
    response_hidden_states: torch.Tensor # (num_tokens, num_layers+1, hidden_dim)
    query_hidden_states: torch.Tensor    # (prompt_len, num_layers+1, hidden_dim)


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
        """Generate responses for a batch of queries (stable & batched)."""
        input_ids, attention_mask = self._prepare_batch_input(queries)
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        # Pass 1: Generate response batch + response hidden states
        # We don't pass PKV here to ensure stability across library versions.
        # But we capture hidden states for the NEW tokens.
        gen_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Pass 2: Forward pass on prompt batch to get prompt hidden states
        # This is fast (parallel) and much more stable than PKV injection.
        prompt_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # 3. Post-process each sample in the batch
        results = []
        for i in range(batch_size):
            full_seq = gen_outputs.sequences[i]
            # Since we used left-padding for generation, the actual sequence might vary
            # but transformers handles the prompt_len offset correctly.
            gen_ids = full_seq[prompt_len:]
            response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Capture prompt hidden states for sample i
            # prompt_out.hidden_states is (num_layers+1, batch_size, prompt_len, hidden_dim)
            prompt_hs_i = torch.stack(prompt_out.hidden_states)[:, i, :, :]
            prompt_hs_i = prompt_hs_i.transpose(0, 1).contiguous().cpu() # (prompt_len, num_layers+1, hidden_dim)

            # Capture response hidden states for sample i
            if hasattr(gen_outputs, "hidden_states") and gen_outputs.hidden_states:
                res_hs_list = []
                for step_hs in gen_outputs.hidden_states:
                    # step_hs is a tuple of (num_layers+1) tensors of (batch_size, 1, hidden_dim)
                    step_tensor_i = torch.stack(step_hs)[:, i, 0, :] # (num_layers+1, hidden_dim)
                    res_hs_list.append(step_tensor_i)
                response_hs_i = torch.stack(res_hs_list).cpu()
            else:
                response_hs_i = torch.empty(0)

            results.append(GenerationOutput(
                response_text=response_text,
                generated_token_ids=gen_ids.cpu(),
                response_hidden_states=response_hs_i,
                query_hidden_states=prompt_hs_i,
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

