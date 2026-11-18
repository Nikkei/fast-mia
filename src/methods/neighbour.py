# Copyright (c) 2026 Nikkei Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
from heapq import nlargest
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class NeighbourMethod(BaseMethod):
    """Neighbourhood comparison membership inference method.

    Calibrates the target sample's loss by subtracting the average loss of
    synthetically generated neighbour texts. Neighbours are produced by a
    masked language model (e.g. BERT) using single-word replacements with
    strong embedding dropout, following Mattern et al. (ACL Findings 2023).
    """

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        """Initialize Neighbour method

        Args:
            method_config: Method configuration
                - search_model_id: Hugging Face masked LM ID
                - num_neighbours: Maximum neighbours generated per text
                - dropout: Embedding dropout rate applied to target position
                - top_k: Top replacement candidates inspected per position
                - max_length: Maximum token length for masked LM
                - batch_size: Number of masked positions evaluated per forward
                  pass (default 32)
                - pad_to_batch_size: Pad final minibatch to ``batch_size`` so
                  XLA/TPU avoids tail-shape graph recompilation (default None,
                  which enables padding automatically on XLA devices)

        Raises:
            ValueError: If ``batch_size`` is less than 1
        """
        super().__init__("neighbour", method_config)
        self.search_model_id = self.method_config.get(
            "search_model_id", "bert-base-uncased"
        )
        self.num_neighbours = int(self.method_config.get("num_neighbours", 100))
        self.dropout = float(self.method_config.get("dropout", 0.7))
        self.top_k = int(self.method_config.get("top_k", 10))
        self.max_length = int(self.method_config.get("max_length", 512))
        self.batch_size = int(self.method_config.get("batch_size", 32))
        if self.batch_size < 1:
            raise ValueError("NeighbourMethod requires 'batch_size' >= 1.")
        raw_pad = self.method_config.get("pad_to_batch_size")
        self.pad_to_batch_size: bool | None = (
            bool(raw_pad) if raw_pad is not None else None
        )

    def process_output(self, output: RequestOutput) -> float:
        """Process model output and calculate mean log-likelihood

        Args:
            output: Model output

        Returns:
            Mean token log-probability (higher for likely sequences)
        """
        token_log_probs = self._extract_token_log_probs(output)
        return np.mean(token_log_probs)

    @staticmethod
    def _get_embeddings_module(search_model: "PreTrainedModel") -> torch.nn.Module:
        """Locate the input-embeddings submodule of a masked LM backbone.

        Args:
            search_model: Hugging Face masked LM

        Returns:
            The embeddings module exposing ``embeddings(input_ids)``
        """
        backbone = getattr(search_model, search_model.base_model_prefix, None)
        embeddings = getattr(backbone, "embeddings", None)
        if embeddings is None:
            raise ValueError(
                f"Could not locate the embeddings module for search model "
                f"'{search_model.__class__.__name__}'."
            )
        return embeddings

    def _should_pad_batch(self, device: torch.device) -> bool:
        """Determine whether tail minibatches should be padded to batch_size."""
        if self.pad_to_batch_size is not None:
            return self.pad_to_batch_size
        return getattr(device, "type", "") == "xla"

    def _generate_neighbours(
        self,
        text: str,
        search_model: "PreTrainedModel",
        search_tokenizer: "PreTrainedTokenizerBase",
        device: torch.device,
    ) -> tuple[str, list[str]]:
        """Generate single-word-replacement neighbours for a text using batched MLM.

        Faithful port of attack.py's ``generate_neighbours_alt`` with batched
        masked-position evaluation: instead of issuing ``seq_len - 1`` separate
        batch-size-1 forward passes, masked positions are evaluated in
        minibatches of ``self.batch_size`` (optionally padded to fixed shape on
        XLA/TPU to prevent tail graph recompilation).

        Args:
            text: Original text
            search_model: Masked LM used to propose replacements
            search_tokenizer: Tokenizer matching ``search_model``
            device: Device on which to run the masked LM

        Returns:
            Tuple of the search-tokenizer-normalised original text and up to
            ``num_neighbours`` neighbour texts, ranked by swap score
        """
        token_dropout = torch.nn.Dropout(p=self.dropout)
        embeddings = self._get_embeddings_module(search_model)

        ids = search_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.to(device)

        orig_dec = search_tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        seq_len = int(ids.shape[1])
        if seq_len <= 1:
            return orig_dec, []

        base_embeds = embeddings(ids)
        # (position, candidate_token_id) -> replacement score
        replacements: dict[tuple[int, int], float] = {}

        # Skip the leading special token (e.g. [CLS]) and batch positions.
        positions = list(range(1, seq_len))
        pad_batch = self._should_pad_batch(device)

        for start in range(0, len(positions), self.batch_size):
            chunk_positions = positions[start : start + self.batch_size]
            chunk_len = len(chunk_positions)
            batch_rows = (
                self.batch_size
                if (pad_batch and chunk_len < self.batch_size)
                else chunk_len
            )

            batch_embeds = base_embeds.repeat(batch_rows, 1, 1)
            for row_idx, pos in enumerate(chunk_positions):
                batch_embeds[row_idx, pos, :] = token_dropout(base_embeds[0, pos, :])

            logits = search_model(inputs_embeds=batch_embeds).logits
            pos_logits = torch.stack(
                [
                    logits[row_idx, pos, :]
                    for row_idx, pos in enumerate(chunk_positions)
                ],
                dim=0,
            )
            pos_probs = torch.softmax(pos_logits, dim=-1)
            effective_k = min(self.top_k, int(pos_probs.shape[-1]))
            top_probs_batch, top_cands_batch = torch.topk(
                pos_probs, effective_k, dim=-1
            )

            target_tokens = [int(ids[0, pos].item()) for pos in chunk_positions]
            orig_probs = [
                float(pos_probs[row_idx, tok].item())
                for row_idx, tok in enumerate(target_tokens)
            ]
            top_probs_list = top_probs_batch.detach().cpu().tolist()
            top_cands_list = top_cands_batch.detach().cpu().tolist()

            for pos, target_tok, orig_prob, row_cands, row_probs in zip(
                chunk_positions,
                target_tokens,
                orig_probs,
                top_cands_list,
                top_probs_list,
                strict=True,
            ):
                # Faithful to attack.py: a degenerate original_prob == 1 is
                # floored to 0.1 to avoid a division by zero.
                denom = 1.0 - 0.9 if orig_prob == 1.0 else 1.0 - orig_prob
                for cand, prob in zip(row_cands, row_probs, strict=True):
                    cand_id = int(cand)
                    if cand_id == target_tok:
                        continue
                    replacements[(pos, cand_id)] = float(prob) / denom

        # Keep the highest-scored swaps and decode all neighbours in one call.
        top_keys = nlargest(self.num_neighbours, replacements, key=replacements.get)
        if top_keys:
            alt_rows = [
                torch.cat(
                    (ids[:, :i], ids.new_tensor([[cand]]), ids[:, i + 1 :]), dim=1
                )
                for i, cand in top_keys
            ]
            neighbours = list(
                search_tokenizer.batch_decode(
                    torch.cat(alt_rows, dim=0), skip_special_tokens=True
                )
            )
        else:
            neighbours = []

        return orig_dec, neighbours

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        """Neighbour algorithm to calculate scores for a list of texts

        Args:
            texts: List of texts
            model: LLM model (target)
            sampling_params: Sampling parameters
            lora_request: LoRA request
            data_config: Data configuration

        Returns:
            List of Neighbour scores (higher for training members)
        """
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading search model for Neighbour: {self.search_model_id}")
        search_tokenizer = AutoTokenizer.from_pretrained(self.search_model_id)
        search_model = AutoModelForMaskedLM.from_pretrained(self.search_model_id)
        search_model = search_model.to(device).eval()

        # Generate neighbours for every text up front. Each call also returns
        # the search-tokenizer-normalised original text, so the original and
        # its neighbours are scored through the same tokenisation (matching
        # attack.py's main loop, which scores ``orig_dec`` rather than the raw
        # text).
        with torch.no_grad():
            logging.info("Generating neighbours")
            decoded = [
                self._generate_neighbours(text, search_model, search_tokenizer, device)
                for text in tqdm(texts)
            ]
        orig_texts = [orig_dec for orig_dec, _ in decoded]
        neighbours_per_text = [nbrs for _, nbrs in decoded]

        # Release the search model before invoking the vLLM target model.
        del search_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Mean log-likelihood of the original texts under the target model.
        orig_outputs = self.get_outputs(
            orig_texts, model, sampling_params, lora_request, data_config
        )
        orig_scores = [self.process_output(output) for output in orig_outputs]

        # Mean log-likelihood of all neighbours in a single batched call.
        flat_neighbours = [n for nbrs in neighbours_per_text for n in nbrs]
        neighbour_scores: list[float] = []
        if flat_neighbours:
            nbr_outputs = self.get_outputs(
                flat_neighbours, model, sampling_params, lora_request, data_config
            )
            neighbour_scores = [self.process_output(output) for output in nbr_outputs]

        scores = []
        cursor = 0
        for orig_score, nbrs in zip(orig_scores, neighbours_per_text, strict=True):
            count = len(nbrs)
            if count == 0:
                # No valid neighbours: fall back to the uncalibrated original score.
                scores.append(orig_score)
                continue
            mean_neighbour = np.mean(neighbour_scores[cursor : cursor + count])
            cursor += count
            scores.append(orig_score - mean_neighbour)

        return scores
