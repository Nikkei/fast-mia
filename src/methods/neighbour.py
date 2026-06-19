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
        """
        super().__init__("neighbour", method_config)
        self.search_model_id = self.method_config.get(
            "search_model_id", "bert-base-uncased"
        )
        self.num_neighbours = self.method_config.get("num_neighbours", 100)
        self.dropout = self.method_config.get("dropout", 0.7)
        self.top_k = self.method_config.get("top_k", 10)
        self.max_length = self.method_config.get("max_length", 512)

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
    def _clean_special_tokens(text: str) -> str:
        """Strip BERT-style markers, mirroring attack.py's main loop.

        Args:
            text: Decoded text that may still contain ``[CLS]``/``[SEP]``

        Returns:
            Text with the leading ``[CLS]`` and trailing ``[SEP]`` removed
        """
        return text.replace(" [SEP]", " ").replace("[CLS] ", " ")

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

    def _generate_neighbours(
        self,
        text: str,
        search_model: "PreTrainedModel",
        search_tokenizer: "PreTrainedTokenizerBase",
        device: torch.device,
    ) -> tuple[str, list[str]]:
        """Generate single-word-replacement neighbours for a text.

        Faithful port of attack.py's ``generate_neighbours_alt`` (the variant
        actually used in the reference main loop): replacement candidates are
        scored per ``(position, candidate-token)`` rather than per decoded
        text, the top ``num_neighbours`` swaps are kept, and each is then
        materialised into a one-token-replacement neighbour.

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

        base_embeds = embeddings(ids)
        # (position, candidate_token_id) -> replacement score
        replacements: dict[tuple[int, int], float] = {}

        # Skip the leading special token (e.g. [CLS]).
        for i in range(1, ids.shape[1]):
            target_token = ids[0, i]
            embeds = torch.cat(
                (
                    base_embeds[:, :i, :],
                    token_dropout(base_embeds[:, i, :]).unsqueeze(dim=1),
                    base_embeds[:, i + 1 :, :],
                ),
                dim=1,
            )
            token_probs = torch.softmax(
                search_model(inputs_embeds=embeds).logits, dim=2
            )
            original_prob = token_probs[0, i, target_token]
            top_probs, top_cands = torch.topk(token_probs[0, i, :], self.top_k)

            for cand, prob in zip(top_cands, top_probs, strict=True):
                if cand == target_token:
                    continue
                # Faithful to attack.py: a degenerate original_prob == 1 is
                # floored to 0.1 to avoid a division by zero.
                if original_prob.item() == 1:
                    score = prob.item() / (1 - 0.9)
                else:
                    score = prob.item() / (1 - original_prob.item())
                replacements[(i, int(cand))] = score

        # Keep the highest-scored swaps and build one neighbour per swap.
        top_keys = nlargest(self.num_neighbours, replacements, key=replacements.get)
        neighbours = []
        for i, cand in top_keys:
            alt_ids = torch.cat(
                (ids[:, :i], ids.new_tensor([[cand]]), ids[:, i + 1 :]), dim=1
            )
            alt_text = self._clean_special_tokens(
                search_tokenizer.batch_decode(alt_ids)[0]
            )
            neighbours.append(alt_text)

        orig_dec = self._clean_special_tokens(search_tokenizer.batch_decode(ids)[0])
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
