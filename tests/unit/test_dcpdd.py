import math
from unittest import mock

import numpy as np
import pytest

from src.methods.dcpdd import DCPDDMethod


def make_output(token_ids, token_log_probs, skip_first=True):
    """Build a mock RequestOutput matching vLLM's prompt_logprobs layout."""
    output = mock.MagicMock()
    output.prompt_token_ids = list(token_ids)
    prompt_logprobs = []
    log_probs = list(token_log_probs)
    if skip_first:
        prompt_logprobs.append(None)
        assert len(token_ids) == len(log_probs) + 1
    for token_id, value in zip(token_ids[len(prompt_logprobs) :], log_probs):
        entry = mock.MagicMock()
        entry.logprob = value
        prompt_logprobs.append({token_id: entry})
    output.prompt_logprobs = prompt_logprobs
    return output


class TestFreqDistCachePath:
    def test_includes_model_id_file_num_and_max_token_length(self):
        method = DCPDDMethod({"file_num": 10, "max_token_length": 128})
        path = method._freq_dist_cache_path("facebook/opt-125m")
        assert path.name == "freq_dist_facebook--opt-125m_10_128.json"

    def test_different_models_use_different_caches(self):
        method = DCPDDMethod({"file_num": 15})
        path_a = method._freq_dist_cache_path("org/model-a")
        path_b = method._freq_dist_cache_path("org/model-b")
        assert path_a != path_b

    def test_different_max_token_length_use_different_caches(self):
        method_a = DCPDDMethod({"file_num": 15, "max_token_length": 128})
        method_b = DCPDDMethod({"file_num": 15, "max_token_length": 1024})
        path_a = method_a._freq_dist_cache_path("org/model")
        path_b = method_b._freq_dist_cache_path("org/model")
        assert path_a != path_b

    def test_unknown_model_id(self):
        method = DCPDDMethod({"file_num": 15, "max_token_length": 1024})
        path = method._freq_dist_cache_path("")
        assert path.name == "freq_dist_unknown-model_15_1024.json"


class TestDCPDDProcessOutput:
    def test_aligns_tail_token_ids_for_bos_and_bos_less_models(self):
        method = DCPDDMethod({"alpha": 100.0})
        freq_dist = [10, 2, 5]
        log_probs = [math.log(0.5), math.log(0.25)]

        # Changing the leading token (e.g. BOS vs first content token in a
        # BOS-less model) must not change the aligned tail token IDs [1, 2].
        out_with_bos = make_output([0, 1, 2], log_probs)
        out_without_bos = make_output([99, 1, 2], log_probs)

        score_bos = method.process_output(out_with_bos, freq_dist)
        score_no_bos = method.process_output(out_without_bos, freq_dist)

        eps = 1e-10
        expected_ce = np.array([0.5, 0.25]) * np.log(
            1.0 / (np.array([2.0, 5.0]) + eps)
        )
        assert score_bos == pytest.approx(-float(np.mean(expected_ce)))
        assert score_no_bos == pytest.approx(score_bos)

    def test_uses_only_first_occurrence_of_duplicate_tokens(self):
        method = DCPDDMethod({"alpha": 100.0})
        freq_dist = [1, 4]
        # Tail token IDs are [1, 1]; second occurrence (prob 0.1) is ignored.
        output = make_output([0, 1, 1], [math.log(0.5), math.log(0.1)])
        eps = 1e-10
        expected = -(0.5 * math.log(1.0 / (4.0 + eps)))
        assert method.process_output(output, freq_dist) == pytest.approx(expected)

    def test_out_of_range_token_id_uses_zero_frequency(self):
        method = DCPDDMethod({"alpha": 100.0})
        output = make_output([0, 50], [math.log(0.5)])
        score = method.process_output(output, [3, 4])
        assert math.isfinite(score)

    def test_empty_log_probs_returns_zero(self):
        method = DCPDDMethod()
        output = make_output([0], [])
        assert method.process_output(output, [1, 2]) == 0.0


class TestDCPDDRun:
    def test_run_does_not_reencode_texts_via_tokenizer(self):
        method = DCPDDMethod({"alpha": 0.01})
        tokenizer = mock.MagicMock()
        tokenizer.encode.side_effect = AssertionError(
            "DCPDD.run must use output.prompt_token_ids, not tokenizer.encode"
        )
        outputs = [make_output([5, 1, 2], [-1.0, -2.0])]

        model = mock.MagicMock()
        model.model_config.model = "test-model"
        with (
            mock.patch(
                "src.methods.dcpdd.load_or_build_freq_dist",
                return_value=[1, 10, 20, 30, 40, 50],
            ),
            mock.patch.object(method, "get_outputs", return_value=outputs),
        ):
            scores = method.run(
                ["hello world"],
                model,
                tokenizer,
                mock.MagicMock(),
            )

        assert len(scores) == 1
        assert math.isfinite(scores[0])
