import math
from unittest import mock

import numpy as np
import pytest

from src.methods.aeca import AECAMethod
from src.methods.dcpdd import DCPDDMethod


def make_output(token_ids, token_log_probs, skip_first=True):
    """Build a mock RequestOutput.

    vLLM reports ``None`` for the first prompt position, so by default the
    output carries one more token ID than it carries log probabilities.
    """
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


class TestAECAInit:
    def test_defaults(self):
        method = AECAMethod()
        assert method.method_name == "aeca"
        assert method.lambda_coef == 1.0
        assert method.alpha == 1.0
        assert method.file_num == 15
        assert method.max_token_length == 1024
        assert method.default_i_ref == 0.5

    def test_custom_params(self):
        method = AECAMethod({"lambda_coef": 3.0, "alpha": 0.5, "file_num": 2})
        assert method.lambda_coef == 3.0
        assert method.alpha == 0.5
        assert method.file_num == 2

    @pytest.mark.parametrize("alpha", [0, -1.0])
    def test_non_positive_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="positive 'alpha'"):
            AECAMethod({"alpha": alpha})


class TestAECACachePath:
    def test_shares_cache_file_with_dcpdd(self):
        params = {"file_num": 10, "max_token_length": 128}
        aeca_path = AECAMethod(params)._freq_dist_cache_path("facebook/opt-125m")
        dcpdd_path = DCPDDMethod(params)._freq_dist_cache_path("facebook/opt-125m")
        assert aeca_path == dcpdd_path
        assert aeca_path.name == "freq_dist_facebook--opt-125m_10_128.json"


class TestAECABuildIRefTable:
    def test_laplace_smoothed_self_information(self):
        # counts = [1, 3], alpha = 1 -> N + alpha * |V| = 4 + 2 = 6
        # I_self = -log((c + 1) / 6) = [log(3), log(1.5)]
        table = AECAMethod()._build_i_ref_table([1, 3])
        assert table == pytest.approx([math.log(3.0), math.log(1.5)])

    def test_rare_tokens_carry_more_self_information(self):
        table = AECAMethod()._build_i_ref_table([1, 100])
        assert table[0] > table[1]


class TestAECAProcessOutput:
    def test_matches_hand_computed_score(self):
        method = AECAMethod({"lambda_coef": 1.0})
        i_ref_table = np.array([1.0, 2.0, 4.0])
        log_probs = [math.log(0.5), math.log(0.25), math.log(0.5)]
        # Phi = p * I_self = [0.5, 0.5, 2.0]
        # S[t] = Phi(t) - Phi(t + 1), last element keeps Phi(T)
        expected = np.std([0.0, -1.5, 2.0]) - np.std([-lp for lp in log_probs])

        output = make_output([0, 0, 1, 2], log_probs)
        assert method.process_output(output, i_ref_table) == pytest.approx(expected)

    def test_lambda_scales_the_nll_volatility_term(self):
        i_ref_table = np.array([1.0, 2.0, 4.0])
        log_probs = [math.log(0.5), math.log(0.25), math.log(0.5)]
        output = make_output([0, 0, 1, 2], log_probs)

        score_1 = AECAMethod({"lambda_coef": 1.0}).process_output(output, i_ref_table)
        score_3 = AECAMethod({"lambda_coef": 3.0}).process_output(output, i_ref_table)
        nll_std = np.std([-lp for lp in log_probs])
        assert score_1 - score_3 == pytest.approx(2 * nll_std)

    def test_token_ids_are_aligned_to_the_tail(self):
        """The leading token, whose logprob is None, must not shift the IDs."""
        method = AECAMethod()
        i_ref_table = np.array([1.0, 2.0, 4.0])
        log_probs = [math.log(0.5), math.log(0.25), math.log(0.5)]

        # The first ID is only consumed by the dropped position, so changing it
        # must not change the score.
        score_a = method.process_output(
            make_output([0, 0, 1, 2], log_probs), i_ref_table
        )
        score_b = method.process_output(
            make_output([2, 0, 1, 2], log_probs), i_ref_table
        )
        assert score_a == pytest.approx(score_b)

    def test_out_of_range_token_uses_default_i_ref(self):
        method = AECAMethod({"default_i_ref": 7.0})
        i_ref_table = np.array([1.0, 2.0])
        log_probs = [math.log(0.5), math.log(0.5)]
        # Second token ID (99) is outside the reference vocabulary.
        # Phi = [0.5 * 1.0, 0.5 * 7.0] = [0.5, 3.5]
        expected = np.std([-3.0, 3.5]) - np.std([-lp for lp in log_probs])

        output = make_output([0, 0, 99], log_probs)
        assert method.process_output(output, i_ref_table) == pytest.approx(expected)

    def test_empty_log_probs_returns_zero(self):
        method = AECAMethod()
        output = make_output([0], [])
        assert method.process_output(output, np.array([1.0])) == 0.0

    def test_single_token_has_zero_volatility_penalty(self):
        method = AECAMethod()
        i_ref_table = np.array([2.0])
        output = make_output([0, 0], [math.log(0.5)])
        # A single position: std of both streams is 0.
        assert method.process_output(output, i_ref_table) == pytest.approx(0.0)
