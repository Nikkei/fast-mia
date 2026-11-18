import math
from unittest import mock

import pytest

from src.methods.surp import SURPMethod


def make_single_lp_output(token_ids, token_log_probs):
    output = mock.MagicMock()
    output.prompt_token_ids = list(token_ids)
    prompt_logprobs = [None]
    for token_id, lp in zip(token_ids[1:], token_log_probs):
        entry = mock.MagicMock()
        entry.logprob = lp
        prompt_logprobs.append({token_id: entry})
    output.prompt_logprobs = prompt_logprobs
    return output


def make_topk_output(token_ids, position_candidates):
    """Build RequestOutput with top-K candidate dicts per position."""
    output = mock.MagicMock()
    output.prompt_token_ids = list(token_ids)
    prompt_logprobs = [None]
    for cand_dict in position_candidates:
        step = {}
        for tid, lp in cand_dict.items():
            entry = mock.MagicMock()
            entry.logprob = lp
            step[tid] = entry
        prompt_logprobs.append(step)
    output.prompt_logprobs = prompt_logprobs
    return output


class TestSURPInit:
    def test_defaults(self):
        method = SURPMethod()
        assert method.method_name == "surp"
        assert method.ratio == 0.2
        assert method.entropy_percentile == 50.0
        assert method.entropy_threshold is None
        assert method.window_size == 5

    @pytest.mark.parametrize(
        "bad_params",
        [
            {"ratio": 0.0},
            {"ratio": 1.5},
            {"entropy_percentile": 0.0},
            {"entropy_percentile": 120.0},
            {"entropy_threshold": -0.1},
            {"window_size": 0},
        ],
    )
    def test_invalid_params_raise(self, bad_params):
        with pytest.raises(ValueError):
            SURPMethod(bad_params)


class TestSURPScoring:
    def test_topk_mode_filters_out_high_entropy_positions(self):
        # Position 1: peaked distribution (low entropy), target token 10 matches peak
        peaked_member = {10: math.log(0.95), 20: math.log(0.05)}
        # Position 2: near-uniform distribution (high entropy), target token 11 has low prob
        ambiguous_pos = {
            11: math.log(0.10),
            21: math.log(0.30),
            22: math.log(0.30),
            23: math.log(0.30),
        }
        # Position 1 for nonmember: peaked distribution, but target token 99 is surprising!
        peaked_nonmember = {
            10: math.log(0.94),
            20: math.log(0.05),
            99: math.log(0.01),
        }

        member_out = make_topk_output([0, 10, 11], [peaked_member, ambiguous_pos])
        nonmember_out = make_topk_output([0, 99, 11], [peaked_nonmember, ambiguous_pos])

        method = SURPMethod({"ratio": 1.0, "entropy_percentile": 50.0})
        member_score = method.process_output(member_out)
        nonmember_score = method.process_output(nonmember_out)

        assert member_score == pytest.approx(math.log(0.95))
        assert nonmember_score == pytest.approx(math.log(0.01))
        assert member_score > nonmember_score

    def test_causal_window_mode_detects_surprising_drop_without_topk(self):
        # Confident context [-0.1, -0.1, -0.1] followed by a surprising spike -6.0
        nonmember_out = make_single_lp_output(
            [0, 1, 2, 3, 4], [-0.1, -0.1, -0.1, -6.0]
        )
        member_out = make_single_lp_output([0, 1, 2, 3, 4], [-0.1, -0.1, -0.1, -0.2])

        method = SURPMethod({"ratio": 0.25, "entropy_percentile": 75.0, "window_size": 2})
        assert method.process_output(member_out) > method.process_output(nonmember_out)

    def test_fallback_when_entropy_threshold_excludes_all_tokens(self):
        out = make_single_lp_output([0, 1, 2], [-1.0, -3.0])
        method = SURPMethod({"ratio": 0.5, "entropy_threshold": 0.0})
        # Falls back to all token_log_probs instead of returning NaN
        assert method.process_output(out) == pytest.approx(-3.0)

    def test_empty_output_returns_zero(self):
        out = make_single_lp_output([0], [])
        assert SURPMethod().process_output(out) == 0.0

    def test_benchmark_config_loads_cleanly(self):
        from pathlib import Path

        from src.config import Config
        from src.methods.factory import MethodFactory

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "benchmark_surp.yaml"
        config = Config(cfg_path)
        methods = MethodFactory.create_methods(config.methods)
        assert any(isinstance(m, SURPMethod) for m in methods)

