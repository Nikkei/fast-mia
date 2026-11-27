import math
from unittest import mock

import pytest

from src.methods.base import BaseMethod
from src.methods.minkpp import MinKPPMethod


class DummySamplingParams:
    def __init__(self, prompt_logprobs=0, max_tokens=1, temperature=0.0):
        self.prompt_logprobs = prompt_logprobs
        self.max_tokens = max_tokens
        self.temperature = temperature


def make_topk_output(token_ids, position_candidates):
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


class TestMinKPPInit:
    def test_defaults(self):
        method = MinKPPMethod()
        assert method.method_name == "minkpp_0.2"
        assert method.ratio == 0.2
        assert method.top_k == 20
        assert method.eps == 1e-6

    @pytest.mark.parametrize(
        "bad_params",
        [
            {"ratio": 0.0},
            {"ratio": 1.2},
            {"top_k": 1},
            {"eps": 0.0},
        ],
    )
    def test_invalid_params_raise(self, bad_params):
        with pytest.raises(ValueError):
            MinKPPMethod(bad_params)


class TestMinKPPScoring:
    def test_standardized_score_matches_manual_calculation(self):
        # Two candidates with probabilities 0.8 and 0.2
        lp_hi = math.log(0.8)
        lp_lo = math.log(0.2)
        mu = 0.8 * lp_hi + 0.2 * lp_lo
        sigma = math.sqrt(0.8 * (lp_hi - mu) ** 2 + 0.2 * (lp_lo - mu) ** 2)

        member_out = make_topk_output([0, 1], [{1: lp_hi, 2: lp_lo}])
        nonmember_out = make_topk_output([0, 2], [{1: lp_hi, 2: lp_lo}])

        method = MinKPPMethod({"ratio": 1.0, "top_k": 2})
        assert method.process_output(member_out) == pytest.approx((lp_hi - mu) / sigma)
        assert method.process_output(nonmember_out) == pytest.approx(
            (lp_lo - mu) / sigma
        )
        assert method.process_output(member_out) > method.process_output(nonmember_out)

    def test_run_upgrades_prompt_logprobs_without_mutating_caller_params(self):
        BaseMethod.clear_cache()
        model = mock.MagicMock()
        model.model_config.model = "dummy-model"
        caller_params = DummySamplingParams(prompt_logprobs=0)

        out = make_topk_output(
            [0, 1], [{1: math.log(0.7), 2: math.log(0.2), 3: math.log(0.1)}]
        )
        model.generate.return_value = [out]

        method = MinKPPMethod({"ratio": 1.0, "top_k": 15})
        scores = method.run(["sample text"], model, caller_params)

        assert len(scores) == 1
        # Caller's SamplingParams remains untouched for other prompt_logprobs=0 methods
        assert caller_params.prompt_logprobs == 0
        passed_params = model.generate.call_args[0][1]
        assert passed_params.prompt_logprobs == 15

    def test_preserves_sampling_params_when_prompt_logprobs_already_sufficient(self):
        method = MinKPPMethod({"top_k": 10})
        params = DummySamplingParams(prompt_logprobs=25)
        assert method._ensure_top_k_sampling_params(params) is params

    def test_empty_and_single_candidate_fallback(self):
        method = MinKPPMethod()
        empty_out = make_topk_output([0], [])
        single_cand_out = make_topk_output([0, 1], [{1: -1.25}])
        assert method.process_output(empty_out) == 0.0
        assert method.process_output(single_cand_out) == pytest.approx(-1.25)

    def test_benchmark_config_loads_cleanly(self):
        from pathlib import Path

        from src.config import Config
        from src.methods.factory import MethodFactory

        cfg_path = (
            Path(__file__).resolve().parents[2] / "config" / "benchmark_minkpp.yaml"
        )
        config = Config(cfg_path)
        methods = MethodFactory.create_methods(config.methods)
        assert any(isinstance(m, MinKPPMethod) for m in methods)

