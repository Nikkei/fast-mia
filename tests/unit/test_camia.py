from unittest import mock

import numpy as np
import pytest

from src.methods.base import BaseMethod
from src.methods.camia import CAMIAMethod, _empirical_cdf


def make_output(token_ids, token_log_probs):
    output = mock.MagicMock()
    output.prompt_token_ids = list(token_ids)
    prompt_logprobs = [None]
    for token_id, lp in zip(token_ids[1:], token_log_probs):
        entry = mock.MagicMock()
        entry.logprob = lp
        prompt_logprobs.append({token_id: entry})
    output.prompt_logprobs = prompt_logprobs
    return output


class TestCAMIAInit:
    def test_defaults(self):
        method = CAMIAMethod()
        assert method.method_name == "camia"
        assert method.signal == "combined"
        assert method.aggregation == "edgington"
        assert method.diversity_calibration is True

    def test_signal_in_method_name(self):
        method = CAMIAMethod({"signal": "slope"})
        assert method.method_name == "camia_slope"

    def test_invalid_signal_raises(self):
        with pytest.raises(ValueError, match="Unsupported CAMIA signal"):
            CAMIAMethod({"signal": "invalid"})

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Unsupported CAMIA aggregation"):
            CAMIAMethod({"aggregation": "invalid"})


class TestCAMIAScoring:
    def test_empirical_cdf_handles_ties(self):
        cdfs = _empirical_cdf(np.array([10.0, 20.0, 20.0, 30.0]))
        assert cdfs.tolist() == pytest.approx([0.25, 0.625, 0.625, 1.0])

    def test_decreasing_nll_ranks_higher_than_increasing_nll(self):
        # Memorized text transitions from higher initial NLL to very low NLL
        member_out = make_output([1, 2, 3, 4, 5], [-3.0, -1.5, -0.4, -0.1])
        nonmember_out = make_output([1, 2, 3, 4, 5], [-0.5, -1.5, -3.0, -4.5])

        slope_method = CAMIAMethod({"signal": "slope", "diversity_calibration": False})
        assert slope_method.process_output(member_out) > 0.0
        assert slope_method.process_output(nonmember_out) < 0.0

        cbpm_method = CAMIAMethod({"signal": "cbpm", "diversity_calibration": False})
        assert cbpm_method.process_output(member_out) == pytest.approx(1.0)
        assert cbpm_method.process_output(nonmember_out) == pytest.approx(0.0)

    def test_diversity_calibration_penalizes_repetition(self):
        log_probs = [-2.0, -1.0, -0.5, -0.2]
        diverse_out = make_output([10, 11, 12, 13, 14], log_probs)
        repetitive_out = make_output([10, 11, 11, 11, 11], log_probs)

        method = CAMIAMethod({"signal": "combined"})
        assert method.process_output(diverse_out) > method.process_output(
            repetitive_out
        )

    def test_run_reuses_shared_cache_and_edgington_ranks_member_above_nonmember(self):
        BaseMethod.clear_cache()
        model = mock.MagicMock()
        model.model_config.model = "dummy-model"
        sampling_params = mock.MagicMock()
        sampling_params.prompt_logprobs = 0

        member_out = make_output([1, 2, 3, 4, 5], [-2.5, -1.0, -0.3, -0.1])
        nonmember_out = make_output([1, 6, 7, 8, 9], [-2.0, -2.5, -3.5, -4.0])
        model.generate.return_value = [member_out, nonmember_out]

        method = CAMIAMethod()
        scores = method.run(["member text", "nonmember text"], model, sampling_params)

        assert len(scores) == 2
        assert scores[0] > scores[1]
        assert model.generate.call_count == 1

    def test_empty_and_single_token_outputs(self):
        method = CAMIAMethod()
        empty_out = make_output([1], [])
        single_out = make_output([1, 2], [-0.5])
        assert method.process_output(empty_out) == 0.0
        assert isinstance(method.process_output(single_out), float)

    def test_benchmark_config_loads_cleanly(self):
        from pathlib import Path

        from src.config import Config
        from src.methods.factory import MethodFactory

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "benchmark_camia.yaml"
        config = Config(cfg_path)
        methods = MethodFactory.create_methods(config.methods)
        assert any(isinstance(m, CAMIAMethod) for m in methods)

