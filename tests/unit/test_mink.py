from unittest import mock

from src.methods.mink import MinKMethod


def make_output(token_log_probs):
    """Build a mock RequestOutput whose prompt token IDs index their logprobs."""
    output = mock.MagicMock()
    output.prompt_token_ids = list(range(len(token_log_probs)))
    prompt_logprobs = []
    for token_id, value in enumerate(token_log_probs):
        entry = mock.MagicMock()
        entry.logprob = value
        prompt_logprobs.append({token_id: entry})
    output.prompt_logprobs = prompt_logprobs
    return output


class TestMinKInit:
    def test_default_ratio(self):
        method = MinKMethod()
        assert method.method_config["ratio"] == 0.5
        assert method.method_name == "mink_0.5"

    def test_ratio_added_when_missing(self):
        method = MinKMethod({"other": 1})
        assert method.method_config["ratio"] == 0.5

    def test_custom_ratio_in_name(self):
        method = MinKMethod({"ratio": 0.2})
        assert method.method_name == "mink_0.2"


class TestMinKProcessOutput:
    def test_averages_lowest_k_percent(self):
        method = MinKMethod({"ratio": 0.5})
        # 4 tokens, lowest 50% = [-4.0, -3.0], mean = -3.5
        output = make_output([-1.0, -2.0, -3.0, -4.0])
        assert method.process_output(output) == -3.5

    def test_at_least_one_token_kept_for_small_ratio(self):
        method = MinKMethod({"ratio": 0.01})
        # max(1, int(2 * 0.01)) = 1 -> lowest single token = -5.0
        output = make_output([-1.0, -5.0])
        assert method.process_output(output) == -5.0
