from unittest import mock

from src.methods.lower import LowerMethod


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


class TestLowerProcessOutput:
    def test_returns_negative_mean_log_prob(self):
        method = LowerMethod({})
        # loss = -mean([-2.0, -4.0]) = 3.0
        assert method.process_output(make_output([-2.0, -4.0])) == 3.0


class TestLowerRun:
    def test_score_is_lowercase_loss_over_original_loss(self):
        method = LowerMethod({})
        texts = ["Hello World"]

        # Original loss -> 2.0, lowercased loss -> 4.0, ratio ~= 2.0
        original_output = make_output([-2.0])
        lower_output = make_output([-4.0])

        def fake_get_outputs(input_texts, *args, **kwargs):
            # Distinguish the two internal calls by the text casing.
            if input_texts[0].islower():
                return [lower_output]
            return [original_output]

        with mock.patch.object(
            method, "get_outputs", side_effect=fake_get_outputs
        ):
            scores = method.run(texts, mock.MagicMock(), mock.MagicMock())

        assert len(scores) == 1
        assert abs(scores[0] - 2.0) < 1e-6
