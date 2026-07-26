from unittest import mock

import pytest

from src.methods.ref import RefMethod


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


class TestRefValidation:
    def test_missing_reference_model_rejected(self):
        with pytest.raises(ValueError, match="reference_model"):
            RefMethod({})

    def test_missing_model_id_rejected(self):
        with pytest.raises(ValueError, match="model_id"):
            RefMethod({"reference_model": {"tensor_parallel_size": 1}})

    def test_valid_config(self):
        method = RefMethod({"reference_model": {"model_id": "org/ref"}})
        assert method.ref_model_id == "org/ref"
        assert method.ref_model is None


class TestRefProcessOutput:
    def test_returns_mean_log_prob(self):
        method = RefMethod({"reference_model": {"model_id": "org/ref"}})
        assert method.process_output(make_output([-2.0, -4.0])) == -3.0


class TestRefRun:
    def test_score_is_target_loss_minus_ref_loss(self):
        method = RefMethod({"reference_model": {"model_id": "org/ref"}})
        texts = ["hello"]

        target_output = make_output([-2.0])  # mean = -2.0
        ref_output = make_output([-5.0])  # mean = -5.0

        def fake_get_outputs(input_texts, model, *args, **kwargs):
            # The reference model is the lazily created LLM instance.
            if model is method.ref_model:
                return [ref_output]
            return [target_output]

        with (
            mock.patch("src.methods.ref.LLM", return_value=mock.MagicMock()),
            mock.patch.object(method, "get_outputs", side_effect=fake_get_outputs),
            mock.patch.object(method, "cleanup_model") as cleanup,
        ):
            scores = method.run(texts, mock.MagicMock(), mock.MagicMock())

        # target_loss (-2.0) - ref_loss (-5.0) = 3.0
        assert len(scores) == 1
        assert abs(scores[0] - 3.0) < 1e-6
        # Reference model is released and reset after the run.
        cleanup.assert_called_once()
        assert method.ref_model is None
