from types import SimpleNamespace
from unittest import mock

import pytest

from src.methods.neighbour import NeighbourMethod


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


class TestNeighbourInit:
    def test_defaults(self):
        method = NeighbourMethod({})
        assert method.search_model_id == "bert-base-uncased"
        assert method.num_neighbours == 100
        assert method.dropout == 0.7


class TestNeighbourProcessOutput:
    def test_returns_mean_log_prob(self):
        method = NeighbourMethod({})
        assert method.process_output(make_output([-2.0, -4.0])) == -3.0


class TestGetEmbeddingsModule:
    def test_returns_embeddings_of_backbone(self):
        embeddings = object()
        backbone = SimpleNamespace(embeddings=embeddings)
        search_model = SimpleNamespace(base_model_prefix="bert", bert=backbone)
        assert NeighbourMethod._get_embeddings_module(search_model) is embeddings

    def test_raises_when_embeddings_missing(self):
        # base_model_prefix points to a missing attribute -> backbone is None.
        search_model = SimpleNamespace(base_model_prefix="bert")
        with pytest.raises(ValueError, match="embeddings module"):
            NeighbourMethod._get_embeddings_module(search_model)


class TestNeighbourRun:
    def _patch_transformers(self):
        # run() imports these lazily; both are mocked away.
        return mock.patch.dict(
            "sys.modules",
            {"transformers": mock.MagicMock()},
        )

    def test_score_is_original_minus_mean_neighbour(self):
        method = NeighbourMethod({})
        texts = ["hello world"]

        orig_output = make_output([-2.0])  # mean = -2.0
        # Two neighbours with means -4.0 and -6.0 -> average -5.0
        nbr_outputs = [make_output([-4.0]), make_output([-6.0])]

        def fake_get_outputs(input_texts, *args, **kwargs):
            if len(input_texts) == 1:
                return [orig_output]
            return nbr_outputs

        with (
            self._patch_transformers(),
            mock.patch.object(
                method,
                "_generate_neighbours",
                return_value=("hello world", ["hallo world", "hello earth"]),
            ),
            mock.patch.object(method, "get_outputs", side_effect=fake_get_outputs),
        ):
            scores = method.run(texts, mock.MagicMock(), mock.MagicMock())

        # -2.0 - mean([-4.0, -6.0]) = -2.0 - (-5.0) = 3.0
        assert len(scores) == 1
        assert abs(scores[0] - 3.0) < 1e-6

    def test_falls_back_to_original_when_no_neighbours(self):
        method = NeighbourMethod({})
        texts = ["hello world"]
        orig_output = make_output([-2.5])

        with (
            self._patch_transformers(),
            mock.patch.object(
                method, "_generate_neighbours", return_value=("hello world", [])
            ),
            mock.patch.object(method, "get_outputs", return_value=[orig_output]),
        ):
            scores = method.run(texts, mock.MagicMock(), mock.MagicMock())

        # No neighbours: uncalibrated original score is returned.
        assert scores == [-2.5]
