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
        assert method.batch_size == 32
        assert method.pad_to_batch_size is None

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            NeighbourMethod({"batch_size": 0})

    def test_should_pad_batch_auto_enables_for_xla(self):
        method = NeighbourMethod({})
        assert method._should_pad_batch(SimpleNamespace(type="xla")) is True
        assert method._should_pad_batch(SimpleNamespace(type="cuda")) is False

        forced = NeighbourMethod({"pad_to_batch_size": True})
        assert forced._should_pad_batch(SimpleNamespace(type="cpu")) is True


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


class FakeTensor:
    """Lightweight NumPy-backed tensor to test batched _generate_neighbours without torch."""

    def __init__(self, arr):
        import numpy as np

        self.arr = np.asarray(arr)

    @property
    def shape(self):
        return self.arr.shape

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.arr.tolist()

    def item(self):
        return self.arr.item()

    def repeat(self, r0, r1, r2):
        import numpy as np

        return FakeTensor(np.tile(self.arr, (r0, r1, r2)))

    def new_tensor(self, data):
        import numpy as np

        return FakeTensor(np.asarray(data, dtype=self.arr.dtype))

    def __getitem__(self, idx):
        res = self.arr[idx]
        return FakeTensor(res)

    def __setitem__(self, idx, val):
        self.arr[idx] = val.arr if isinstance(val, FakeTensor) else val


class TestBatchedGenerateNeighbours:
    def test_batches_positions_and_pads_for_xla(self):
        import numpy as np

        # Sequence length 4 -> 3 modifiable positions (1, 2, 3).
        # With batch_size=2 and XLA device, forward passes receive batch shape (2, 4, 2)
        # on BOTH minibatches (second minibatch has 1 active position padded to 2).
        ids = FakeTensor([[101, 1, 2, 3]])
        base_embeds = FakeTensor(np.ones((1, 4, 2), dtype=np.float64))

        seen_batch_shapes = []

        def fake_forward(inputs_embeds):
            seen_batch_shapes.append(inputs_embeds.shape)
            b_rows, s_len, _ = inputs_embeds.shape
            # Vocab size 5; candidate token 4 has highest logit everywhere
            logits = np.zeros((b_rows, s_len, 5), dtype=np.float64)
            logits[:, :, 4] = 3.0
            logits[:, :, 0] = 1.0
            return SimpleNamespace(logits=FakeTensor(logits))

        backbone = SimpleNamespace(embeddings=lambda _ids: base_embeds)
        search_model = mock.MagicMock(side_effect=fake_forward)
        search_model.base_model_prefix = "bert"
        search_model.bert = backbone

        search_tokenizer = mock.MagicMock()
        search_tokenizer.return_value = SimpleNamespace(input_ids=ids)
        search_tokenizer.batch_decode.side_effect = (
            lambda tensor, skip_special_tokens=True: [
                " ".join(str(tok) for tok in row[1:]) for row in tensor.tolist()
            ]
        )

        def fake_stack(tensors, dim=0):
            return FakeTensor(np.stack([t.arr for t in tensors], axis=dim))

        def fake_softmax(tensor, dim=-1):
            shifted = tensor.arr - np.max(tensor.arr, axis=dim, keepdims=True)
            exps = np.exp(shifted)
            return FakeTensor(exps / np.sum(exps, axis=dim, keepdims=True))

        def fake_topk(tensor, k, dim=-1):
            order = np.argsort(-tensor.arr, axis=dim)[:, :k]
            vals = np.take_along_axis(tensor.arr, order, axis=dim)
            return FakeTensor(vals), FakeTensor(order)

        def fake_cat(tensors, dim=0):
            return FakeTensor(np.concatenate([t.arr for t in tensors], axis=dim))

        method = NeighbourMethod({"batch_size": 2, "num_neighbours": 3, "top_k": 2})

        with (
            mock.patch("src.methods.neighbour.torch.nn.Dropout", return_value=lambda x: x),
            mock.patch("src.methods.neighbour.torch.stack", side_effect=fake_stack),
            mock.patch("src.methods.neighbour.torch.softmax", side_effect=fake_softmax),
            mock.patch("src.methods.neighbour.torch.topk", side_effect=fake_topk),
            mock.patch("src.methods.neighbour.torch.cat", side_effect=fake_cat),
        ):
            orig_dec, nbrs = method._generate_neighbours(
                "dummy",
                search_model,
                search_tokenizer,
                SimpleNamespace(type="xla"),
            )

        assert orig_dec == "1 2 3"
        # 3 positions with batch_size=2 -> only 2 forward passes (instead of 3),
        # and both forward passes have static batch size 2 for XLA!
        assert seen_batch_shapes == [(2, 4, 2), (2, 4, 2)]
        assert len(nbrs) == 3
        assert set(nbrs) == {"4 2 3", "1 4 3", "1 2 4"}


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
