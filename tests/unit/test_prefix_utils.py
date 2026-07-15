from unittest import mock

from src.methods.prefix_utils import process_prefix


def make_model(max_model_len):
    model = mock.MagicMock()
    model.llm_engine.model_config.max_model_len = max_model_len
    return model


def make_tokenizer(tokens_per_shot):
    tokenizer = mock.MagicMock()
    tokenizer.encode.side_effect = lambda text: [0] * tokens_per_shot
    return tokenizer


class TestProcessPrefix:
    def test_pass_window_skips_check(self):
        prefix = ["shot1", "shot2"]
        result, num_shots = process_prefix(
            make_model(1), make_tokenizer(100), prefix,
            avg_length=100, pass_window=True, num_shots=2,
        )
        assert result == prefix
        assert num_shots == 2

    def test_all_shots_fit(self):
        prefix = ["shot1", "shot2"]
        # 2 shots * 10 tokens + avg_length 10 = 30 <= 100
        result, num_shots = process_prefix(
            make_model(100), make_tokenizer(10), prefix,
            avg_length=10, pass_window=False, num_shots=2,
        )
        assert result == prefix
        assert num_shots == 2

    def test_truncates_to_fitting_shots(self):
        prefix = ["shot1", "shot2", "shot3"]
        # avg_length 10 + 1 shot * 10 = 20 <= 25, 2 shots would be 30 > 25
        result, num_shots = process_prefix(
            make_model(25), make_tokenizer(10), prefix,
            avg_length=10, pass_window=False, num_shots=3,
        )
        assert num_shots == 1
        assert result == ["shot3"]

    def test_no_shot_fits_returns_empty(self):
        prefix = ["shot1", "shot2", "shot3"]
        # avg_length alone already exceeds max_model_len: no shot fits.
        # prefix[-0:] would wrongly return the whole prefix here.
        result, num_shots = process_prefix(
            make_model(50), make_tokenizer(100), prefix,
            avg_length=60, pass_window=False, num_shots=3,
        )
        assert num_shots == 0
        assert result == []
