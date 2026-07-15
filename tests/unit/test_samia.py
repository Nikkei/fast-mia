from unittest import mock

from src.methods.samia import SaMIAMethod, get_prefix, get_suffix, rouge_n


class TestSuffixHelpers:
    def test_get_prefix(self):
        assert get_prefix("a b c d", 0.5) == "a b"

    def test_get_suffix(self):
        assert get_suffix("a b c d", 0.5, 4) == ["c", "d"]

    def test_rouge_n_empty(self):
        assert rouge_n([], ["a"]) == 0


class TestSaMIARun:
    def make_output(self, text, num_samples):
        sample = mock.MagicMock()
        sample.text = text
        output = mock.MagicMock()
        output.outputs = [sample] * num_samples
        return output

    def run_method(self, data_config):
        method = SaMIAMethod({"num_samples": 2})
        texts = ["one two three four five six seven eight"]
        outputs = [self.make_output("one two three four five six seven eight", 2)]
        with mock.patch.object(method, "get_outputs", return_value=outputs):
            return method.run(texts, mock.MagicMock(), data_config=data_config)

    def test_run_without_text_length_uses_default(self):
        # Missing data.text_length must not raise (round(None * 0.5) TypeError)
        scores = self.run_method(data_config={})
        assert len(scores) == 1

    def test_run_with_text_length(self):
        scores = self.run_method(data_config={"text_length": 8})
        assert len(scores) == 1
        assert scores[0] > 0
