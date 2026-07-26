from types import SimpleNamespace
from unittest import mock

from src.methods.base import BaseMethod
from src.methods.loss import LossMethod


class TestExtractTokenLogProbs:
    def test_picks_logprob_of_actual_token(self):
        # With prompt_logprobs > 0 the dict also contains top-k candidates;
        # the entry for the actual prompt token must be used, not the first.
        def logprob(value):
            entry = mock.MagicMock()
            entry.logprob = value
            return entry

        output = mock.MagicMock()
        output.prompt_token_ids = [10, 20]
        output.prompt_logprobs = [
            None,  # First token has no logprob
            {5: logprob(-0.1), 20: logprob(-2.0)},  # 5 is a top-k candidate
        ]
        assert LossMethod._extract_token_log_probs(output) == [-2.0]


def make_sampling_params(**overrides):
    params = {
        "max_tokens": 1,
        "min_tokens": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
        "seed": None,
        "logprobs": None,
        "prompt_logprobs": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    params.update(overrides)
    return SimpleNamespace(**params)


class TestModelCacheKey:
    texts = ["text a", "text b"]
    model = object()  # No model_config attribute -> empty model id

    def get_key(self, sampling_params):
        return LossMethod._get_model_cache_key(
            self.texts, sampling_params, self.model
        )

    def test_same_params_same_key(self):
        key1 = self.get_key(make_sampling_params())
        key2 = self.get_key(make_sampling_params())
        assert key1 == key2

    def test_different_n_different_key(self):
        # Two SaMIA configs with different num_samples must not collide
        key1 = self.get_key(make_sampling_params(n=5))
        key2 = self.get_key(make_sampling_params(n=10))
        assert key1 != key2

    def test_different_top_k_different_key(self):
        key1 = self.get_key(make_sampling_params(top_k=-1))
        key2 = self.get_key(make_sampling_params(top_k=50))
        assert key1 != key2

    def test_different_prompt_logprobs_different_key(self):
        key1 = self.get_key(make_sampling_params(prompt_logprobs=0))
        key2 = self.get_key(make_sampling_params(prompt_logprobs=5))
        assert key1 != key2

    def test_different_texts_different_key(self):
        params = make_sampling_params()
        key1 = self.get_key(params)
        key2 = LossMethod._get_model_cache_key(["other text"], params, self.model)
        assert key1 != key2

    def test_lora_request_changes_key(self):
        params = make_sampling_params()
        lora = SimpleNamespace(lora_int_id=1, lora_name="adapter")
        key_without = LossMethod._get_model_cache_key(self.texts, params, self.model)
        key_with = LossMethod._get_model_cache_key(
            self.texts, params, self.model, lora
        )
        assert key_without != key_with


class TestModelCacheBehavior:
    def setup_method(self):
        BaseMethod.clear_cache()
        BaseMethod.set_max_cache_size(1000)

    def teardown_method(self):
        BaseMethod.clear_cache()
        BaseMethod.set_max_cache_size(1000)

    def make_method(self, generated):
        method = LossMethod({})
        model = mock.MagicMock()
        model.model_config.model = "test-model"
        model.generate.return_value = generated
        return method, model

    def test_second_call_hits_cache_and_skips_generate(self):
        method, model = self.make_method(generated=["out"])
        params = make_sampling_params()

        first = method.get_outputs(["hello"], model, params)
        second = method.get_outputs(["hello"], model, params)

        assert first == ["out"]
        assert second == ["out"]
        # Model inference only runs on the miss, not the hit.
        model.generate.assert_called_once()
        stats = BaseMethod.get_cache_stats()
        assert stats["model_hits"] == 1
        assert stats["model_misses"] == 1
        assert stats["model_hit_rate"] == "50.00%"

    def test_space_stripped_for_non_space_delimited_language(self):
        method, model = self.make_method(generated=["out"])
        params = make_sampling_params()
        method.get_outputs(
            ["a b c"], model, params, data_config={"space_delimited_language": False}
        )
        called_texts = model.generate.call_args[0][0]
        assert called_texts == ["abc"]

    def test_max_cache_size_evicts_oldest_entry(self):
        method, model = self.make_method(generated=["out"])
        params = make_sampling_params()

        BaseMethod.set_max_cache_size(1)
        method.get_outputs(["first"], model, params)
        method.get_outputs(["second"], model, params)

        # Only the most recent entry survives eviction.
        assert BaseMethod.get_cache_stats()["model_cache_size"] == 1

    def test_clear_cache_resets_stats(self):
        method, model = self.make_method(generated=["out"])
        method.get_outputs(["hello"], model, make_sampling_params())

        BaseMethod.clear_cache()

        stats = BaseMethod.get_cache_stats()
        assert stats["model_misses"] == 0
        assert stats["model_cache_size"] == 0
        assert stats["model_hit_rate"] == "0.00%"


class TestCleanupModel:
    def test_deletes_model_and_frees_memory(self):
        # torch and vllm are mocked in conftest; assert the teardown runs cleanly.
        with mock.patch("src.methods.base.destroy_model_parallel") as destroy:
            LossMethod.cleanup_model(mock.MagicMock())
        destroy.assert_called_once()
