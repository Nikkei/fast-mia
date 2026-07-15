from types import SimpleNamespace

from src.methods.loss import LossMethod


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
