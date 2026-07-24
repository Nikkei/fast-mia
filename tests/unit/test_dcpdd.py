from src.methods.dcpdd import DCPDDMethod


class TestFreqDistCachePath:
    def test_includes_model_id_file_num_and_max_token_length(self):
        method = DCPDDMethod({"file_num": 10, "max_token_length": 128})
        path = method._freq_dist_cache_path("facebook/opt-125m")
        assert path.name == "freq_dist_facebook--opt-125m_10_128.json"

    def test_different_models_use_different_caches(self):
        method = DCPDDMethod({"file_num": 15})
        path_a = method._freq_dist_cache_path("org/model-a")
        path_b = method._freq_dist_cache_path("org/model-b")
        assert path_a != path_b

    def test_different_max_token_length_use_different_caches(self):
        method_a = DCPDDMethod({"file_num": 15, "max_token_length": 128})
        method_b = DCPDDMethod({"file_num": 15, "max_token_length": 1024})
        path_a = method_a._freq_dist_cache_path("org/model")
        path_b = method_b._freq_dist_cache_path("org/model")
        assert path_a != path_b

    def test_unknown_model_id(self):
        method = DCPDDMethod({"file_num": 15, "max_token_length": 1024})
        path = method._freq_dist_cache_path("")
        assert path.name == "freq_dist_unknown-model_15_1024.json"
