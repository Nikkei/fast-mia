from src.methods.dcpdd import DCPDDMethod


class TestFreqDistCachePath:
    def test_includes_model_id_and_file_num(self):
        method = DCPDDMethod({"file_num": 10})
        path = method._freq_dist_cache_path("facebook/opt-125m")
        assert path.name == "freq_dist_facebook--opt-125m_10.json"

    def test_different_models_use_different_caches(self):
        method = DCPDDMethod({"file_num": 15})
        path_a = method._freq_dist_cache_path("org/model-a")
        path_b = method._freq_dist_cache_path("org/model-b")
        assert path_a != path_b

    def test_unknown_model_id(self):
        method = DCPDDMethod({"file_num": 15})
        path = method._freq_dist_cache_path("")
        assert path.name == "freq_dist_unknown-model_15.json"
