import pytest

from src.methods.pac import PACMethod


class TestPACValidation:
    def test_default_params_valid(self):
        method = PACMethod({})
        assert method.alpha == 0.3
        assert method.N == 5

    def test_alpha_zero_rejected(self):
        with pytest.raises(ValueError, match="'alpha' > 0"):
            PACMethod({"alpha": 0})

    def test_alpha_negative_rejected(self):
        with pytest.raises(ValueError, match="'alpha' > 0"):
            PACMethod({"alpha": -0.1})

    def test_n_zero_rejected(self):
        with pytest.raises(ValueError, match="'N' >= 1"):
            PACMethod({"N": 0})
