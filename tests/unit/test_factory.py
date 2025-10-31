import pytest
from src.methods.factory import MethodFactory
from src.methods.loss import LossMethod
from src.methods.lower import LowerMethod
from src.methods.zlib import ZlibMethod
from src.methods.mink import MinKMethod
from src.methods.pac import PACMethod
from src.methods.recall import ReCaLLMethod

class TestMethodFactory:
    @pytest.mark.parametrize("type_,cls,params", [
        ("loss", LossMethod, {}),
        ("lower", LowerMethod, {}),
        ("zlib", ZlibMethod, {}),
        ("mink", MinKMethod, {"ratio": 0.1}),
        ("pac", PACMethod, {"alpha": 0.2, "N": 3}),
        ("recall", ReCaLLMethod, {"num_shots": 5}),
    ])
    def test_create_method(self, type_, cls, params):
        method = MethodFactory.create_method({"type": type_, "params": params})
        assert isinstance(method, cls)
        # method_name should contain the type name
        assert type_ in method.method_name

    def test_unknown_method_type(self):
        with pytest.raises(ValueError, match="Unknown method type"): 
            MethodFactory.create_method({"type": "unknown", "params": {}})
