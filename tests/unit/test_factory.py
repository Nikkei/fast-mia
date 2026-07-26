import pytest
from src.methods.factory import MethodFactory
from src.methods.aeca import AECAMethod
from src.methods.conrecall import CONReCaLLMethod
from src.methods.dcpdd import DCPDDMethod
from src.methods.loss import LossMethod
from src.methods.lower import LowerMethod
from src.methods.mink import MinKMethod
from src.methods.neighbour import NeighbourMethod
from src.methods.pac import PACMethod
from src.methods.recall import ReCaLLMethod
from src.methods.ref import RefMethod
from src.methods.samia import SaMIAMethod
from src.methods.zlib import ZlibMethod

class TestMethodFactory:
    @pytest.mark.parametrize("type_,cls,params", [
        ("loss", LossMethod, {}),
        ("lower", LowerMethod, {}),
        ("zlib", ZlibMethod, {}),
        ("mink", MinKMethod, {"ratio": 0.1}),
        ("pac", PACMethod, {"alpha": 0.2, "N": 3}),
        ("recall", ReCaLLMethod, {"num_shots": 5}),
        ("samia", SaMIAMethod, {"num_samples": 3}),
        ("conrecall", CONReCaLLMethod, {"num_shots": 5}),
        ("dcpdd", DCPDDMethod, {"file_num": 10}),
        ("ref", RefMethod, {"reference_model": {"model_id": "dummy"}}),
        ("neighbour", NeighbourMethod, {}),
        ("aeca", AECAMethod, {}),
    ])
    def test_create_method(self, type_, cls, params):
        method = MethodFactory.create_method({"type": type_, "params": params})
        assert isinstance(method, cls)
        # method_name should contain the type name
        assert type_ in method.method_name

    def test_unknown_method_type(self):
        with pytest.raises(ValueError, match="Unknown method type 'unknown'"): 
            MethodFactory.create_method({"type": "unknown", "params": {}})

    def test_missing_method_type(self):
        with pytest.raises(ValueError, match="must include a 'type'"):
            MethodFactory.create_method({"params": {}})

    def test_raises_when_ref_is_not_last(self):
        configs = [
            {"type": "ref", "params": {"reference_model": {"model_id": "dummy"}}},
            {"type": "loss", "params": {}},
        ]
        with pytest.raises(ValueError, match="Move 'ref' to the end"):
            MethodFactory.create_methods(configs)

    def test_no_error_when_ref_is_last(self):
        configs = [
            {"type": "loss", "params": {}},
            {"type": "ref", "params": {"reference_model": {"model_id": "dummy"}}},
        ]
        methods = MethodFactory.create_methods(configs)
        assert [type(m) for m in methods] == [LossMethod, RefMethod]
