"""
Pytest configuration file to mock vllm imports when vllm is not installed.
This allows tests to run without installing vllm, which significantly reduces CI time.
"""
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock


class MockTensor:
    """Mock class for torch.Tensor that can be used with isinstance()"""
    pass


class MockNNModule:
    """Mock class for torch.nn.Module that can be used with issubclass()"""
    pass


class MockNN(ModuleType):
    """Mock torch.nn module"""
    def __init__(self):
        super().__init__('torch.nn')
        object.__setattr__(self, '__spec__', ModuleSpec('torch.nn', None))
        object.__setattr__(self, 'Module', MockNNModule)
    
    def __getattr__(self, name):
        if name in object.__getattribute__(self, '__dict__'):
            return object.__getattribute__(self, name)
        return MagicMock()


class MockModule(ModuleType):
    """Mock module that returns MagicMock for any attribute access"""
    def __init__(self, name):
        super().__init__(name)
        # Set __spec__ to avoid errors when packages check for it
        object.__setattr__(self, '__spec__', ModuleSpec(name, None))
        object.__setattr__(self, '__name__', name)
        # For torch module, set Tensor to a real class and nn to a submodule
        if name == 'torch':
            object.__setattr__(self, 'Tensor', MockTensor)
            object.__setattr__(self, 'nn', MockNN())
    
    def __getattr__(self, name):
        # Check in __dict__ to avoid infinite recursion
        if name in object.__getattribute__(self, '__dict__'):
            return object.__getattribute__(self, name)
        # Return MagicMock for any other attribute
        return MagicMock()


# List of packages to mock
PACKAGES_TO_MOCK = ['vllm', 'torch', 'transformers']

for package_name in PACKAGES_TO_MOCK:
    try:
        __import__(package_name)
    except ImportError:
        # Create a mock module that automatically mocks all submodules and attributes
        mock_module = MockModule(package_name)
        
        # Add to sys.modules so imports work
        sys.modules[package_name] = mock_module
        
        # Create a custom importer that automatically mocks all submodules
        class MockFinder:
            def __init__(self, package_name):
                self.package_name = package_name
            
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(self.package_name):
                    return ModuleSpec(fullname, MockLoader())
                return None
        
        class MockLoader:
            @staticmethod
            def create_module(spec):
                # Special handling for torch.nn
                if spec.name == 'torch.nn':
                    return MockNN()
                # Create a new mock module for any other submodule
                return MockModule(spec.name)
            
            @staticmethod
            def exec_module(module):
                # No need to execute anything, it's just a mock
                pass
        
        # Register the custom importer
        sys.meta_path.insert(0, MockFinder(package_name))

