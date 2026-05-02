# Adding MIA Methods

This guide explains how to add a new membership inference method to Fast-MIA.
For examples, see the recently added
[`DCPDDMethod`](https://github.com/Nikkei/fast-mia/pull/19) and
[`RefMethod`](https://github.com/Nikkei/fast-mia/pull/43), which cover
method-specific parameters, custom runtime inputs, and additional model logic.

## Overview

A method is a subclass of `BaseMethod` registered in `MethodFactory`. In most
cases, adding a method involves these changes:

1. Add a new implementation file under `src/methods/`.
2. Register the method in `src/methods/factory.py`.
3. Export the class from `src/methods/__init__.py`.
4. Add or update tests.
5. If you plan to contribute the method back to Fast-MIA, document it in the
   supported method tables and configuration guide.

If the method needs extra inputs, such as labels or the tokenizer, declare those
requirements on the method class so the evaluator can pass the correct
arguments.

## Implement the Method

Create a file such as `src/methods/my_method.py` and subclass `BaseMethod`.
Implement the method-specific flow in `run()`, and call `self.get_outputs(...)`
inside `run()` whenever the method needs target model outputs. `BaseMethod` is
abstract, so each subclass must also implement `process_output()`; use it for the
per-output scoring logic that `run()` applies to model outputs.

```python
from typing import Any

import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .base import BaseMethod


class MyMethod(BaseMethod):
    """My membership inference method."""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        super().__init__("my_method", method_config)
        self.scale = self.method_config.get("scale", 1.0)

    def process_output(self, output: RequestOutput) -> float:
        token_log_probs = self._extract_token_log_probs(output)
        return float(np.mean(token_log_probs) * self.scale)

    def run(
        self,
        texts: list[str],
        model: LLM,
        sampling_params: SamplingParams,
        lora_request: LoRARequest = None,
        data_config: dict[str, Any] = None,
    ) -> list[float]:
        outputs = self.get_outputs(
            texts, model, sampling_params, lora_request, data_config
        )
        return [self.process_output(output) for output in outputs]
```

`get_outputs()` handles shared model-output caching, LoRA requests, and
language-specific text normalization. Use it instead of calling
`model.generate(...)` directly unless the method has a specific reason to bypass
the shared behavior.

## Method Requirements

The evaluator builds the `run()` arguments from class-level requirement flags:

| Flag | Default | Effect |
|------|---------|--------|
| `requires_labels` | `False` | Passes `labels` immediately after `texts`. |
| `requires_tokenizer` | `False` | Passes the tokenizer after `model`. |
| `requires_sampling_params` | `True` | Passes `sampling_params` after `model` or `tokenizer`. |

For example, `DCPDDMethod` sets `requires_tokenizer = True`, so its `run()`
signature receives `tokenizer` between `model` and `sampling_params`:

```python
def run(
    self,
    texts,
    model,
    tokenizer,
    sampling_params,
    lora_request=None,
    data_config=None,
):
    ...
```

Set these flags only when the method genuinely needs the additional input.

## Customize `run()` When Needed

Add the method-specific logic to `run()` when scoring needs more than a direct
one-score-per-output transformation. Common cases include:

- The method needs additional preprocessing or cached statistics before scoring,
  as in `DCPDDMethod`.
- The method needs another model, as in `RefMethod`.
- The method generates multiple augmented samples or prompts per input.
- The method needs labels or tokenizer-specific processing.

When implementing `run()`, keep these conventions:

- Return one numeric score for each input text, in the same order.
- Use `self.get_outputs(...)` instead of calling `model.generate(...)` directly
  unless there is a specific reason to bypass the shared cache.
- Accept `lora_request=None` and `data_config=None` so the evaluator can pass
  runtime context consistently.
- Validate required method-specific config in `__init__()` and raise `ValueError`
  with a clear message when required keys are missing.
- If a temporary vLLM model is loaded, release it with `self.cleanup_model(...)`
  when the method finishes.

## Register the Method

Import the class and add it to `METHOD_BUILDERS` in `src/methods/factory.py`.
This key is the YAML `type` value users will configure.

```python
from .my_method import MyMethod

METHOD_BUILDERS = {
    ...
    "my_method": MyMethod,
}
```

Also export the class from `src/methods/__init__.py`:

```python
from .my_method import MyMethod

__all__ = [
    ...
    "MyMethod",
]
```

After registration, add the method to the YAML config before running
Fast-MIA. The `type` value must match the key added to `METHOD_BUILDERS`.

```yaml
methods:
  - type: "my_method"
    params:
      scale: 1.0
```

## Update Tests

At minimum, add the method to `tests/unit/test_factory.py` so factory
registration is covered:

```python
from src.methods.my_method import MyMethod

@pytest.mark.parametrize("type_,cls,params", [
    ...
    ("my_method", MyMethod, {"scale": 1.0}),
])
def test_create_method(self, type_, cls, params):
    method = MethodFactory.create_method({"type": type_, "params": params})
    assert isinstance(method, cls)
```

Add method-specific unit tests when the scoring logic has parameters,
preprocessing, custom cache behavior, required configuration, or custom `run()`
logic. For methods that load external resources or additional models, mock those
dependencies in unit tests.

## Update Documentation for Contributions

When contributing a method to Fast-MIA, document it wherever supported methods
are listed:

- `README.md`
- `docs/docs/index.md`
- `docs/docs/how-to-use.md`

The `how-to-use.md` entry should include the YAML `type`, a short description,
and all method-specific `params` with defaults and required keys. If the method
needs a sample configuration file, add or update a file under `config/`.

## Checklist

- [ ] Add `src/methods/<method>.py`.
- [ ] Subclass `BaseMethod`, implement `process_output()`, and put the
      method-specific flow in `run()`.
- [ ] Set `requires_labels`, `requires_tokenizer`, or
      `requires_sampling_params` if needed.
- [ ] Register the method in `METHOD_BUILDERS`.
- [ ] Export the class from `src/methods/__init__.py`.
- [ ] Add tests for factory registration and method-specific behavior.
- [ ] For contributions, update the supported method tables in README and docs.
- [ ] For contributions, add a sample configuration file when the method has
      non-trivial parameters.
