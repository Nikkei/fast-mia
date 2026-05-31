# Repository Guidelines

This file provides guidance to LLM agents when working with code in this repository.

## Overview

Fast-MIA is a framework for evaluating Membership Inference Attacks (MIA) against LLMs. It runs multiple MIA methods efficiently using vLLM and a shared model output cache.

**Note**: vLLM is a runtime dependency, not listed in `pyproject.toml`. It is injected at execution time via `--with 'vllm==0.15.1'`.

## Commands

```bash
# Initial setup (install dependencies + pre-commit hooks)
make setup
# or manually:
uv sync --group dev
uv run pre-commit install

# Run evaluation
uv run --with 'vllm==0.15.1' python main.py --config config/sample.yaml

# Run with detailed report (metadata, per-sample scores, visualizations)
uv run --with 'vllm==0.15.1' python main.py --config config/sample.yaml --detailed-report

# Lint and format (ruff)
make lint
# or manually:
uv run ruff check --fix --config pyproject.toml
uv run ruff format --config pyproject.toml

# Run all tests
make test
# or manually:
uv run pytest --cov tests -v

# Run a single test file
uv run pytest tests/unit/test_factory.py

# Run a specific test
uv run pytest tests/unit/test_factory.py::TestMethodFactory::test_create_method
```

Python 3.12+ is required. Tests use `src` as the pythonpath root (set in `pyproject.toml`).

## Architecture

### Execution Flow

`main.py` (top-level entry point) → `src/main.py::main()` orchestrates:

1. **Config** (`src/config.py`): Loads YAML config, exposes `.model`, `.data`, `.methods`, `.sampling_parameters`, `.lora`
2. **DataLoader** (`src/data_loader.py`): Loads member/non-member texts from CSV, JSON, JSONL, Parquet, or HuggingFace datasets (WikiMIA, Mimir). Mimir requires `HUGGINGFACE_TOKEN` in `.env`.
3. **ModelLoader** (`src/model_loader.py`): Wraps vLLM's `LLM` class for loading target (and reference) models
4. **MethodFactory** (`src/methods/factory.py`): Instantiates `BaseMethod` subclasses from config using the `METHOD_BUILDERS` dict
5. **Evaluator** (`src/evaluator.py`): Runs each method against the data, collects scores, computes AUROC/FPR@95/TPR@5 metrics. Returns `EvaluationResult`.
6. **ResultWriter** (`src/result_writer.py`): Saves `config.yaml`, `results.csv`, `report.txt` to `results/YYYYMMDD-HHMMSS/`
7. **Visualizer** (`src/visualizer.py`): Generates ROC curves and score distribution plots (only in `--detailed-report` mode)

### Method System

All MIA methods live in `src/methods/` and subclass `BaseMethod` (`src/methods/base.py`):

- **`process_output(output: RequestOutput) -> float`** (abstract): Per-output scoring logic
- **`run(...) -> list[float]`**: Default implementation calls `get_outputs()` then `process_output()`. Override when a method needs custom flow (e.g., extra models, augmented prompts).
- **`get_outputs(...)`**: Handles shared LRU model output cache (`BaseMethod._model_cache`, `OrderedDict`, class-level, shared across all methods) and language normalization. Always prefer this over `model.generate()` directly.

#### Method Requirement Flags (class-level)

| Flag | Default | Effect |
|------|---------|--------|
| `requires_labels` | `False` | Evaluator passes `labels` after `texts` |
| `requires_tokenizer` | `False` | Evaluator passes tokenizer after `model` |
| `requires_sampling_params` | `True` | Evaluator passes `sampling_params` |

#### Registering a New Method

Use the `/add-mia-method` skill (supported in Claude Code, Codex, and other
skill-compatible assistants) to automate the steps below from a paper URL:

```
/add-mia-method <paper-url> [github-url]
```

Manual steps:

1. Create `src/methods/<method>.py`, subclass `BaseMethod`, implement `process_output()`
2. Add to `METHOD_BUILDERS` in `src/methods/factory.py`
3. Export from `src/methods/__init__.py`
4. Add factory registration test in `tests/unit/test_factory.py`

See `docs/docs/adding-methods.md` for the full checklist including documentation updates.

### Configuration

YAML config keys: `model`, `lora` (optional), `data`, `sampling_parameters`, `methods`, `output_dir`. See `config/sample.yaml` for a full example.

`data.space_delimited_language: false` causes `get_outputs()` to strip spaces before inference (used for non-space-delimited languages like Japanese/Chinese).

## Coding Style

- Python 3.12+ compatibility required
- Line length: 88 characters (enforced by ruff)
- Prefer `pathlib.Path` over string paths
- Use `snake_case` for modules, functions, variables, and method identifiers
- Method module names should match their config identifier (e.g., `src/methods/loss.py` for `loss`)
- Ruff enforces: pycodestyle, Pyflakes, import sorting, pyupgrade, annotations, bugbear, pathlib, pep8-naming

## Testing Guidelines

- Test files must be named `test_*.py`
- Fast isolated unit tests go in `tests/unit/`
- Config and input fixtures go in `tests/fixtures/`
- Broader data/config flow tests are integration tests at `tests/` top level
- Always add or update factory tests in `tests/unit/test_factory.py` when registering a new method

## Commit & PR Guidelines

- Short subject lines with lightweight prefixes: `docs:`, `build(deps):`, `feat:`, `fix:`, etc.
- Keep commits focused (e.g., `docs: clarify adding methods guide`)
- PRs should describe the change, include test results, and link related issues
- Mention config, data, or documentation updates in the PR description
- Include screenshots or rendered examples for user-visible output or docs changes

## Security & Configuration

- Never commit secrets, model tokens, generated caches, or large result artifacts
- Use `.env` for sensitive values such as `HUGGINGFACE_TOKEN`
- `.env.example` documents expected keys
- vLLM is injected at runtime via `--with`; do not add it to core dependencies
