# How to Use

This guide explains how to run the CLI and how to describe each section of the YAML configuration file.

## Run Fast-MIA

1. Create a configuration file (refer to `config/sample.yaml`)
2. Run the following command:

```bash
uv run --with 'vllm==0.15.1' python main.py --config config/your_own_configuration.yaml
```

**Note**: When using T4 GPUs (e.g., Google Colab), set the environment variable to avoid attention backend issues:
> ```bash
> VLLM_ATTENTION_BACKEND=XFORMERS uv run --with 'vllm==0.15.1' python main.py --config config/sample.yaml
> ```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eOf6JSz6vPc7Im0tMw1Us04JxbAnXXx?usp=sharing)

### CLI Arguments

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `--config` | ✅ | – | Path to the YAML configuration file. |
| `--seed` | ❌ | `42` | Global seed passed to `random`, `torch`, `numpy`, and Python to make evaluations reproducible. |
| `--max-cache-size` | ❌ | `1000` | Maximum number of vLLM generations cached across methods; the default is sufficient for most runs. |
| `--detailed-report` | ❌ | Off | Generate detailed report with metadata, per-sample scores, and visualizations. |

## Configuration Files

All behavior is driven by a YAML file. Each top-level key toggles a different subsystem:

| Key | Purpose |
|-----|---------|
| `model` | Required. Parameters are directly forwarded to `vllm.LLM` (model, dtype, etc.). |
| `sampling_parameters` | Optional. Parameters are directly forwarded to `vllm.SamplingParams`. `prompt_logprobs` defaults to `0`, `max_tokens` defaults to `1`, `temperature` defaults to `0.0`, and `top_p` defaults to `1.0` when omitted. |
| `lora` | Optional. Parameters are directly forwarded to `vllm.lora.request.LoRARequest`. Omit when no adapter is needed. |
| `data` | Required. Specifies the dataset source, file format, column names, etc. |
| `methods` | Required. Ordered list of evaluation methods. Each entry declares a `type` and method-specific `params`. |
| `output_dir` | Optional. Directory where evaluation CSVs are stored (`./results` by default). |

### Example Configuration

Please refer to `config/sample.yaml` for a complete example configuration file.

### `model` Block

| Field | Required | Notes |
|-------|----------|-------|
| `model_id` | ✅ | The name or path of a HuggingFace Transformers model that `vllm.LLM` can load. (= model) |
| Other keys | ❌ | Forwarded directly to `vllm.LLM`. Select params to fit the model onto your hardware, following the [vLLM.LLM API Reference](https://docs.vllm.ai/en/stable/api/vllm/index.html#vllm.LLM). |

### `sampling_parameters` Block

Values are passed to `vllm.SamplingParams`. Select params following the [vLLM.SamplingParams API Reference](https://docs.vllm.ai/en/stable/api/vllm/index.html#vllm.SamplingParams). Fast-MIA automatically enforces `prompt_logprobs: 0`, `max_tokens: 1`, `temperature: 0.0`, and `top_p: 1.0` whenever the config omits those keys to ensure deterministic, efficient scoring, but you can override any of these defaults by explicitly setting them inside this block.

**Recommended defaults for deterministic scoring**

| Field | Purpose |
|-------|---------|
| `max_tokens` | Use `1` to request only the immediate next token; Fast-MIA defaults to this. |
| `prompt_logprobs` | Use `0` to get prompt text log-probabilities; Fast-MIA defaults to this. |
| `temperature` | Set to `0.0` for deterministic runs; Fast-MIA defaults to this. |
| `top_p` | Leave at `1.0` so determinism is governed solely by `temperature`; Fast-MIA defaults to this. |

### `lora` Block (Optional)

Values are passed to `vllm.lora.request.LoRARequest`. Select params following the [vLLM.lora.request.LoRARequest API Reference](https://docs.vllm.ai/en/stable/api/vllm/lora/request.html#vllm.lora.request.LoRARequest).

| Field | Purpose |
|-------|---------|
| `lora_name` | Human-readable adapter name (used in logs). |
| `lora_int_id` | Integer identifier that vLLM uses to cache the adapter. Use different IDs for simultaneous adapters. |
| `lora_path` | Filesystem path to the adapter weights. |

Omit the entire block if you evaluate the base model. When enabled, the adapter is transparently applied to all methods.

**Note**: If you enable LoRA, there is a known prompt-logprob bug (https://discuss.vllm.ai/t/bug-wrong-lora-mapping-during-prompt-logprobs-computing/500/2). Setting `sampling_parameters.prompt_logprobs` currently raises an error, so methods like `loss` or `mink` cannot run.

### `data` Block

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `data_path` | ✅ | – | Path to a local file or one of the dedicated Hugging Face datasets (`swj0419/WikiMIA` or `iamgroot42/mimir_{domain}_{ngram}`). |
| `format` | ❌ | `csv` | One of `csv`, `jsonl`, `json`, `parquet`. Use `huggingface` **only** in combination with `swj0419/WikiMIA` or `iamgroot42/mimir_{domain}_{ngram}`. |
| `text_column` | ❌ | `text` | Column containing the raw text to probe. |
| `label_column` | ❌ | `label` | Column containing membership labels (`1` = member, `0` = non-member). |
| `text_length` | ❌ | `32` | Number of words kept from each sample. WikiMIA requires one of `32`, `64`, `128`, `256`; MIMIR requires `200`. |
| `space_delimited_language` | ❌ | `true` | Set to `false` for languages without whitespace like Japanese. |

**Note**: If `space_delimited_language` is `false`, you must pre-process your text and insert spaces between words beforehand; Fast-MIA assumes the input is already space-separated and will simply strip the spaces back out during scoring.

#### Supported File Formats

| Format | Reader |
|--------|--------|
| `csv` | `pandas.read_csv(data_path)` |
| `jsonl` | `pandas.read_json(data_path, lines=True)` |
| `json` | `pandas.read_json(data_path)` |
| `parquet` | `pandas.read_parquet(data_path)` |
| `huggingface` | Not available for arbitrary datasets. Use only with the dedicated WikiMIA/MIMIR loaders described below. |

#### Supported Hugging Face Datasets

Currently, only the following datasets are supported via the `huggingface` format.

- The [WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA) dataset is handled specially. If you set `data_path` to "swj0419/WikiMIA", it will be automatically recognized. For this dataset, the data corresponding to the specified text length (32, 64, 128, or 256) will be automatically loaded (e.g., "WikiMIA_length64").
- The [MIMIR](https://huggingface.co/datasets/iamgroot42/mimir) dataset is handled specially too. If you set `data_path` to "iamgroot42/mimir_{domain}_{ngram}", it will be automatically recognized. For this dataset, the data corresponding to the specified domain and ngram will be automatically loaded (e.g., "iamgroot42/mimir", "pile_cc", "ngram_7_0.2").

**Note**: To use the MIMIR dataset, you need to create a `.env` file in the project root directory with your Hugging Face token. Create a `.env` file with the following content:

```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

Replace `your_huggingface_token_here` with your actual Hugging Face token. You can obtain a token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### `methods` Block

Declare any number of methods. Each entry looks like:

```yaml
- type: "method_name"
  params:
    # method-specific keys
```

Available method types and their parameters:

| Method Name | Description | Parameters |
|------|-------------|----------------|
| `loss` | Uses the model's loss | – |
| `zlib` | Uses the ratio of information content calculated by Zlib compression | – |
| `lower` | Uses the ratio of loss after lowercasing the text | – |
| `mink` | https://github.com/swj0419/detect-pretrain-code | `ratio` (`0.0–1.0`, default `0.5`). |
| `pac` | https://github.com/yyy01/PAC | `alpha` (augmentation strength, default `0.3`), `N` (augmentations per sample, default `5`). |
| `recall` | https://github.com/ruoyuxie/recall | `num_shots` (number of prefix texts, default `10`), `pass_window` (skip max-length trimming, default `False`). |
| `conrecall` | https://github.com/WangCheng0116/CON-RECALL | Same as `recall` plus `gamma` (ratio of member prefixes loss, default `0.5`). |
| `samia` | https://github.com/nlp-titech/samia | `num_samples` (number of samples, default `5`), `prefix_ratio` (ratio of prefix, default `0.5`), `zlib` (Use Zlib, default `True`). |
| `dcpdd` | https://github.com/zhang-wei-chao/DC-PDD | `file_num` (number of C4 text files, default `15`), `max_token_length` (max token length, default `1024`), `alpha` (upper bound of score, default `0.01`). |

### `output_dir`

Directory where CSV results are written. The folder is created if missing.

## Output Files

### Default Output

By default, Fast-MIA saves the following files in a timestamped folder:

```
results/
└── YYYYMMDD-HHMMSS/
    ├── config.yaml    # Copy of the configuration used
    ├── results.csv    # Summary metrics (AUROC, FPR@95, TPR@5)
    └── report.txt     # Human-readable summary report
```

## Detailed Report Mode

When you run with `--detailed-report`, Fast-MIA generates additional outputs for benchmarking and analysis:

```bash
uv run --with 'vllm==0.15.1' python main.py --config config/sample.yaml --detailed-report
```

### Output Structure

```
results/
└── YYYYMMDD-HHMMSS/           # Timestamped folder for each run
    ├── config.yaml            # Copy of the configuration used
    ├── results.csv            # Summary metrics (AUROC, FPR@95, TPR@5)
    ├── detailed_scores.csv    # Per-sample scores for each method
    ├── metadata.json          # Execution metadata (JSON format)
    ├── metadata.yaml          # Execution metadata (YAML format)
    ├── report.txt             # Human-readable summary report
    └── figures/
        ├── roc_curves.png         # ROC curves for all methods
        ├── score_distributions.png # Score histograms (member vs non-member)
        └── metrics_comparison.png  # Bar chart comparing metrics
```

### Metadata Contents

The metadata files (`metadata.json` / `metadata.yaml`) include:

| Section | Contents |
|---------|----------|
| `experiment` | Start/end time, duration |
| `environment` | Python version, platform, hostname |
| `git` | Commit hash, branch, dirty status |
| `model` | Model ID, parameters |
| `data` | Dataset path, format, sample counts |
| `sampling_parameters` | vLLM sampling configuration |
| `methods` | List of evaluated methods with parameters |
| `cache` | Cache hit/miss statistics |

### Detailed Scores

The `detailed_scores.csv` file contains per-sample scores:

```csv
label,loss,zlib,mink_0.2,recall
1,0.832,0.654,0.721,0.891
0,0.234,0.312,0.287,0.156
...
```

This allows for post-hoc analysis, custom visualizations, or statistical testing.
