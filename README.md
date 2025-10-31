# Fast-MIA

A framework for efficiently evaluating Membership Inference Attacks (MIA) against Large Language Models (LLMs). This tool enables fast execution of representative membership inference methods using vLLM.

## Features

- **Reduced Execution Time**: Efficiently runs multiple inference methods using vLLM and result caching while preserving evaluation accuracy.
- **Cross-Method Evaluation**: Compare and evaluate methods (LOSS, PPL/zlib, Min-K% Prob, etc.) under the same conditions.
- **Flexibility & Extensibility**: Easily change models, datasets, evaluation methods, and parameters using YAML configuration files.
- **Multiple Data Formats**: Supports CSV, JSON, JSONL, Parquet.
- **Hugging Face Datasets Support**: Directly load datasets from the Hugging Face Datasets library (WikiMIA, MIMIR).

### Supported Inference Methods

Currently, the following methods are supported.
The identifier is the name used in this framework's configuration.

| Type      | Method Name (identifier)     | Description                                                        |
|:----------|:-----------------|:-------------------------------------------------------------------|
| Baseline | **LOSS** (`loss`) | Uses the model's loss |
|| **PPL/zlib** (`zlib`) | Uses the ratio of information content calculated by Zlib compression |
| Token distribution | **Min-K% Prob** (`mink`) | https://github.com/swj0419/detect-pretrain-code |
| Text alternation | **Lowercase** (`lower`) | Uses the ratio of loss after lowercasing the text |
|| **PAC** (`pac`) | https://github.com/yyy01/PAC |
|| **ReCaLL** (`recall`) | https://github.com/ruoyuxie/recall |
|| **Con-ReCall** (`conrecall`) | https://github.com/WangCheng0116/CON-RECALL |
| Black-box | **SaMIA** (`samia`) | https://github.com/nlp-titech/samia |

## Quick Start

### Environment

Supported environments are Linux & NVIDIA GPUs.
It basically supports the same [GPU requirements as vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html).
For example, it takes a few minutes to run using NVIDIA A100 80GB.

### Installation

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# clone repository
git clone https://github.com/Nikkei/fast-mia.git
# install dependencies
cd fast-mia
uv sync
source .venv/bin/activate
```

### Execution

```bash
uv run --with 'vllm==0.10.2' python main.py --config config/sample.yaml
```

## How to Use

1. Create a configuration file (refer to `config/sample.yaml`)
2. Run the following command:

```bash
uv run --with 'vllm==0.10.2' python main.py --config config/your_own_configuration.yaml
```

### Parameters

- `--config`: Path to the configuration file (required)
- `--seed`: Random seed (default: 42)
- `--max-cache-size`: Maximum cache size (default: 1000)

### Configuration File

The configuration file is in YAML format and includes the following elements:

```yaml
# Model settings
model:
  model_id: "Model ID"
  # Other model parameters...
  # trust_remote_code=true is assumed

# LoRA settings (optional)
lora:
  name: "adapter_name"
  id: 1
  path: "adapter_path"

# Data settings
data:
  # Basic settings
  data_path: "./data/sample_data.csv"  # Path to data file (for Hugging Face datasets, use dataset name)
  format: "csv"                        # Data format (csv, jsonl, json, parquet, huggingface)
  
  # For using the WikiMIA dataset
  # data_path: "swj0419/WikiMIA"      # WikiMIA dataset name
  # format: "huggingface"             # Specify huggingface format

  # Column names
  text_column: "text"                # Name of the text column
  label_column: "label"              # Name of the label column

  # Token length setting
  token_length: 64                   # Number of tokens to use (for WikiMIA: 32, 64, 128, or 256)

  # Whether the language is space-delimited
  space_delimited_language: true

# Evaluation methods
methods:
  - type: "loss"
    params: {}
  # Other methods...

# Output settings
output_dir: "./results"
```

### Data Requirements

This supports various data formats.
For each format, the file of `data_path` is expected to be read as follows:

- "csv": `pd.read_csv(data_path)`
- "jsonl": `pandas.read_json(data_path, lines=True)`
- "json": `pd.read_json(data_path)`
- "parquet": `pandas.read_parquet(data_path)`
- "huggingface": `load_dataset.load_dataset(data_path)`

The data must contain at least the following columns (column names can be changed as long as they match those specified in the YAML):

- **text**: The text to be evaluated
- **label**: Membership label (1: member, 0: non-member)

For well-known benchmark datasets, dedicated specification options are provided:

- The [WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA) dataset is handled specially. If you set `data_path` to "swj0419/WikiMIA", it will be automatically recognized. For this dataset, the data corresponding to the specified token length (32, 64, 128, or 256) will be automatically loaded (e.g., "WikiMIA_length64").
- The [MIMIR](https://huggingface.co/datasets/iamgroot42/mimir) dataset is handled specially too. If you set `data_path` to "iamgroot42/mimir_{domain}_{ngram}", it will be automatically recognized. For this dataset, the data corresponding to the specified domain and ngram will be automatically loaded (e.g., "iamgroot42/mimir", "pile_cc", "ngram_7_0.2").

**Note**: To use the MIMIR dataset, you need to create a `.env` file in the project root directory with your Hugging Face token. Create a `.env` file with the following content:

```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

Replace `your_huggingface_token_here` with your actual Hugging Face token. You can obtain a token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

## Performance Comparison

Below is a performance comparison of Fast-MIA (left) and Transformers-based implementations (right) in terms of AUC, inference time, FPR@95, and TPR@5 for running membership inference evaluation. Performance metrics such as AUC remain almost the same, while inference time is approximately 5 times faster.

| Type             | Method                  | AUC        | Inference time (ratio)       | FPR@95       | TPR@5       |
|------------------|-------------------------|------------|------------------------------|--------------|-------------|
| baseline         | LOSS                    | 69.4 / 69.4 | 12s / 57s (×4.75)            | 84.3 / 84.3  | 18.3 / 18.3 |
|                  | PPL/zlib                | 69.8 / 69.8 | 12s / 57s (×4.75)            | 80.2 / 80.2  | 14.5 / 14.5 |
| token distribution | Min-K% Prob (K=0.1)   | 67.2 / 67.2 | 12s / 57s (×4.75)            | 83.5 / 83.3  | 17.3 / 17.3 |
|                  | Min-K% Prob (K=0.2)     | 69.3 / 69.3 | 12s / 57s (×4.75)            | 82.3 / 82.3  | 22.0 / 22.0 |
|                  | Min-K% Prob (K=0.3)     | 70.1 / 70.1 | 12s / 57s (×4.75)            | 82.3 / 82.3  | 19.6 / 19.6 |
|                  | Min-K% Prob (K=0.5)     | 69.7 / 69.7 | 12s / 57s (×4.75)            | 82.5 / 82.5  | 18.1 / 18.1 |
|                  | Min-K% Prob (K=0.8)     | 69.5 / 69.5 | 12s / 57s (×4.75)            | 84.3 / 84.3  | 18.1 / 18.3 |
|                  | Min-K% Prob (K=1.0)     | 69.4 / 69.4 | 12s / 57s (×4.75)            | 84.3 / 84.3  | 18.3 / 18.3 |
| text alternation | Lowercase               | 64.1 / 64.1 | 25s / 1m59s (×4.76)          | 83.5 / 83.8  | 11.6 / 11.6 |
|                  | PAC                     | 73.3 / 73.4 | 1m17s / 6m24s (×4.99)        | 82.3 / 77.9  | 27.6 / 24.3 |
|                  | ReCaLL                  | 90.7 / 90.3 | 55s / 2m10s (×2.36)          | 28.5 / 34.7  | 50.4 / 48.8 |
|                  | Con-ReCall          | 96.8 / 96.1 | 1m53s / 3m30s (×1.86)        | 10.8 / 12.9  | 78.0 / 73.6 |
| black-box        | SaMIA               | 65.5 / 64.5 | 2h3m24s / 40h9m53s (×19.5)   | 90.5 / 90.7  | 22.7 / 15.5 |

### Experimental Setup

- Hardware: NVIDIA A100 80GB
- Model: LLaMA 30B (16-bit)
- Dataset: WikiMIA (length: 32)
- Configuration: The Fast-MIA experimental results were obtained using `config/llama30b-exp.yaml`.

## How to contribute

Feel free to contact the [maintainers](https://github.com/Nikkei/fast-mia/blob/main/pyproject.toml#L8).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

See [NOTICE](NOTICE) for information about third-party code and modifications.

## Reference

```
@misc{XXXX.XXXX,
  Author = {Hiromu Takahashi and Shotaro Ishihara},
  Title = {{Fast-MIA}: Efficient and Scalable Membership Inference for LLMs},
  Year = {2025},
  Eprint = {arXiv:2510.23074},
  URL = {https://arxiv.org/abs/2510.23074}
}
```
