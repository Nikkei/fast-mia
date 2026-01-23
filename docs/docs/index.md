# Fast-MIA

**Fast-MIA** is a framework for efficiently evaluating Membership Inference Attacks (MIA) against Large Language Models (LLMs). This tool enables fast execution of representative membership inference methods using vLLM.

## ✨ Features

- 🚀 **Reduced Execution Time**: Efficiently runs multiple inference methods using vLLM and result caching while preserving evaluation accuracy.
- 📊 **Cross-Method Evaluation**: Compare and evaluate methods (LOSS, PPL/zlib, Min-K% Prob, etc.) under the same conditions.
- 🔧 **Flexibility & Extensibility**: Easily change models, datasets, evaluation methods, and parameters using YAML configuration files.
- 🎯 **Multiple Data Formats**: Supports CSV, JSON, JSONL, Parquet, and Hugging Face Datasets.

## 🚀 Quick Start

### Environment

Supported environments are Linux & NVIDIA GPUs. It basically supports the same [GPU requirements as vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html). For example, it takes a few minutes to run using NVIDIA A100 80GB.

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
**Note**: When using T4 GPUs (e.g., Google Colab, Kaggle), set the environment variable to avoid attention backend issues:
> ```bash
> VLLM_ATTENTION_BACKEND=XFORMERS uv run --with 'vllm==0.10.2' python main.py --config config/sample.yaml
> ```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eOf6JSz6vPc7Im0tMw1Us04JxbAnXXx?usp=sharing)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sishihara/fast-mia-config-sample-yaml)

### Detailed Report Mode

For benchmarking with detailed outputs (metadata, per-sample scores, visualizations):

```bash
uv run --with 'vllm==0.10.2' python main.py --config config/sample.yaml --detailed-report
```

## 📚 Supported MIA Methods

Fast-MIA supports the following MIA methods:

| Type | Method Name (identifier) | Description |
|:-----|:------------------------|:------------|
| Baseline | **LOSS** (`loss`) | Uses the model's loss |
| | **PPL/zlib** (`zlib`) | Uses the ratio of information content calculated by Zlib compression |
| Token distribution | **Min-K% Prob** (`mink`) | https://github.com/swj0419/detect-pretrain-code |
| Text alternation | **Lowercase** (`lower`) | Uses the ratio of loss after lowercasing the text |
| | **PAC** (`pac`) | https://github.com/yyy01/PAC |
| | **ReCaLL** (`recall`) | https://github.com/ruoyuxie/recall |
| | **Con-ReCall** (`conrecall`) | https://github.com/WangCheng0116/CON-RECALL |
| Black-box | **SaMIA** (`samia`) | https://github.com/nlp-titech/samia |

## 📖 Documentation

- [API Reference](api-reference.md) - Detailed API documentation

## 🔗 Links

- [GitHub Repository](https://github.com/Nikkei/fast-mia)
- [Issue Tracker](https://github.com/Nikkei/fast-mia/issues)

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/Nikkei/fast-mia/blob/main/LICENSE) file for details.

## 📑 Reference

```
@misc{takahashi_ishihara_fastmia,
  Author = {Hiromu Takahashi and Shotaro Ishihara},
  Title = {{Fast-MIA}: Efficient and Scalable Membership Inference for LLMs},
  Year = {2025},
  Eprint = {arXiv:2510.23074},
  URL = {https://arxiv.org/abs/2510.23074}
}
```
