# Running on Google Cloud

Fast-MIA provides scripts to submit GPU evaluation jobs to Google Compute Engine (GCE). The workflow automates instance creation, environment setup, job execution, result upload to Google Cloud Storage (GCS), and instance stop/cleanup.

## Prerequisites

Before using the GCP scripts, ensure you have:

1. **gcloud CLI** installed and authenticated (`gcloud auth login`)
2. **GPU quota** in the target zone (e.g., A100 80GB GPUs in `asia-southeast1-c`). You can check and request quota increases in the [Google Cloud Console](https://console.cloud.google.com/iam-admin/quotas).
3. **A GCS bucket** for storing results:
    ```bash
    gsutil mb gs://your-bucket-name
    ```

## Quick Start

Jobs use `vllm==0.29.0` with Python 3.12. New instances use
`common-cu129-ubuntu-2204-nvidia-580` from the
[Deep Learning VM image catalog](https://docs.cloud.google.com/deep-learning-vm/docs/images).
The default PyPI dependency resolution includes CUDA 13 packages, which require
[NVIDIA driver 580 or newer](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The image's system CUDA toolkit and the Python environment's CUDA runtime can
have different versions; the host driver must support the runtime.

The submission script also includes `vllm-bnb-plugin==0.0.3` in the runtime
environment so that `config/llama30b-exp-ref.yaml` can use BitsAndBytes for both
the target and reference model. vLLM 0.29.0 requires this external plugin;
`bitsandbytes` alone is not sufficient.

Before launching a job, `gcp/check_driver.sh` checks the driver on both new and
reused instances. Older instances using driver 570 must have their driver upgraded
or be replaced with a new instance. A failed preflight leaves the instance running;
stop it manually if you are not going to upgrade it and retry.

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://your-bucket/fast-mia-results \
  --zone ZONE \
  --machine-type a2-ultragpu-1g \
  --accelerator-type nvidia-a100-80gb
```

This command will:

1. Create a GCE instance with an A100 80GB GPU (Deep Learning VM image with CUDA drivers)
2. Transfer the project files to the instance
3. Install `uv` and Python dependencies
4. Run `main.py` with the specified config
5. Upload `results/` to the GCS bucket
6. Stop the instance (preserving model caches for reuse)

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to the YAML configuration file |
| `--bucket` | (required) | GCS bucket URI for results (e.g., `gs://my-bucket/results`) |
| `--project` | gcloud default | GCP project ID |
| `--zone` | `us-central1-b` | GCE zone (e.g., `us-central1-f`, `asia-southeast1-c`) |
| `--machine-type` | `a2-highgpu-1g` | Machine type (see [GPU machine types](https://cloud.google.com/compute/docs/gpus)) |
| `--accelerator-type` | `nvidia-tesla-a100` | GPU type |
| `--accelerator-count` | `1` | Number of GPUs |
| `--boot-disk-size` | `200GB` | Boot disk size |
| `--instance-name` | `fast-mia-job-<timestamp>` | Instance name |
| `--extra-args` | – | Additional arguments for `main.py` |
| `--delete-after` | off | Delete the instance after the job instead of stopping it |

## Examples

### Basic run with LLaMA-30B on A100 80GB

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://my-bucket/fast-mia-results \
  --zone ZONE \
  --machine-type a2-ultragpu-1g \
  --accelerator-type nvidia-a100-80gb
```

### Reusing a stopped instance

When a job completes, the instance is stopped by default. You can reuse it for the next run by specifying `--instance-name` and `--zone`, which skips environment setup and reuses model caches:

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://my-bucket/fast-mia-results \
  --instance-name fast-mia-job-XXXXXXXX-XXXXXX \
  --zone ZONE \
  --machine-type a2-ultragpu-1g \
  --accelerator-type nvidia-a100-80gb
```

### Detailed report with specific project

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://my-bucket/fast-mia-results \
  --project my-gcp-project \
  --zone ZONE \
  --machine-type a2-ultragpu-1g \
  --accelerator-type nvidia-a100-80gb \
  --extra-args "--seed 42 --detailed-report"
```

### Delete instance after job

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://my-bucket/fast-mia-results \
  --zone ZONE \
  --machine-type a2-ultragpu-1g \
  --accelerator-type nvidia-a100-80gb \
  --delete-after
```

### Using A100 40GB for smaller models

For smaller models (e.g., Qwen2.5-0.5B), you can use the default A100 40GB:

```bash
./gcp/submit_job.sh \
  --config config/sample.yaml \
  --bucket gs://my-bucket/fast-mia-results \
  --zone ZONE
```

## Retrieving Results

Results are uploaded to the GCS bucket under the same timestamped directory structure as local runs:

```bash
# List results
gsutil ls gs://your-bucket/fast-mia-results/

# Download results locally
gsutil -m cp -r gs://your-bucket/fast-mia-results/YYYYMMDD-HHMMSS ./results/
```

## Cost Considerations

- **A100 80GB instances** (`a2-ultragpu-1g`) are expensive. The script automatically stops the instance after the job completes to prevent unnecessary charges. Use `--delete-after` to delete it entirely.
- **Stopped instances** still incur disk storage costs (small), but GPU charges stop immediately.
- **Reusing instances** saves time on model downloads and environment setup, which can be significant for large models like LLaMA-30B (~60GB).
- Monitor your spending in the [Google Cloud Console Billing](https://console.cloud.google.com/billing) page.

## Troubleshooting

### Validating the vLLM upgrade

Local unit tests mock vLLM and do not verify CUDA kernels or real model loading.
If reusing a local `uv.lock` from an older setup, refresh the dependencies whose
minimum versions increased (`aiohttp>=3.13.3`, `huggingface-hub>=1.28.0`):

```bash
uv lock --upgrade-package aiohttp --upgrade-package huggingface-hub
```

`uv.lock` is not tracked in this repository; run this in the checkout transferred
to the VM if it contains an old lockfile.
On the GPU VM, start with the small sample configuration:

```bash
bash gcp/check_driver.sh
uv run --with 'vllm==0.29.0' python main.py --config config/sample.yaml --detailed-report
```

Check that scores are finite and the report and plots are produced. Then run your
evaluation configuration, including any reference model, LoRA adapters, and
generation methods you use. These paths require separate GPU validation; existing
benchmark results have not been recomputed for this upgrade.

### GPU quota exceeded

If you see a quota error, request a quota increase for the GPU type in the target zone via the [Quotas page](https://console.cloud.google.com/iam-admin/quotas).

### Zone resource pool exhausted (stockout)

This means the zone has no available GPUs. Try a different zone. Available zones for A100 80GB include `asia-southeast1-c`, `us-central1-c`, and `us-east4-c`.

### NVIDIA driver not ready

The setup script waits up to 5 minutes for NVIDIA drivers to initialize. If it times out, the Deep Learning VM image may not have finished installing drivers. Try re-running with the same `--instance-name` to reuse the instance.

### SSH connection timeout

The script waits up to 10 minutes for the instance to become reachable. If it times out, check that your firewall rules allow SSH (port 22) and that the instance started correctly in the [VM Instances page](https://console.cloud.google.com/compute/instances).
