# Running on Google Cloud

Fast-MIA provides scripts to submit GPU evaluation jobs to Google Compute Engine (GCE). The workflow automates instance creation, environment setup, job execution, result upload to Google Cloud Storage (GCS), and instance cleanup.

## Prerequisites

Before using the GCP scripts, ensure you have:

1. **gcloud CLI** installed and authenticated (`gcloud auth login`)
2. **GPU quota** in the target zone (e.g., A100 GPUs in `us-central1-a`). You can check and request quota increases in the [Google Cloud Console](https://console.cloud.google.com/iam-admin/quotas).
3. **A GCS bucket** for storing results:
    ```bash
    gsutil mb gs://your-bucket-name
    ```

## Quick Start

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://your-bucket/fast-mia-results
```

This command will:

1. Create a GCE instance with an A100 GPU (Deep Learning VM image with CUDA drivers)
2. Transfer the project files to the instance
3. Install `uv` and Python dependencies
4. Run `main.py` with the specified config
5. Upload `results/` to the GCS bucket
6. Delete the instance

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to the YAML configuration file |
| `--bucket` | (required) | GCS bucket URI for results (e.g., `gs://my-bucket/results`) |
| `--project` | gcloud default | GCP project ID |
| `--zone` | `us-central1-a` | GCE zone |
| `--machine-type` | `a2-highgpu-1g` | Machine type (see [GPU machine types](https://cloud.google.com/compute/docs/gpus)) |
| `--accelerator-type` | `nvidia-tesla-a100` | GPU type |
| `--accelerator-count` | `1` | Number of GPUs |
| `--boot-disk-size` | `200GB` | Boot disk size |
| `--instance-name` | `fast-mia-job-<timestamp>` | Instance name |
| `--extra-args` | – | Additional arguments for `main.py` |
| `--keep-instance` | off | Keep the instance alive after the job |

## Examples

### Basic run with LLaMA-30B on A100

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://my-bucket/fast-mia-results
```

### Detailed report with specific project and zone

```bash
./gcp/submit_job.sh \
  --config config/llama30b-exp.yaml \
  --bucket gs://my-bucket/fast-mia-results \
  --project my-gcp-project \
  --zone us-west1-b \
  --extra-args "--seed 42 --detailed-report"
```

### Keep the instance for debugging

```bash
./gcp/submit_job.sh \
  --config config/sample.yaml \
  --bucket gs://my-bucket/fast-mia-results \
  --keep-instance
```

After debugging, delete it manually:

```bash
gcloud compute instances delete fast-mia-job-XXXXXXXX-XXXXXX --zone=us-central1-a --quiet
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

- **A100 instances** (`a2-highgpu-1g`) are expensive. The script automatically deletes the instance after the job completes unless `--keep-instance` is specified.
- For smaller models (e.g., Qwen2.5-0.5B), consider using cheaper GPU types:
    ```bash
    ./gcp/submit_job.sh \
      --config config/sample.yaml \
      --bucket gs://my-bucket/fast-mia-results \
      --machine-type n1-standard-8 \
      --accelerator-type nvidia-tesla-t4
    ```
- Monitor your spending in the [Google Cloud Console Billing](https://console.cloud.google.com/billing) page.

## Troubleshooting

### GPU quota exceeded

If you see a quota error, request a quota increase for the GPU type in the target zone via the [Quotas page](https://console.cloud.google.com/iam-admin/quotas).

### NVIDIA driver not ready

The setup script waits up to 5 minutes for NVIDIA drivers to initialize. If it times out, the Deep Learning VM image may not have finished installing drivers. Try re-running or using `--keep-instance` to SSH in and debug.

### SSH connection timeout

The script waits up to 10 minutes for the instance to become reachable. If it times out, check that your firewall rules allow SSH (port 22) and that the instance started correctly in the [VM Instances page](https://console.cloud.google.com/compute/instances).
