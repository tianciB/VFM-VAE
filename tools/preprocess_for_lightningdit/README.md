# ⚙️ Preprocessing and Training for LightningDiT

This module provides scripts for adapting [LightningDiT](https://github.com/hustvl/LightningDiT) to the **VFM-VAE** tokenizer.  
It includes latent preprocessing, two-stage diffusion training, and sampling procedures fully compatible with multi-GPU distributed training.
⚠️ **Note:** To successfully run all commands, please replace every occurrence of `your_path` with a valid local path on your system.

---

## 📄 Files

| File | Description |
|:--|:--|
| `prefetch.py` | Extract and encode ImageNet images into `.safetensors` latents using a pretrained VFM-VAE. |
| `train.py` | Modified LightningDiT training script supporting **Accelerate** for gradient accumulation. |
| `sample.py` | Sampling script for decoding latent diffusion outputs (after training). |
| `train_lightningdit_xl_1_stage_0.yaml` | Stage-0 training config (with lognorm sampling). |
| `train_lightningdit_xl_1_stage_1.yaml` | Stage-1 training config (lognorm off, fine-tuning). |
| `*.sh` | Example shell scripts for prefetching, training, and sampling. |
---

## ⚙️ Workflow

### 1. Prefetch Latent Tensors

Before training, convert ImageNet WebDataset `.tar` files into `.safetensors` latents using the pretrained VFM-VAE:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/preprocess_for_lightningdit/prefetch.py \
  --data-path /path/to/tars \
  --output-dir /path/to/data \
  --vae-pth /path/to/vae_checkpoint.pth \
  --use-config configs/vae_config.yaml
```

The extracted `.safetensors` latents can be directly consumed by LightningDiT for diffusion training.
The corresponding latent mean and variance statistics (for channel normalization before diffusion input) are provided in latents_stats.pt, available on [Hugging Face](https://huggingface.co/tiancibi/VFM-VAE/blob/main/checkpoints_imagenet256/diffusions/lightningdit_with_vfm_vae/latents_stats.pt).

---

⚠️ **Note:** Enter the LightningDiT repository before training and sampling.

### 2. Train LightningDiT with VFM-VAE Latents

Replace the original `train.py` with the provided version here to enable Accelerate-based gradient accumulation (important for GPUs with limited memory).

LightningDiT training is conducted in two stages (to match the official training schedule — see [issue](https://github.com/hustvl/LightningDiT/issues/23)):

```bash
# Stage 0: with lognorm sampling
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py \
  --config train_lightningdit_xl_1_stage_0.yaml

# Stage 1: fine-tuning without lognorm sampling
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py \
  --config train_lightningdit_xl_1_stage_1.yaml
```

Pretrained diffusion checkpoints are available on [Hugging Face — VFM-VAE LightningDiT Models](https://huggingface.co/tiancibi/VFM-VAE/tree/main/checkpoints_imagenet256/diffusions/lightningdit_with_vfm_vae).

### 3. Sampling

After training, copy the `sample.py` script into the LightningDiT directory and generate latent samples:

```
torchrun --nnodes=1 --nproc_per_node=8 --standalone sample.py \
  --config train_lightningdit_xl_1_stage_1.yaml
```

Sampling parameters (e.g., steps, guidance, batch size) are defined in the YAML configuration file.