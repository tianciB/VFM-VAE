# ⚙️ Preprocessing and Training for REG

This module provides scripts for adapting [REG](https://github.com/Martinser/REG) to the **VFM-VAE** tokenizer.  
It includes latent preprocessing, model modification instructions, diffusion training, and sampling procedures compatible with multi-GPU distributed environments.

⚠️ **Note:** To successfully run all commands, please replace every occurrence of `your_path` with a valid local path on your system.

---

## 📄 Files

| File | Description |
|:--|:--|
| `prefetch.py` | Extract and encode ImageNet images into `.safetensors` latents using a pretrained VFM-VAE. |
| `train.py` | Modified REG training script supporting latent integration and additional training options. |
| `sit.py` | Updated SiT backbone implementation with QK-Norm for stable training. |
| `loss.py` | Adapted for DeepSpeed + mixed-precision. |
| `sample.py` | Sampling script for generating diffusion latents from trained REG models. |
| `*.sh` | Example shell scripts for prefetching, training, and sampling. |
---

## ⚙️ Workflow

### 1. Prefetch Latent Tensors

Before training, convert ImageNet WebDataset `.tar` files into `.safetensors` latents using the pretrained VFM-VAE:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/preprocess_for_reg/prefetch.py \
  --data-path your_path/tars \
  --output-dir your_path/data \
  --vae-pth your_path/vae_checkpoint.pth \
  --use-config configs/vae_config.yaml
```

The extracted `.safetensors` latents can be directly consumed by REG for diffusion training.
The corresponding latent mean and variance statistics (for channel normalization before diffusion input) are provided in latents_stats.pt, available on [Hugging Face](https://huggingface.co/tiancibi/VFM-VAE/blob/main/checkpoints_imagenet256/diffusions/reg_with_vfm_vae/latents_stats.pt).

---

⚠️ **Note:** Enter the REG repository before training and sampling.

### 2. Train REG with VFM-VAE Latents

Before launching training, replace the following files in the REG repository:

 - `models/sit.py` → use the version provided here (adds SiT-XL/1, QK-Norm support).

 - `train.py` → use the version here to include latent-related parameters (latent size, dimension, and encoder configs).

 - `loss.py` → use the version here to adapt for DeepSpeed + mixed-precision.

REG training is conducted with the following recommended configuration:

```bash
accelerate launch --multi_gpu --num_processes 8 train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/1" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="your_path/output" \
  --exp-name="reg_sit_xl_1_with_vfm_vae" \
  --batch-size=1024 \
  --num-workers 8 \
  --learning-rate 2e-4 \
  --adam-beta2 0.95 \
  --max-train-steps 800_000 \
  --checkpointing-steps 10_000 \
  --data-dir="your_path/data" \
  --latent-size=16 \
  --vae-latent-dim=32 \
  --cls=0.03 \
  --qk-norm
```
Key settings:
 - `batch-size = 1024`
 - `learning-rate = 2e-4`
 - `adam-beta2 = 0.95`
 - Long-term stable training enabled via `QK-Norm`
 - Fast training enabled via `DeepSpeed`

⚠️ **Note:** `batch-size` is the batch size per micro forward, `batch-size-per-gpu` is the batch size per gpu per micro forward, aligned with REG.

Pretrained REG diffusion checkpoints are available on [Hugging Face — VFM-VAE REG Models](https://huggingface.co/tiancibi/VFM-VAE/tree/main/checkpoints_imagenet256/diffusions/reg_with_vfm_vae).

### 3. Sampling

After training, copy the provided `sample.py` to the REG repository and generate latent samples:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone sample.py \
    --global-seed=0 \
    --ckpt="your_path/reg_checkpoint.pt" \
    --sample-dir="your_path/samples" \
    --latents-stats-dir="your_path/latents_stats" \
    --model="SiT-XL/1" \
    --num-classes=1000 \
    --path-type=linear \
    --encoder-depth=8 \
    --projector-embed-dims=768 \
    --resolution=256 \
    --latent-size=16 \
    --vae-latent-dim=32 \
    --per-proc-batch-size 64 \
    --mode="sde" \
    --cfg-scale=4.0 \
    --cls-cfg-scale=4.0 \
    --guidance-high=0.8 \
    --num-steps=250 \
    --cls=768 \
    --qk-norm

# cfg scale:
#   for 64  epochs, FID=2.03 with cfg scale=2.0
#   for 480 epochs, FID=1.34 with cfg scale=4.0
#   for 640 epochs, FID=1.31 with cfg scale=4.0
```

Sampling parameters (e.g., steps, guidance, batch size) are defined in the YAML configuration file.
We also provide pre-generated latent samples on [Hugging Face](https://huggingface.co/tiancibi/VFM-VAE/tree/main/samples_50k/reg_with_vfm_vae) for direct decoding via [tools/decode](https://github.com/tianciB/VFM-VAE/tree/main/tools/decode).
