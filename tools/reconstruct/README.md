# 🧩 Reconstruction Tools

This module provides scripts for **dataset extraction**, **image reconstruction**, and **quality evaluation** using pretrained **VFM-VAE** models.

---

## 📄 Files

| File | Description |
|:--|:--|
| `extract.py` | Decode WebDataset (`.tar`) shards (e.g., ImageNet-1K WDS) into standard image folders. |
| `reconstruct.py` | Reconstruct images from RGB folders using pretrained VFM-VAE checkpoints. Supports multi-GPU inference. |
| `evaluate.py` | Evaluate reconstruction quality using LPIPS, PSNR, and SSIM metrics. |
| `reconstruct.sh` | Example shell script for reconstruction. |
---

## ⚙️ Usage

### 1. Extract validation images
```bash
python tools/reconstruct/extract.py \
  --input-dir your_path/imagenet_1k_wds_validation \
  --output-dir your_path/imagenet_val_images
```

### 2. Reconstruct images

```bash
torchrun --nproc_per_node=8 tools/reconstruct/reconstruct.py \
  --input-dir your_path/imagenet_val_images \
  --output-dir your_path/reconstructed_images \
  --vae-pth your_path/vfm_vae.pth \
  --use-config configs/vfm_vae.yaml
```

### 3. Evaluate reconstruction metrics

```
python tools/reconstruct/evaluate.py \
  --ref-dir your_path/reconstructed_images/inputs \
  --pred-dir your_path/reconstructed_images/outputs
```