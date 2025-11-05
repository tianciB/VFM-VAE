# 🪞 Decoding Tools

This module provides utilities for decoding **diffusion-generated latents** back into RGB images or evaluation-ready datasets using a pretrained **VFM-VAE** decoder.

---

## 📄 Files

| File | Description |
|:--|:--|
| `decode_latents_to_images.py` | Decode `.safetensors` latent tensors into RGB images using a pretrained VFM-VAE. |
| `decode_latents_to_labels.py` | Decode class labels from latent sample metadata into JSON format. |
| `save_images_as_npz.py` | Pack decoded images into `.npz` format for FID/IS evaluation. |
| `decode.sh` | Example shell script for decoding. |
---

## ⚙️ Usage

### 1. Decode Latent Tensors to Images

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/decode/decode_latents_to_images.py \
  --input-dir your_path/samples \
  --output-dir your_path/generated_images \
  --vae-pth your_path/vfm_vae.pth \
  --use-config configs/vfm_vae.yaml
```

### 2. Decode Class Labels (Optional)

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/decode/decode_latents_to_labels.py \
  --input-dir your_path/samples \
  --output-json your_path/labels.json \
  --class-index-json imagenet_info/imagenet_1k_cls_to_text.json
```

### 3. Pack Images for Evaluation

```bash
python tools/decode/save_images_as_npz.py \
  --input-dir your_path/generated_images \
  --output-npz your_path/generated_images.npz \
  --num 50000
```


The resulting .npz file can be directly used for evaluation, following ADM’s evaluation protocol, [→ Back to Generation Section](../../README.md#generation).
