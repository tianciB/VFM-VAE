# 📊 Alignment Analysis (SE-CKNNA)

This module provides scripts for **representation alignment analysis** across **VFMs**, **VAEs**, and **diffusion models**, following the SE-CKNNA protocol (Semantic–Equivariant-CKNNA analysis). It evaluates how consistently different models preserve feature similarity under **semantic-preserving transformations such as rotation, scaling, and additive noise**.

---

## 📄 Files

| File               | Description                                                                                                                                                                                                                                                |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preprocess.py`    | Apply semantic-equivariant transformations (rotation, scale, noise) to clean images.                                                                                                                                                                       |
| `metrics.py`       | Compute SE-CKNNA and related alignment metrics between two feature sets.                                                                                                                                                                                   |
| `requirements.txt` | Dependency list for reproducible evaluation.                                                                                                                                                                                                               |
| `vfms/`            | Feature extraction scripts for Vision Foundation Models (DINOv2, SigLIP2).                                                                                                                                                                                 |
| `vaes/`            | Feature extraction scripts for VAEs (SD-VAE, VA-VAE, VFM-VAE).                                                                                                                                                                                             |
| `diffusions/`      | Feature extraction and latent prefetch scripts for diffusion models ([LightningDiT](https://github.com/hustvl/LightningDiT), [REG](https://github.com/Martinser/REG), [REPA](https://github.com/sihyun-yu/REPA), [SiT](https://github.com/willisma/SiT)). |
| `extract.sh`       | Example pipeline for extracting features from all VFMs, VAEs, and diffusion models.                                                                                                                                                                        |
---

## ⚙️ Workflow

### 1. Prepare Transformed Images

Provide a resized **256×256 clean validation set** (e.g., 10,000 images randomly sampled from the ImageNet validation split).  
Apply semantic-equivariant transformations to obtain the following structure:

```
transformed_images/
  ├── clean/
  ├── noise_0.050/
  ├── noise_0.100/
  ├── noise_0.150/
  ├── noise_0.200/
  ├── equivariance_transforms.json
```

Ready-made transformed images are available on [Hugging Face](https://huggingface.co/tiancibi/VFM-VAE/blob/main/alignment_analysis/transformed_images.tar.gz).

### 2. Feature Extraction

* Run the unified multi-GPU pipeline in `extract.sh` to extract features across VFMs, VAEs, and diffusion models.  
- For **diffusion models**: 
  > features require **prefetching clean images into latent tensors** before extraction.  
  > prefetched features needed **latents_stats.pt** (mean and variance statistics) prior to normalized.  
  > default settings include **linear noise schedule**, **t = 0.5**, and **null label** during feature extraction.  
* All features are averaged by **meaning across patch tokens** and saved in `.pt` format, **sorted by image name** for consistent matching.  
* All detailed commands and parameter examples are included in `extract.sh`.


### 3. Compute CKNNA Metric

After feature extraction:

```bash
pip install -r requirements.txt
python metrics.py --feat-path1 path/to/features_A.pt --feat-path2 path/to/features_B.pt
```

The computation follows [Platonic Representations](https://github.com/minyoungg/platonic-rep).
Pre-extracted features are also available on [Hugging Face](https://huggingface.co/tiancibi/VFM-VAE/tree/main/alignment_analysis/features).
