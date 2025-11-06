export PYTHONPATH=$(pwd):$PYTHONPATH

# 1. Extract VFM features for transformed images

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/evaluate_alignment/vfms/extract_features_by_dinov2.py \
    --input-dir your_path/transformed_images \
    --output-dir your_path/outputs \
    --output-prefix dinov2_{giant, large, base} \
    --model-size {giant, large, base} \
    --mode {clean, equivariance, noise}

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/evaluate_alignment/vfms/extract_features_by_siglip2_large.py \
    --input-dir your_path/transformed_images \
    --output-dir your_path/outputs \
    --output-prefix siglip2_large \
    --model-path your_path/huggingface/siglip2-large-patch16-512/ \
    --resolution 512 \
    --mode {clean, equivariance, noise}

# 2. Extract VAE features for transformed images

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/evaluate_alignment/vaes/extract_features_by_sd_vae.py \
    --input-dir your_path/transformed_images \
    --output-dir your_path/outputs \
    --output-prefix sd_vae \
    --mode {clean, equivariance, noise}

# used in LightningDiT Repository
torchrun --nnodes=1 --nproc_per_node=8 --standalone extract_features_by_va_vae.py \
    --input-dir your_path/transformed_images \
    --output-dir your_path/outputs \
    --output-prefix va_vae_dinov2 \
    --use-config vavae/configs/f16d32_vfdinov2.yaml \
    --mode {clean, equivariance, noise}

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/evaluate_alignment/vaes/extract_features_by_vfm_vae.py \
    --input-dir your_path/transformed_images \
    --output-dir your_path/outputs \
    --output-prefix vfm_vae_40m \
    --vae-pth your_path/vfm_vae.pth \
    --use-config configs/vae_config.yaml \
    --resolution 256 \
    --mode {clean, equivariance, noise}

# 3. Extract Diffusion features for transformed images

## (1) Prefetching examples for diffusion models

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/evaluate_alignment/diffusions/prefetch_for_diffusion_by_sd_vae.py \
    --input-dir your_path/transformed_images/clean \
    --output-dir your_path/outputs

# used in LightningDiT Repository
torchrun --nnodes=1 --nproc_per_node=8 --standalone prefetch_for_diffusion_by_va_vae.py \
    --input-dir your_path/transformed_images/clean \
    --output-dir your_path/outputs/ \
    --use-config vavae/configs/f16d32_vfdinov2.yaml

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/evaluate_alignment/diffusions/prefetch_for_diffusion_by_vfm_vae.py \
    --input-dir your_path/transformed_images/clean \
    --output-dir your_path/outputs/ \
    --vae-pth your_path/vfm_vae.pth \
    --use-config configs/vae_config.yaml

## (2) Get block features for diffusion models (Below are examples)

# used in LightningDiT Repository
torchrun --nnodes=1 --nproc_per_node=8 --standalone get_block_features_from_lightningdit.py \
    --input-dir your_path/prefetched_latents \
    --output-prefix your_path/outputs/lightningdit_ \
    --use-config configs/lightningdit_config.yaml

# used in REG Repository
torchrun --nnodes=1 --nproc_per_node=8 --standalone get_block_features_from_reg.py \
    --input-dir=your_path/prefetched_latents \
    --output-prefix=your_path/outputs/reg_ \
    --checkpoint-path="your_path/reg_checkpoint.pt" \
    --model={"SiT-XL/1", "SiT-XL/2"} \
    --latent-size=your_vae_latent_size \
    --vae-latent-dim=your_vae_latent_dim

# used in REPA Repository
torchrun --nnodes=1 --nproc_per_node=8 --standalone get_block_features_from_repa.py \
    --input-dir=your_path/prefetched_latents \
    --output-prefix=your_path/outputs/repa_sit_xl_2_ \
    --checkpoint-path="your_path/repa_checkpoint.pt" \
    --model="SiT-XL/2"

# used in SiT Repository
torchrun --nnodes=1 --nproc_per_node=8 --standalone get_block_features_from_sit.py \
    --input-dir=your_path/prefetched_latents \
    --output-prefix=your_path/outputs/sit_1400ep_with_sd_vae_use_label/sit_xl_2_ \
    --checkpoint-path=your_path/sit_checkpoint.pt \
    --model="SiT-XL/2"