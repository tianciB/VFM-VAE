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