export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/reconstruct/reconstruct.py \
  --input-dir your_path/inputs \
  --output-dir your_path/outputs \
  --vae-pth your_path/vae.pth \
  --use-config configs/vae_config.yaml
