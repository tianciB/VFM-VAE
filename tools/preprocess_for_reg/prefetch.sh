export PYTHONPATH=$(pwd):$PYTHONPATH
 
torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/preprocess_for_reg/prefetch.py \
  --data-path your_path/tars \
  --output-dir your_path/data \
  --vae-pth your_path/vae_checkpoint.pth \
  --use-config configs/vae_config.yaml