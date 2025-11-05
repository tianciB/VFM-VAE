export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/decode/decode_latents_to_images.py \
  --input-dir your_path/samples \
  --output-dir your_path/images \
  --vae-pth your_path/vae.pth \
  --use-config your_path/vae_config.yaml

torchrun --nnodes=1 --nproc_per_node=8 --standalone tools/decode/decode_latents_to_labels.py \
  --input-dir your_path/samples \
  --output-json your_path/labels.json \
  --class-index-json your_path/imagenet_1k_cls_to_text.json

python tools/decode/save_images_as_npz.py \
  --input-dir your_path/images \
  --output-npz your_path/output.npz \
  --num 50000
