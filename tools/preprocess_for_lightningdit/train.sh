torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py \
    --config train_lightningdit_xl_1_stage_0.yaml

torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py \
    --config train_lightningdit_xl_1_stage_1.yaml