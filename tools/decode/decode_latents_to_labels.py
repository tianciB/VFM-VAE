# ------------------------------------------------------------------------------
# Multi-GPU label decoding from .safetensors (aligned with image decoding)
# ------------------------------------------------------------------------------

import os
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from safetensors.torch import load_file


# --------------------------- Helpers --------------------------- #
def load_imagenet_classes(json_path):
    """Load ImageNet class index mapping (id → class_name)."""
    with open(json_path, "r") as f:
        class_index = json.load(f)
    return {int(k): v for k, v in class_index.items()}


# --------------------------- Core Logic --------------------------- #
@torch.no_grad()
def extract_labels_ddp(
    input_dir,
    output_json,
    class_index_json,
    rank,
    world_size,
):
    """
    Extract labels from .safetensors files across multiple GPUs.
    Matching the file-splitting logic of run_latent_decoding().
    """
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    class_index = load_imagenet_classes(class_index_json)
    local_map = {}

    # --- Collect all .safetensors files ---
    all_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))
    assert all_files, f"No .safetensors files found in {input_dir}"

    # --- Split files evenly across ranks ---
    files = all_files[rank::world_size]
    print(f"[Rank {rank}] Processing {len(files)} of {len(all_files)} files...")

    global_index = 0
    for file in tqdm(files, desc=f"[Rank {rank}] Extracting labels"):
        data_path = os.path.join(input_dir, file)
        try:
            data = load_file(data_path)
        except Exception as e:
            print(f"⚠️  Failed to load {file}: {e}")
            continue

        if "labels" not in data:
            print(f"⚠️  Missing 'labels' in {file}")
            continue

        labels = data["labels"]
        for i, label_tensor in enumerate(labels):
            label_id = int(label_tensor.item())
            class_name = class_index.get(label_id, "unknown")
            filename = f"rank{rank:02d}_{global_index:06d}.png"
            local_map[filename] = {"label_id": label_id, "class_name": class_name}
            global_index += 1

    # --- Gather results to rank 0 ---
    gathered_maps = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_map, gathered_maps, dst=0)

    # --- Merge & save on rank 0 ---
    if rank == 0:
        merged = {}
        for m in gathered_maps:
            merged.update(m)
        merged = dict(sorted(merged.items()))
        with open(output_json, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"✅ Saved merged label map to {output_json} ({len(merged)} entries)")

    dist.barrier()
    print(f"[Rank {rank}] Done. Processed {len(local_map)} labels.")


# --------------------------- Entry Point --------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-GPU label extraction (aligned with run_latent_decoding).")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .safetensors files.")
    parser.add_argument("--output-json", type=str, required=True, help="Path to save merged label JSON (on rank 0).")
    parser.add_argument("--class-index-json", type=str, required=True, help="Path to ImageNet class index JSON.")
    args = parser.parse_args()

    # --- Initialize DDP ---
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # --- Run label extraction ---
    extract_labels_ddp(
        input_dir=args.input_dir,
        output_json=args.output_json,
        class_index_json=args.class_index_json,
        rank=rank,
        world_size=world_size,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
