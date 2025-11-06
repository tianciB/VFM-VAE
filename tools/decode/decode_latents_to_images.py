# ------------------------------------------------------------------------------
# Multi-GPU image decoding from .safetensors
# ------------------------------------------------------------------------------

import os
import yaml
import dill
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from safetensors.torch import load_file
from torchvision.transforms.functional import to_pil_image
from torch_utils import misc
import dnnlib


# --------------------------- Utils --------------------------- #
def safe_save(tensor, path):
    """Safely save a tensor as a PNG image."""
    img = to_pil_image(tensor.clamp(0, 1))
    img.save(path)
    img.close()


@torch.no_grad()
def run_latent_decoding(
    vae_decoder,
    input_dir,
    output_dir,
    batch_size_per_gpu,
    rank,
    world_size,
    device,
    max_images_per_gpu=None,
):
    """
    Decode latent .safetensors files into images using a pretrained VAE decoder.
    Each GPU handles a subset of files based on rank (same split as label extractor).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Collect all .safetensors files ---
    all_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))
    assert all_files, f"No .safetensors files found in {input_dir}"

    # --- Split files evenly across ranks ---
    files = all_files[rank::world_size]
    print(f"[Rank {rank}] Processing {len(files)} of {len(all_files)} files...")

    global_index = 0
    saved_count = 0

    for file in tqdm(files, desc=f"[Rank {rank}] Decoding"):
        if max_images_per_gpu is not None and saved_count >= max_images_per_gpu:
            break

        data_path = os.path.join(input_dir, file)
        try:
            data = load_file(data_path)
        except Exception as e:
            print(f"⚠️  Failed to load {file}: {e}")
            continue

        if "latents" not in data:
            print(f"⚠️  Missing 'latents' in {file}")
            continue

        latents = data["latents"].to(device)
        labels = data.get("labels", torch.zeros(latents.size(0), device=device))

        # --- Handle conditional decoding ---
        if getattr(vae_decoder, "conditional", False):
            if getattr(vae_decoder, "label_type", None) == "cls2id":
                labels = F.one_hot(labels.long(), num_classes=vae_decoder.c_dim).float()
            else:
                # Extend for text-conditional or CLIP-conditional here if needed
                pass

        # --- Batch decode ---
        for start in range(0, latents.size(0), batch_size_per_gpu):
            if max_images_per_gpu is not None and saved_count >= max_images_per_gpu:
                break

            end = min(start + batch_size_per_gpu, latents.size(0))
            batch_latents = latents[start:end]
            batch_labels = labels[start:end]

            with torch.cuda.amp.autocast(enabled=False):
                images = vae_decoder.decode(batch_latents, batch_labels)
                images = ((images + 1) / 2).clamp(0, 1)

            for i, img_tensor in enumerate(images):
                if max_images_per_gpu is not None and saved_count >= max_images_per_gpu:
                    break
                index = global_index + i
                out_path = os.path.join(output_dir, f"rank{rank:02d}_{index:06d}.png")
                safe_save(img_tensor.cpu(), out_path)
                saved_count += 1

            global_index += images.size(0)

    dist.barrier()
    print(f"[Rank {rank}] Done. Saved {saved_count} images.")


# --------------------------- Main --------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Decode latent .safetensors files using .pth VAE model weights.")
    parser.add_argument("--input-dir",          type=str, required=True, help="Directory containing .safetensors latent files.")
    parser.add_argument("--output-dir",         type=str, required=True, help="Directory to save decoded images.")
    parser.add_argument("--vae-pth",            type=str, required=True, help="Path to pretrained VAE checkpoint (.pth).")
    parser.add_argument("--use-config",         type=str, required=True, help="Path to model config (YAML).")
    parser.add_argument("--batch-size-per-gpu", type=int, default=32, help="Batch size per GPU for decoding.")
    parser.add_argument("--max-images-per-gpu", type=int, default=None, help="Maximum number of images to decode per GPU.")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.use_config, "r") as f:
        full_cfg = yaml.safe_load(f)

    # Extract model construction parameters
    vae_kwargs = full_cfg.get("G_kwargs", {})
    vae_kwargs["label_dim"] = 1000                  # default for ImageNet
    vae_kwargs["img_resolution"] = 256              # default resolution  
    vae_kwargs["conditional"] = False               # reconstruction is unconditional
    vae_kwargs["label_type"] = "cls2text"           # dummy, not used
    vae_kwargs["num_fp16_res"] = 0                  # disable fp16 for validation

    # --- Initialize DDP ---
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # --- Build VAE decoder ---
    vae_decoder = dnnlib.util.construct_class_by_name(**vae_kwargs).to(device)
    vae_decoder.requires_grad_(False)

    # --- Load .pth weights ---
    print(f"🔍 Loading checkpoint: {args.vae_pth}")
    checkpoint = torch.load(args.vae_pth, map_location=device)
    if isinstance(checkpoint, dict) and "G_ema" in checkpoint:
        print("✅ Found 'G_ema' in checkpoint, loading its weights...")
        state_dict = checkpoint["G_ema"]
    elif isinstance(checkpoint, dict):
        print("⚠️ No 'G_ema' found, using entire checkpoint as state_dict.")
        state_dict = checkpoint
    else:
        raise TypeError(f"❌ Unexpected checkpoint type: {type(checkpoint)}")

    vae_decoder.load_state_dict(state_dict, strict=False)
    vae_decoder.eval()

    # --- Run decoding ---
    run_latent_decoding(
        vae_decoder=vae_decoder,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size_per_gpu=args.batch_size_per_gpu,
        rank=rank,
        world_size=world_size,
        device=device,
        max_images_per_gpu=args.max_images_per_gpu,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
