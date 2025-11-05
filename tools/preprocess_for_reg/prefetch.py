# ------------------------------------------------------------------------------
# Multi-GPU Image Saving and Latent Prefetching for REG using WebDataset
# ------------------------------------------------------------------------------


import os
import sys
import yaml
import dnnlib
import random
import argparse
import warnings
import numpy as np
import torch
import torch.distributed as dist
import PIL.Image
import webdataset as wds
import subprocess

from tqdm import tqdm
from glob import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ------------------------------------------------------------------------------ #
# Environment setup
# ------------------------------------------------------------------------------ #
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def mean_logvar_to_mean_std(moments: torch.Tensor) -> torch.Tensor:
    """Convert (mean || logvar) into (mean || std)."""
    mean, logvar = torch.chunk(moments, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    return torch.cat([mean, std], dim=1)


def log_and_continue(exn) -> bool:
    """Log error and continue processing."""
    print(f"Handling webdataset error ({type(exn).__name__}): {exn}")
    return True


def center_crop_imagenet(image_size: int, arr: np.ndarray):
    """Center cropping implementation from ADM (Dhariwal et al.)."""
    pil_image = PIL.Image.fromarray(arr)
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def preprocess_image_adm(img: PIL.Image.Image, resolution: int) -> np.ndarray:
    """Preprocess image using ADM's center-crop-dhariwal method."""
    arr = np.array(img)
    cropped = center_crop_imagenet(resolution, arr)
    cropped = cropped.transpose(2, 0, 1)
    return cropped.astype(np.uint8)


def preprocess_batch_vae(images: torch.Tensor) -> torch.Tensor:
    """Preprocess batch for VAE input: [B,3,H,W] uint8 → float [0,1]."""
    if images.dtype == torch.uint8:
        images = images.float().div(255.0)
    return images


# ------------------------------------------------------------------------------
# WebDataset Loader
# ------------------------------------------------------------------------------

def create_image_vae_dataloader(
    train_data: list[str],
    *,
    batch_size_per_gpu: int,
    resolution: int = 256,
    workers: int = 4,
    one_epoch: bool = True,
) -> wds.WebLoader:
    """Create WebDataset dataloader for image + label extraction."""
    source = wds.SimpleShardList(train_data) if one_epoch else wds.ResampledShards(train_data)
    pipeline = [
        source,
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", label="cls", key="__key__"),
        wds.map_dict(image=lambda img: preprocess_image_adm(img, resolution)),
        wds.to_tuple("image", "label", "key"),
        wds.batched(batch_size_per_gpu),
    ]
    dataset = wds.DataPipeline(*pipeline)
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=workers)
    return loader


# ------------------------------------------------------------------------------
# Save Functions
# ------------------------------------------------------------------------------

def save_images_as_png(images, keys, labels, output_dir):
    """Save images as PNG under structured subfolders."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for img, key, label in zip(images, keys, labels):
        subfolder = key.split("_")[0]
        folder = os.path.join(output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{key}.png")

        # --- FIX: handle both Tensor and numpy correctly ---
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        elif img.ndim != 3 or img.shape[2] not in [1, 3]:
            print(f"[WARN] Unexpected shape {img.shape} for {key}")
            continue
        # ---------------------------------------------------

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        PIL.Image.fromarray(img).save(path, "PNG")
        records.append([f"{subfolder}/{key}.png", int(label)])
    return records


def save_vae_latents(features, keys, labels, output_dir):
    """Save VAE latent features as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for feat, key, label in zip(features, keys, labels):
        subfolder = key.split("_")[0]
        folder = os.path.join(output_dir, subfolder)
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, f"{key}.npy"), feat.astype(np.float32))
        records.append([f"{subfolder}/{key}.npy", int(label)])
    return records


def save_json(records, path):
    """Save label mappings as JSON (fixed quotes)."""
    with open(path, "w") as f:
        f.write('{"labels": [\n')
        for i, rec in enumerate(records):
            # Convert Python repr to JSON-like string
            line = f'  ["{rec[0]}", {rec[1]}]'
            if i < len(records) - 1:
                f.write(f"{line},\n")
            else:
                f.write(f"{line}\n")
        f.write("]}\n")


# ------------------------------------------------------------------------------
# Compute VAE Statistics
# ------------------------------------------------------------------------------

def compute_vae_stats_sampled(vae_dir, num_samples=10000):
    """Compute latent mean/std from random 10k samples."""
    npy_files = [os.path.join(r, f)
                 for r, _, files in os.walk(vae_dir)
                 for f in files if f.endswith(".npy")]
    if len(npy_files) == 0:
        print(f"[WARN] No latent files found in {vae_dir}")
        return

    sampled = random.sample(npy_files, min(num_samples, len(npy_files)))
    feats = []
    for fp in tqdm(sampled, desc="Computing latent stats"):
        try:
            z = np.load(fp)
            z = torch.from_numpy(z)
            mean, std = torch.chunk(z, 2, dim=0)
            z_sample = mean + std * torch.randn_like(mean)
            feats.append(z_sample.unsqueeze(0))
        except Exception as e:
            print(f"Error reading {fp}: {e}")
    if not feats:
        return
    feats = torch.cat(feats, dim=0)
    stats = {"mean": feats.mean(dim=[0, 2, 3], keepdim=True),
             "std": feats.std(dim=[0, 2, 3], keepdim=True)}
    torch.save(stats, os.path.join(vae_dir, "latents_stats.pt"))
    print(f"Saved VAE stats to {vae_dir}/latents_stats.pt")


# ------------------------------------------------------------------------------
# Main Extraction Loop
# ------------------------------------------------------------------------------

@torch.no_grad()
def run_extraction(
    vae_model,
    data_path,
    output_dir,
    rank,
    device,
    resolution,
    batch_size_per_gpu,
    process_images=True,
    images_folder_name="images_png",
    vae_folder_name="vae_latents",
    max_images_per_gpu=None,
):
    """Main extraction loop: image + latent saving."""
    images_dir = os.path.join(output_dir, images_folder_name)
    vae_dir = os.path.join(output_dir, vae_folder_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)

    urls = sorted(glob(os.path.join(data_path, "*.tar"))) if os.path.isdir(data_path) else [data_path]
    print(f"Rank {rank}: Found {len(urls)} shards")

    dataloader = create_image_vae_dataloader(
        train_data=urls, batch_size_per_gpu=batch_size_per_gpu, resolution=resolution
    )

    all_images, all_latents = [], []
    total_processed = 0

    for image_batch, label_batch, key_batch in tqdm(dataloader, desc=f"[Rank {rank}] Extracting"):
        if max_images_per_gpu and total_processed >= max_images_per_gpu:
            break

        remain = None
        if max_images_per_gpu:
            remain = max_images_per_gpu - total_processed
            if remain <= 0:
                break
            image_batch = image_batch[:remain]
            label_batch = label_batch[:remain]
            key_batch = key_batch[:remain]

        vae_model.eval()
        inputs = preprocess_batch_vae(image_batch.to(device))
        latents = mean_logvar_to_mean_std(vae_model.encode(inputs, return_z_before_quantize=True))
        latents = latents.cpu().numpy().astype(np.float32)

        if process_images:
            records_img = save_images_as_png(image_batch.numpy(), key_batch, label_batch, images_dir)
            all_images.extend(records_img)

        records_latent = save_vae_latents(latents, key_batch, label_batch, vae_dir)
        all_latents.extend(records_latent)

        total_processed += len(label_batch)

    # Save per-rank JSON
    save_json(all_latents, os.path.join(vae_dir, f"dataset_rank{rank}.json"))
    if process_images:
        save_json(all_images, os.path.join(images_dir, f"dataset_rank{rank}.json"))

    dist.barrier()

    if rank == 0:
        # Merge JSONs
        def merge_json(folder, final_name):
            records = []
            for f in os.listdir(folder):
                if f.startswith("dataset_rank") and f.endswith(".json"):
                    with open(os.path.join(folder, f), "r") as jf:
                        data = eval(jf.read().split("=", 1)[-1])
                    records.extend(data.get("labels", []))
            save_json(records, os.path.join(folder, final_name))

        merge_json(vae_dir, "dataset.json")
        if process_images:
            merge_json(images_dir, "dataset.json")

        compute_vae_stats_sampled(vae_dir)
        
        # Optional: tar the folders
        # subprocess.run(["tar", "cf", f"{vae_dir}.tar", "-C", vae_dir, "."], check=True)
        # if process_images:
        #     subprocess.run(["tar", "cf", f"{images_dir}.tar", "-C", images_dir, "."], check=True)

    return total_processed


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract images and VAE latents using WebDataset (.pth)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to WebDataset .tar files or folder")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for extracted data")
    parser.add_argument("--vae-pth", type=str, required=True, help="Path to pretrained VAE checkpoint (.pth)")
    parser.add_argument("--use-config", type=str, required=True, help="Path to YAML config defining model")
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--max-images-per-gpu", type=int, default=None)
    parser.add_argument("--no-images", action="store_true", help="Skip PNG saving")
    parser.add_argument("--images-folder-name", type=str, default="images")
    parser.add_argument("--vae-folder-name", type=str, default="vae_latents")
    args = parser.parse_args()

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"[Init] Rank {rank}/{world_size} ready on device {device}.")

    # Load YAML configuration
    with open(args.use_config, "r") as f:
        full_cfg = yaml.safe_load(f)
    vae_kwargs = full_cfg.get("generator_kwargs", full_cfg.get("G_kwargs", {}))
    vae_kwargs["label_dim"] = 1000                  # default for ImageNet
    vae_kwargs["img_resolution"] = args.resolution  # set resolution
    vae_kwargs["conditional"] = False               # reconstruction is unconditional
    vae_kwargs["label_type"] = "cls2text"           # dummy, not used

    # Build model
    vae_model = dnnlib.util.construct_class_by_name(**vae_kwargs).to(device)
    vae_model.requires_grad_(False)

    # Load checkpoint (.pth)
    print(f"[Rank {rank}] Loading checkpoint: {args.vae_pth}")
    ckpt = torch.load(args.vae_pth, map_location=device)
    if isinstance(ckpt, dict) and "G_ema" in ckpt:
        print("Found 'G_ema' in checkpoint, loading its weights...")
        state_dict = ckpt["G_ema"]
    elif isinstance(ckpt, dict):
        print("No 'G_ema' found, loading full state dict.")
        state_dict = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")
    vae_model.load_state_dict(state_dict, strict=False)
    vae_model.eval()

    # Extraction
    total = run_extraction(
        vae_model=vae_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        rank=rank,
        device=device,
        resolution=args.image_resolution,
        batch_size_per_gpu=args.batch_size_per_gpu,
        process_images=not args.no_images,
        images_folder_name=args.images_folder_name,
        vae_folder_name=args.vae_folder_name,
        max_images_per_gpu=args.max_images_per_gpu,
    )

    print(f"✅ [Rank {rank}] Processed {total} samples.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
