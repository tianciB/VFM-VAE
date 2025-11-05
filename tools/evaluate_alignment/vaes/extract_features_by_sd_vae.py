import os
import re
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


# =====================================
# VAE Loader
# =====================================
class LatentVAE(torch.nn.Module):
    """Load SD-VAE and provide latent encoding."""

    def __init__(self, weight_path: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        from diffusers.models import AutoencoderKL
        print(f"🔹 Loading SD-VAE from: {weight_path}")
        self.model = AutoencoderKL.from_pretrained(weight_path)
        self.scaling_factor = getattr(self.model.config, "scaling_factor", 0.18215)

    @torch.no_grad()
    def encode(self, x, apply_scaling: bool = False):
        """Encode image into latent representation."""
        z = self.model.encode(x).latent_dist.sample()  # [B, 4, H/8, W/8]
        if apply_scaling:
            z *= self.scaling_factor
        return z


# =====================================
# Dataset Definitions
# =====================================
class CleanImageDataset(Dataset):
    """Load clean images from <input_dir>/clean for encoding."""

    def __init__(self, input_dir, resolution=256):
        self.input_dir = Path(input_dir) / "clean"
        self.image_paths = sorted(self.input_dir.glob("*.png"), key=lambda p: p.stem)
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"image": img, "name": path.stem}


class EquivarianceDataset(Dataset):
    """Load image-transform pairs for equivariance analysis."""

    def __init__(self, input_dir, transform_json_path, resolution=256):
        self.input_dir = Path(input_dir) / "clean"
        with open(transform_json_path, "r") as f:
            self.transform_records = json.load(f)
        self.image_names = sorted(self.transform_records.keys())
        self.resolution = resolution
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        record = self.transform_records[name]
        path = self.input_dir / f"{name}.png"

        img = Image.open(path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # scaling
        scale = record.get("scale", 1.0)
        target_size = int(self.resolution * scale)
        img = F.interpolate(img.unsqueeze(0), size=(target_size, target_size),
                            mode="bilinear", align_corners=False)[0]

        # rotation
        rot = record.get("rotation", 0)
        if rot in [90, 180, 270]:
            img = torch.rot90(img, rot // 90, dims=[-2, -1])
                        
        img = self.normalize(img)

        transformation = {"mode": "equivariance", "rotation": rot, "scale": scale}
        return {"image": img, "name": name, "transformation": transformation}


class ProcessedImageDataset(Dataset):
    """Load processed (noise/mask) images for feature extraction."""

    def __init__(self, processed_dir, resolution=256):
        self.image_paths = sorted(Path(processed_dir).glob("*.png"), key=lambda p: p.stem)
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"image": img, "name": path.stem}


# =====================================
# Distributed Utilities
# =====================================
def initialize_distributed():
    """Initialize torch.distributed if environment variables are set."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu_id = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(gpu_id)
    else:
        rank, world_size, gpu_id = 0, 1, 0
    return rank, world_size, gpu_id


def cleanup_distributed():
    """Safely shut down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_results(all_latents, all_names, all_transforms, rank, world_size):
    """Gather results from all distributed ranks."""
    if world_size > 1:
        gathered_latents = [None for _ in range(world_size)]
        gathered_names = [None for _ in range(world_size)]
        gathered_transforms = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_latents, all_latents)
        dist.all_gather_object(gathered_names, all_names)
        dist.all_gather_object(gathered_transforms, all_transforms)

        if rank == 0:
            latents_list, names_list, trans_list = [], [], []
            for l, n, t in zip(gathered_latents, gathered_names, gathered_transforms):
                if isinstance(l, torch.Tensor) and l.numel() > 0:
                    latents_list.append(l)
                if n:
                    names_list.extend(n)
                if t:
                    trans_list.extend(t)
            all_latents = torch.cat(latents_list, dim=0) if latents_list else torch.empty(0)
            all_names, all_transforms = names_list, trans_list
        else:
            all_latents, all_names, all_transforms = torch.empty(0), [], []
    return all_latents, all_names, all_transforms


# =====================================
# Feature Extraction
# =====================================
@torch.no_grad()
def extract_features_common(dataset, vae, device, args, rank, world_size):
    """Extract SD-VAE latent features from a given dataset."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False) if world_size > 1 else None
    loader = DataLoader(dataset,
                        batch_size=1 if args.mode == "equivariance" else args.batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                        num_workers=args.num_workers,
                        pin_memory=True)

    all_latents, all_names, all_transforms = [], [], []
    for batch in tqdm(loader, disable=(rank != 0)):
        imgs = batch["image"].to(device, non_blocking=True)
        z = vae.encode(imgs, apply_scaling=args.apply_scaling)
        z = z.mean(dim=[-2, -1])  # global average pooling in latent space
        all_latents.append(z.cpu())
        all_names.extend(batch["name"])

        trans = batch.get("transformation")
        if trans is None:
            all_transforms.extend([{"mode": args.mode}] * len(batch["name"]))
        elif isinstance(trans, list):
            all_transforms.extend([dict(t) for t in trans])
        elif isinstance(trans, dict):
            all_transforms.append(dict(trans))
        else:
            raise TypeError(f"Unexpected transformation type: {type(trans)}")

    if all_latents:
        all_latents = torch.cat(all_latents, dim=0)
    return all_latents, all_names, all_transforms


def extract_features_clean(args, vae, device, rank, world_size):
    dataset = CleanImageDataset(args.input_dir, args.resolution)
    return extract_features_common(dataset, vae, device, args, rank, world_size)


def extract_features_equivariance(args, vae, device, rank, world_size):
    transform_json_path = Path(args.input_dir) / "equivariance_transforms.json"
    if not transform_json_path.exists():
        return torch.empty(0), [], []
    dataset = EquivarianceDataset(args.input_dir, transform_json_path, args.resolution)
    return extract_features_common(dataset, vae, device, args, rank, world_size)


def extract_features_processed(args, vae, device, rank, world_size, processed_dir, mode, level):
    dataset = ProcessedImageDataset(processed_dir, args.resolution)
    if len(dataset) == 0:
        return torch.empty(0), [], []
    all_latents, all_names, all_transforms = extract_features_common(dataset, vae, device, args, rank, world_size)
    if mode == "noise":
        all_transforms = [{"mode": mode, "noise_level": level} for _ in all_names]
    else:
        all_transforms = [{"mode": mode, "mask_ratio": level} for _ in all_names]
    return all_latents, all_names, all_transforms


# =====================================
# Save Utilities
# =====================================
def save_features(all_latents, all_names, all_transforms, output_path, args, mode, extra_info=None):
    save_dict = {
        "features": all_latents,
        "names": all_names,
        "transformations": all_transforms,
        "apply_scaling": args.apply_scaling,
        "num_images": all_latents.shape[0],
        "sorted": True,
    }
    if extra_info:
        save_dict.update(extra_info)
    torch.save(save_dict, output_path)
    print(f"✅ Saved {all_latents.shape[0]} latents to {output_path}")
    if all_latents.shape[0] > 0:
        print(f"Latent shape: {tuple(all_latents.shape)}")
        print("🔹 First 5 samples:")
        for i in range(min(5, len(all_names))):
            print(f"  {i+1}: name={all_names[i]}, transformation={all_transforms[i]}")


def scan_noise_levels(preprocessed_dir):
    noise_dirs = []
    for subdir in Path(preprocessed_dir).iterdir():
        if subdir.is_dir() and subdir.name.startswith("noise_"):
            match = re.match(r"noise_(\d+\.\d+)", subdir.name)
            if match:
                noise_dirs.append((float(match.group(1)), subdir))
    noise_dirs.sort(key=lambda x: x[0])
    return noise_dirs


def scan_mask_ratios(preprocessed_dir):
    mask_dirs = []
    for subdir in Path(preprocessed_dir).iterdir():
        if subdir.is_dir() and subdir.name.startswith("mask_"):
            match = re.match(r"mask_(\d+\.\d+)", subdir.name)
            if match:
                mask_dirs.append((float(match.group(1)), subdir))
    mask_dirs.sort(key=lambda x: x[0])
    return mask_dirs


# =====================================
# Main Controller
# =====================================
@torch.no_grad()
def extract_features(args):
    rank, world_size, gpu_id = initialize_distributed()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    vae = LatentVAE(args.weight_path).to(device).eval()

    if args.mode == "clean":
        feats, names, trans = extract_features_clean(args, vae, device, rank, world_size)
        feats, names, trans = gather_results(feats, names, trans, rank, world_size)
        if rank == 0:
            sorted_idx = sorted(range(len(names)), key=lambda i: names[i])
            feats, names, trans = feats[sorted_idx], [names[i] for i in sorted_idx], [trans[i] for i in sorted_idx]
            save_features(feats, names, trans, Path(args.output_dir) / f"{args.output_prefix}_clean.pt", args, "clean")

    elif args.mode == "equivariance":
        feats, names, trans = extract_features_equivariance(args, vae, device, rank, world_size)
        feats, names, trans = gather_results(feats, names, trans, rank, world_size)
        if rank == 0 and names:
            sorted_idx = sorted(range(len(names)), key=lambda i: names[i])
            feats, names, trans = feats[sorted_idx], [names[i] for i in sorted_idx], [trans[i] for i in sorted_idx]
            save_features(feats, names, trans, Path(args.output_dir) / f"{args.output_prefix}_equivariance.pt", args, "equivariance")

    elif args.mode == "noise":
        noise_dirs = scan_noise_levels(args.input_dir)
        if rank == 0: print(f"Found {len(noise_dirs)} noise level dirs")
        for level, subdir in noise_dirs:
            if rank == 0: print(f"Processing noise level {level:.3f} ...")
            feats, names, trans = extract_features_processed(args, vae, device, rank, world_size, subdir, "noise", level)
            feats, names, trans = gather_results(feats, names, trans, rank, world_size)
            if rank == 0 and names:
                sorted_idx = sorted(range(len(names)), key=lambda i: names[i])
                feats, names, trans = feats[sorted_idx], [names[i] for i in sorted_idx], [trans[i] for i in sorted_idx]
                save_features(feats, names, trans, Path(args.output_dir) / f"{args.output_prefix}_noise_{level:.3f}.pt", args, "noise", {"noise_level": level})

    elif args.mode == "mask":
        mask_dirs = scan_mask_ratios(args.input_dir)
        if rank == 0: print(f"Found {len(mask_dirs)} mask ratio dirs")
        for ratio, subdir in mask_dirs:
            if rank == 0: print(f"Processing mask ratio {ratio:.2f} ...")
            feats, names, trans = extract_features_processed(args, vae, device, rank, world_size, subdir, "mask", ratio)
            feats, names, trans = gather_results(feats, names, trans, rank, world_size)
            if rank == 0 and names:
                sorted_idx = sorted(range(len(names)), key=lambda i: names[i])
                feats, names, trans = feats[sorted_idx], [names[i] for i in sorted_idx], [trans[i] for i in sorted_idx]
                save_features(feats, names, trans, Path(args.output_dir) / f"{args.output_prefix}_mask_{ratio:.2f}.pt", args, "mask", {"mask_ratio": ratio})


# =====================================
# Entry Point
# =====================================
def main():
    parser = argparse.ArgumentParser(description="Extract SD-VAE latents with multiple modes")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--weight-path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["clean", "equivariance", "noise", "mask"], default="clean")
    parser.add_argument("--apply-scaling", action="store_true")
    args = parser.parse_args()

    try:
        extract_features(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
