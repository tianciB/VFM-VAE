# ------------------------------------------------------------------------------
# Extract VA-VAE features for images
# ------------------------------------------------------------------------------

import os
import re
import sys
import json
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Local import
sys.path.append('.')
from tokenizer.vavae import VA_VAE


# ------------------------------
# Dataset Definitions
# ------------------------------

class CleanImageDataset(Dataset):
    """Dataset for clean images."""
    def __init__(self, input_dir, tokenizer, resolution=256):
        self.input_dir = Path(input_dir) / "clean"
        self.image_paths = sorted(list(self.input_dir.glob('*.png')), key=lambda p: p.stem)
        self.tokenizer = tokenizer
        self.resolution = resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        transform = self.tokenizer.img_transform(p_hflip=0.0)
        img = transform(img)
        return {'image': img, 'name': img_path.stem}


class EquivarianceDataset(Dataset):
    """Dataset with equivariance transformations (rotation, scale)."""
    def __init__(self, input_dir, tokenizer, transform_json_path, resolution=256):
        self.input_dir = Path(input_dir) / "clean"
        self.tokenizer = tokenizer
        self.resolution = resolution
        with open(transform_json_path, 'r') as f:
            self.records = json.load(f)
        self.names = sorted(self.records.keys())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        record = self.records[name]
        img_path = self.input_dir / f"{name}.png"

        img = Image.open(img_path).convert('RGB')
        
        # scaling
        scale = record.get('scale', 1.0)
        if scale != 1.0:
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # rotation
        rot = record.get('rotation', 0)
        if rot != 0:
            img = torch.rot90(img, k=rot // 90, dims=[-2, -1])

        transform = self.tokenizer.img_transform(p_hflip=0.0)
        img = transform(img)

        return {'image': img, 'name': name, 'transformation': record}


class NoiseImageDataset(Dataset):
    """Dataset for noise-processed images."""
    def __init__(self, noise_dir, tokenizer, resolution=256):
        self.noise_dir = Path(noise_dir)
        self.image_paths = sorted(list(self.noise_dir.glob('*.png')), key=lambda p: p.stem)
        self.tokenizer = tokenizer
        self.resolution = resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        transform = self.tokenizer.img_transform(p_hflip=0.0)
        img = transform(img)
        return {'image': img, 'name': img_path.stem}


# ------------------------------
# Utility Functions
# ------------------------------

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        if not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                    rank=rank, world_size=world_size)
    else:
        try:
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            gpu_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            rank, world_size, gpu_id = 0, 1, 0
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    return rank, world_size, gpu_id


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_va_vae_model(args):
    return VA_VAE(args.use_config)


def global_pool(v: torch.Tensor) -> torch.Tensor:
    """Global average pooling if tensor has spatial dimensions."""
    if v.ndim == 4:
        return v.mean(dim=[-2, -1])
    elif v.ndim == 3:
        return v.mean(dim=-1)
    return v


def scan_noise_levels(preprocessed_dir):
    path = Path(preprocessed_dir)
    noise_dirs = []
    for subdir in path.iterdir():
        if subdir.is_dir() and subdir.name.startswith('noise_'):
            match = re.match(r'noise_(\d+\.\d+)', subdir.name)
            if match:
                noise_dirs.append((float(match.group(1)), subdir))
    noise_dirs.sort(key=lambda x: x[0])
    return noise_dirs


def gather_results(features, names, transformations, rank, world_size):
    if world_size > 1:
        gathered_f, gathered_n, gathered_t = [None]*world_size, [None]*world_size, [None]*world_size
        dist.all_gather_object(gathered_f, features)
        dist.all_gather_object(gathered_n, names)
        dist.all_gather_object(gathered_t, transformations)
        if rank == 0:
            f_list, n_list, t_list = [], [], []
            for f, n, t in zip(gathered_f, gathered_n, gathered_t):
                if f is not None and isinstance(f, torch.Tensor) and f.numel() > 0:
                    f_list.append(f)
                n_list.extend(n or [])
                t_list.extend(t or [])
            features = torch.cat(f_list, 0) if f_list else torch.empty(0, 512)
            names, transformations = n_list, t_list
        else:
            features, names, transformations = torch.empty(0, 512), [], []
    return features, names, transformations


# ------------------------------
# Feature Extraction
# ------------------------------

def extract_features_clean(args, tokenizer, device, rank, world_size):
    dataset = CleanImageDataset(args.input_dir, tokenizer)
    sampler = DistributedSampler(dataset, world_size, rank, False) if world_size > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)

    feats, names, trans = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0)):
            imgs = batch['image'].to(device)
            v = tokenizer.encode_images(imgs).cpu()
            v = global_pool(v)
            feats.append(v.float())
            names.extend(batch['name'])
            trans.extend([{'mode': 'clean'}] * len(batch['name']))

    all_features = torch.cat(feats, 0) if feats else torch.empty(0, 512)
    return all_features, names, trans


def extract_features_equivariance(args, tokenizer, device, rank, world_size):
    json_path = Path(args.input_dir) / "equivariance_transforms.json"
    if not json_path.exists():
        return torch.empty(0, 512), [], []

    dataset = EquivarianceDataset(args.input_dir, tokenizer, json_path)
    sampler = DistributedSampler(dataset, world_size, rank, False) if world_size > 1 else None
    loader = DataLoader(dataset, 1, sampler=sampler, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    feats, names, trans = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0)):
            img = batch['image'].to(device)
            v = tokenizer.encode_images(img).cpu()[0]
            v = global_pool(v)
            feats.append(v.float())
            names.extend(batch['name'])
            trans.extend([dict(batch['transformation'], mode='equivariance')])

    all_features = torch.stack(feats, 0) if feats else torch.empty(0, 512)
    return all_features, names, trans


def extract_features_noise(args, tokenizer, device, rank, world_size, noise_dir, noise_level):
    dataset = NoiseImageDataset(noise_dir, tokenizer)
    sampler = DistributedSampler(dataset, world_size, rank, False) if world_size > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)

    feats, names, trans = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0)):
            imgs = batch['image'].to(device)
            v = tokenizer.encode_images(imgs).cpu()
            v = global_pool(v)
            feats.append(v.float())
            names.extend(batch['name'])
            trans.extend([{'mode': 'noise', 'noise_level': noise_level}] * len(batch['name']))

    all_features = torch.cat(feats, 0) if feats else torch.empty(0, 512)
    return all_features, names, trans


# ------------------------------
# Main Extraction Logic
# ------------------------------

@torch.no_grad()
def extract_features(args):
    rank, world_size, gpu_id = setup_distributed()
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Using device: {device}, mode: {args.mode}")

    tokenizer = load_va_vae_model(args)
    if world_size > 1:
        dist.barrier()

    if args.mode == 'clean':
        f, n, t = extract_features_clean(args, tokenizer, device, rank, world_size)
    elif args.mode == 'equivariance':
        f, n, t = extract_features_equivariance(args, tokenizer, device, rank, world_size)
    elif args.mode == 'noise':
        for lvl, ndir in scan_noise_levels(args.input_dir):
            if rank == 0:
                print(f"Processing noise level: {lvl}")
            f, n, t = extract_features_noise(args, tokenizer, device, rank, world_size, ndir, lvl)
            f, n, t = gather_results(f, n, t, rank, world_size)
            if rank == 0:
                idx = sorted(range(len(n)), key=lambda i: n[i])
                f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
                save_path = Path(args.output_dir) / f"{args.output_prefix}_noise_{lvl:.3f}.pt"
                torch.save({'features': f, 'names': n, 'transformations': t}, save_path)
                print(f"Saved {len(n)} noise={lvl:.3f} features to {save_path}")
        cleanup_distributed()
        return
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    f, n, t = gather_results(f, n, t, rank, world_size)
    if rank == 0:
        idx = sorted(range(len(n)), key=lambda i: n[i])
        f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
        save_path = Path(args.output_dir) / f"{args.output_prefix}_{args.mode}.pt"
        torch.save({'features': f, 'names': n, 'transformations': t}, save_path)
        print(f"Saved {len(n)} {args.mode} features to {save_path}")

    cleanup_distributed()


# ------------------------------
# Entry Point
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract VA_VAE features (clean/equivariance/noise)")
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--output-prefix', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['clean', 'equivariance', 'noise'], default='clean')
    parser.add_argument('--use-config', type=str, required=True)
    parser.add_argument('--batch-size-per-gpu', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    extract_features(args)


if __name__ == '__main__':
    main()
