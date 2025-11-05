# ------------------------------------------------------------------------------
# Extract VFM-VAE Features for images
# ------------------------------------------------------------------------------

import os
import re
import sys
import yaml
import json
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append('.')
import dnnlib


# ------------------------------
# Dataset Definitions
# ------------------------------

class CleanImageDataset(Dataset):
    """Dataset for loading clean images without any transformations."""
    def __init__(self, input_dir, resolution=256):
        self.image_dir = Path(input_dir) / "clean"
        self.image_paths = sorted(list(self.image_dir.glob('*.png')), key=lambda p: p.stem)
        self.resolution = resolution

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, np.float32) / 255.0
        return {'image': torch.from_numpy(arr).permute(2, 0, 1), 'name': path.stem}


class EquivarianceDataset(Dataset):
    """Dataset for loading images with equivariance transformations based on JSON records."""
    def __init__(self, input_dir, transform_json_path, resolution=256):
        self.image_dir = Path(input_dir) / "clean"
        self.resolution = resolution
        with open(transform_json_path, 'r') as f:
            self.records = json.load(f)
        self.names = sorted(self.records.keys())

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        record = self.records[name]
        path = self.image_dir / f"{name}.png"
        img = Image.open(path).convert('RGB')
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, np.float32)
        rot = record.get('rotation', 0)
        if rot:
            arr = np.rot90(arr, k=rot // 90, axes=(0, 1)).copy()
        arr /= 255.0
        return {'image': torch.from_numpy(arr).permute(2, 0, 1), 'name': name, 'transformation': record}


class ProcessedImageDataset(Dataset):
    """Dataset for loading processed images (with noise or masking)."""
    def __init__(self, processed_dir, resolution=256):
        self.dir = Path(processed_dir)
        self.image_paths = sorted(list(self.dir.glob('*.png')), key=lambda p: p.stem)
        self.resolution = resolution

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, np.float32) / 255.0
        return {'image': torch.from_numpy(arr).permute(2, 0, 1), 'name': path.stem}


# ------------------------------
# Distributed Setup
# ------------------------------

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world)
    else:
        try:
            dist.init_process_group("nccl")
            rank, world = dist.get_rank(), dist.get_world_size()
            gpu = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception as e:
            print(f"Failed to init distributed: {e}")
            rank, world, gpu = 0, 1, 0
    if world > 1 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    return rank, world, gpu


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------
# Model Loading (pth version)
# ------------------------------

def load_vae_model(args, device):
    """Load VAE model from YAML + PTH checkpoint."""
    with open(args.use_config, "r") as f:
        full_cfg = yaml.safe_load(f)

    vae_kwargs = full_cfg.get("G_kwargs", {})
    vae_kwargs["label_dim"] = 1000                  # default for ImageNet
    vae_kwargs["img_resolution"] = args.resolution  # set resolution
    vae_kwargs["conditional"] = False               # reconstruction is unconditional
    vae_kwargs["label_type"] = "cls2text"           # dummy, not used
    vae_kwargs["num_fp16_res"] = 0                  # disable fp16 for validation

    model = dnnlib.util.construct_class_by_name(**vae_kwargs).to(device)
    model.requires_grad_(False)

    print(f"Loading checkpoint from: {args.vae_pth}")
    ckpt = torch.load(args.vae_pth, map_location=device)
    if isinstance(ckpt, dict) and "G_ema" in ckpt:
        state_dict = ckpt["G_ema"]
        print("→ Found 'G_ema' in checkpoint.")
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        print("→ Loading full state_dict directly.")
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️ Unexpected keys: {len(unexpected)}")
        
    print("Model weights loaded successfully.")
    model.eval()
    return model


# ------------------------------
# Scanners
# ------------------------------

def scan_noise_levels(base_dir):
    p = Path(base_dir)
    dirs = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith('noise_'):
            m = re.match(r'noise_(\d+\.\d+)', d.name)
            if m:
                dirs.append((float(m.group(1)), d))
    dirs.sort(key=lambda x: x[0])
    return dirs


def scan_mask_ratios(base_dir):
    p = Path(base_dir)
    dirs = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith('mask_'):
            m = re.match(r'mask_(\d+\.\d+)', d.name)
            if m:
                dirs.append((float(m.group(1)), d))
    dirs.sort(key=lambda x: x[0])
    return dirs


# ------------------------------
# Feature Extraction
# ------------------------------

def extract_features_clean(args, model, device, rank, world):
    dataset = CleanImageDataset(args.input_dir, resolution=args.resolution)
    sampler = DistributedSampler(dataset, world, rank, False) if world > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=4, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, disable=(rank != 0)):
            imgs = b['image'].to(device)
            v = model.encode(imgs, eq_scale_factor=1.0, is_eq_prior=False)
            feats.append(v.mean([-2, -1]).float().cpu())
            names.extend(b['name'])
            trans.extend([{'mode': 'clean'}] * len(b['name']))
    feats = torch.cat(feats, 0) if feats else torch.empty(0, 512)
    return feats, names, trans


def extract_features_equivariance(args, model, device, rank, world):
    jpath = Path(args.input_dir) / "equivariance_transforms.json"
    if not jpath.exists():
        if rank == 0:
            print(f"Missing: {jpath}")
        return torch.empty(0, 512), [], []
    dataset = EquivarianceDataset(args.input_dir, jpath, resolution=args.resolution)
    sampler = DistributedSampler(dataset, world, rank, False) if world > 1 else None
    loader = DataLoader(dataset, 1, sampler=sampler, shuffle=False,
                        num_workers=4, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, disable=(rank != 0)):
            img = b['image'].to(device)
            tf = b['transformation']
            scale = float(tf.get('scale', 1.0))
            v = model.encode(img, eq_scale_factor=scale, is_eq_prior=True) # rotation is applied in dataset
            f = v[0]
            if f.ndim > 1:
                f = f.mean([-2, -1])
            feats.append(f.float().cpu())
            names.extend(b['name'])
            trans.extend([dict(tf, mode='equivariance')])
    feats = torch.stack(feats, 0) if feats else torch.empty(0, 512)
    return feats, names, trans


def extract_features_processed(args, model, device, rank, world, dir_path, mode, lvl):
    dataset = ProcessedImageDataset(dir_path, resolution=args.resolution)
    sampler = DistributedSampler(dataset, world, rank, False) if world > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=4, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, disable=(rank != 0)):
            imgs = b['image'].to(device)
            v = model.encode(imgs, eq_scale_factor=1.0, is_eq_prior=False)
            feats.append(v.mean([-2, -1]).float().cpu())
            names.extend(b['name'])
            if mode == 'noise':
                trans.extend([{'mode': mode, 'noise_level': lvl}] * len(b['name']))
            else:
                trans.extend([{'mode': mode, 'mask_ratio': lvl}] * len(b['name']))
    feats = torch.cat(feats, 0) if feats else torch.empty(0, 512)
    return feats, names, trans


# ------------------------------
# Gathering + Main Logic
# ------------------------------

def gather_results(f, n, t, rank, world):
    if world > 1:
        gf, gn, gt = [None]*world, [None]*world, [None]*world
        dist.all_gather_object(gf, f)
        dist.all_gather_object(gn, n)
        dist.all_gather_object(gt, t)
        if rank == 0:
            F, N, T = [], [], []
            for fi, ni, ti in zip(gf, gn, gt):
                if isinstance(fi, torch.Tensor) and fi.numel() > 0:
                    F.append(fi)
                N.extend(ni or [])
                T.extend(ti or [])
            f = torch.cat(F, 0) if F else torch.empty(0, 512)
            n, t = N, T
        else:
            f, n, t = torch.empty(0, 512), [], []
    return f, n, t


@torch.no_grad()
def extract_features(args):
    rank, world, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Mode: {args.mode}, Device: {device}, World size: {world}")
    model = load_vae_model(args, device)
    if world > 1:
        dist.barrier()

    if args.mode == 'clean':
        f, n, t = extract_features_clean(args, model, device, rank, world)
    elif args.mode == 'equivariance':
        f, n, t = extract_features_equivariance(args, model, device, rank, world)
    elif args.mode == 'noise':
        for lvl, d in scan_noise_levels(args.input_dir):
            if rank == 0:
                print(f"Processing noise level {lvl}")
            f, n, t = extract_features_processed(args, model, device, rank, world, d, 'noise', lvl)
            f, n, t = gather_results(f, n, t, rank, world)
            if rank == 0:
                idx = sorted(range(len(n)), key=lambda i: n[i])
                f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
                out = Path(args.output_dir) / f"{args.output_prefix}_noise_{lvl:.3f}.pt"
                torch.save({'features': f, 'names': n, 'transformations': t}, out)
                print(f"Saved {len(n)} noise={lvl:.3f} → {out}")
        cleanup_distributed(); return
    elif args.mode == 'mask':
        for lvl, d in scan_mask_ratios(args.input_dir):
            if rank == 0:
                print(f"Processing mask ratio {lvl}")
            f, n, t = extract_features_processed(args, model, device, rank, world, d, 'mask', lvl)
            f, n, t = gather_results(f, n, t, rank, world)
            if rank == 0:
                idx = sorted(range(len(n)), key=lambda i: n[i])
                f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
                out = Path(args.output_dir) / f"{args.output_prefix}_mask_{lvl:.2f}.pt"
                torch.save({'features': f, 'names': n, 'transformations': t}, out)
                print(f"Saved {len(n)} mask={lvl:.2f} → {out}")
        cleanup_distributed(); return
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    f, n, t = gather_results(f, n, t, rank, world)
    if rank == 0:
        idx = sorted(range(len(n)), key=lambda i: n[i])
        f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
        out = Path(args.output_dir) / f"{args.output_prefix}_{args.mode}.pt"
        torch.save({'features': f, 'names': n, 'transformations': t}, out)
        print(f"Saved {len(n)} {args.mode} features → {out}")
    cleanup_distributed()


# ------------------------------
# Entry
# ------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract VAE features (pth checkpoint)")
    p.add_argument('--input-dir', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--output-prefix', type=str, required=True)
    p.add_argument('--mode', type=str, choices=['clean', 'noise', 'mask', 'equivariance'], default='clean')
    p.add_argument('--vae-pth', type=str, required=True, help='Path to VAE checkpoint (.pth)')
    p.add_argument('--use-config', type=str, required=True, help='YAML config file')
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--batch-size-per-gpu', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    extract_features(args)


if __name__ == '__main__':
    main()
