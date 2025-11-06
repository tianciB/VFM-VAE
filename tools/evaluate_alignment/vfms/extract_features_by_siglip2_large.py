# ------------------------------------------------------------------------------
# Extract SigLIP2-Large features for images
# ------------------------------------------------------------------------------

import os
import re
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

# Import SigLIP2Encoder
import sys
sys.path.append('.')
from networks.utils.vfms.siglip2_utils import SigLIP2Encoder


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
        img = torch.from_numpy(np.array(img, np.float32) / 255.0).permute(2, 0, 1)
        return {'image': img, 'name': path.stem}


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
        img = torch.from_numpy(np.array(img, np.float32) / 255.0).permute(2, 0, 1)
        rot = record.get('rotation', 0)
        if rot:
            img = torch.rot90(img, k=rot // 90, dims=[-2, -1])
        return {'image': img, 'name': name, 'transformation': record}


class ProcessedImageDataset(Dataset):
    """Dataset for loading processed images (with noise)."""
    def __init__(self, processed_dir, resolution=256):
        self.dir = Path(processed_dir)
        self.image_paths = sorted(list(self.dir.glob('*.png')), key=lambda p: p.stem)
        self.resolution = resolution

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img = torch.from_numpy(np.array(img, np.float32) / 255.0).permute(2, 0, 1)
        return {'image': img, 'name': path.stem}


# ------------------------------
# Distributed Utils
# ------------------------------

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        if not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                    rank=rank, world_size=world)
    else:
        try:
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            rank, world = dist.get_rank(), dist.get_world_size()
            gpu = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            rank, world, gpu = 0, 1, 0
    if world > 1 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    return rank, world, gpu


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------
# Model + Utils
# ------------------------------

def load_siglip2_model(args, device):
    encoder = SigLIP2Encoder(
        model_name=args.model_path,
        conditional=False,
        label_type='cls2text',
        scale_factor=1.0,
        patch_from_layers=[-1],
        amp_enabled=True,
        amp_dtype=torch.bfloat16
    ).to(device)
    encoder.eval()
    return encoder


def scan_noise_levels(base_dir):
    p = Path(base_dir); dirs = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith('noise_'):
            m = re.match(r'noise_(\d+\.\d+)', d.name)
            if m: dirs.append((float(m.group(1)), d))
    dirs.sort(key=lambda x: x[0])
    return dirs


def gather_results(f, n, t, rank, world):
    if world > 1:
        gf, gn, gt = [None]*world, [None]*world, [None]*world
        dist.all_gather_object(gf, f)
        dist.all_gather_object(gn, n)
        dist.all_gather_object(gt, t)
        if rank == 0:
            F, N, T = [], [], []
            for fi, ni, ti in zip(gf, gn, gt):
                if isinstance(fi, torch.Tensor) and fi.numel() > 0: F.append(fi)
                N.extend(ni or []); T.extend(ti or [])
            f = torch.cat(F, 0) if F else torch.empty(0, 1024)
            n, t = N, T
        else:
            f, n, t = torch.empty(0, 1024), [], []
    return f, n, t


# ------------------------------
# Feature Extraction
# ------------------------------

def extract_features_clean(args, encoder, device, rank, world):
    dataset = CleanImageDataset(args.input_dir, resolution=args.resolution)
    sampler = DistributedSampler(dataset, world, rank, False) if world > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, disable=(rank != 0)):
            imgs = b['image'].to(device)
            patch_feats, _ = encoder.encode_image(imgs, eq_scale_factor=1.0, is_eq_prior=False)
            f = patch_feats[0].mean(1)
            feats.append(f.cpu()); names.extend(b['name'])
            trans.extend([{'mode': 'clean'}]*len(b['name']))
    feats = torch.cat(feats, 0) if feats else torch.empty(0, 1024)
    return feats, names, trans


def extract_features_equivariance(args, encoder, device, rank, world):
    jpath = Path(args.input_dir) / "equivariance_transforms.json"
    if not jpath.exists():
        if rank == 0: print(f"Missing: {jpath}")
        return torch.empty(0, 1024), [], []
    dataset = EquivarianceDataset(args.input_dir, jpath, resolution=args.resolution)
    sampler = DistributedSampler(dataset, world, rank, False) if world > 1 else None
    loader = DataLoader(dataset, 1, sampler=sampler, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, disable=(rank != 0)):
            img = b['image'].to(device)
            scale = float(b['transformation'].get('scale', 1.0))
            pf, _ = encoder.encode_image(img, eq_scale_factor=scale, is_eq_prior=True)
            f = pf[0].mean(1)
            feats.append(f.cpu()); names.extend(b['name'])
            trans.extend([dict(b['transformation'], mode='equivariance')])
    feats = torch.cat(feats, 0) if feats else torch.empty(0, 1024)
    return feats, names, trans


def extract_features_noise(args, encoder, device, rank, world, d, lvl):
    dataset = ProcessedImageDataset(d, resolution=args.resolution)
    sampler = DistributedSampler(dataset, world, rank, False) if world > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, disable=(rank != 0)):
            imgs = b['image'].to(device)
            patch_feats, _ = encoder.encode_image(imgs, eq_scale_factor=1.0, is_eq_prior=False)
            f = patch_feats[0].mean(1)
            feats.append(f.cpu()); names.extend(b['name'])
            trans.extend([{'mode': 'noise', 'noise_level': lvl}]*len(b['name']))
    feats = torch.cat(feats, 0) if feats else torch.empty(0, 1024)
    return feats, names, trans


# ------------------------------
# Main Logic
# ------------------------------

@torch.no_grad()
def extract_features(args):
    rank, world, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Mode: {args.mode}, Device: {device}, World size: {world}")
    encoder = load_siglip2_model(args, device)
    if world > 1: dist.barrier()

    if args.mode == 'clean':
        f, n, t = extract_features_clean(args, encoder, device, rank, world)
    elif args.mode == 'equivariance':
        f, n, t = extract_features_equivariance(args, encoder, device, rank, world)
    elif args.mode == 'noise':
        for lvl, d in scan_noise_levels(args.input_dir):
            if rank == 0: print(f"Processing noise level {lvl}")
            f, n, t = extract_features_noise(args, encoder, device, rank, world, d, lvl)
            f, n, t = gather_results(f, n, t, rank, world)
            if rank == 0:
                idx = sorted(range(len(n)), key=lambda i: n[i])
                f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
                out = Path(args.output_dir) / f"{args.output_prefix}_noise_{lvl:.3f}.pt"
                torch.save({'features': f, 'names': n, 'transformations': t}, out)
                print(f"Saved {len(n)} noise={lvl:.3f} → {out}")
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
    p = argparse.ArgumentParser(description="Extract SigLIP2 features (clean / noise / equivariance)")
    p.add_argument('--input-dir', type=str, required=True,
                   help='Directory containing images (expects subfolders clean/, noise_*, etc.)')
    p.add_argument('--output-dir', type=str, required=True,
                   help='Directory to save extracted features')
    p.add_argument('--output-prefix', type=str, required=True,
                   help='Prefix for output files')
    p.add_argument('--mode', type=str, choices=['clean', 'noise', 'equivariance'], default='clean')
    p.add_argument('--model-path', type=str, default='your_path/huggingface/siglip2-large-patch16-512/')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--batch-size-per-gpu', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    extract_features(args)


if __name__ == '__main__':
    main()
