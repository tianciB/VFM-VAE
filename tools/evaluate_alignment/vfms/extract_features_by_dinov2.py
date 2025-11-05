# ------------------------------------------------------------------------------
# Extract DINOv2 features for images (Base / Large / Giant)
# ------------------------------------------------------------------------------

import os
import re
import json
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# ------------------------------------------------------------------------------
# Preprocessing (official DINOv2 transform)
# ------------------------------------------------------------------------------

def get_dinov2_transform(resolution=224):
    """DINOv2 official preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------------------
# Dataset Definition
# ------------------------------------------------------------------------------

class CleanImageDataset(Dataset):
    def __init__(self, input_dir, resolution=224):
        self.image_dir = Path(input_dir) / "clean"
        self.image_paths = sorted(list(self.image_dir.glob('*.png')), key=lambda p: p.stem)
        self.transform = get_dinov2_transform(resolution)
        self.resolution = resolution

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return {'image': img, 'name': path.stem}


class NoiseImageDataset(Dataset):
    def __init__(self, processed_dir, resolution=224):
        self.processed_dir = Path(processed_dir)
        self.image_paths = sorted(list(self.processed_dir.glob('*.png')), key=lambda p: p.stem)
        self.transform = get_dinov2_transform(resolution)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return {'image': img, 'name': path.stem}


class EquivarianceDataset(Dataset):
    def __init__(self, input_dir, transform_json_path, resolution=224):
        self.input_dir = Path(input_dir) / "clean"
        self.resolution = resolution
        with open(transform_json_path, 'r') as f:
            self.records = json.load(f)
        self.names = sorted(self.records.keys())
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        rec = self.records[name]
        path = self.input_dir / f"{name}.png"
        img = Image.open(path).convert('RGB')
        img = np.array(img, np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # rotation
        rot = rec.get('rotation', 0)
        if rot != 0:
            img = torch.rot90(img, k=rot // 90, dims=[-2, -1])

        # scaling
        scale = rec.get('scale', 1.0)
        target = int(self.resolution * scale)
        img = F.interpolate(img.unsqueeze(0), size=target, mode='bilinear', align_corners=False)[0]

        # pad to 14x multiple (safe for ViT-L/G)
        h, w = img.shape[-2:]
        pad_h = (14 - h % 14) % 14
        pad_w = (14 - w % 14) % 14
        if pad_h or pad_w:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

        img = self.normalize(img)
        return {'image': img, 'name': name, 'transformation': rec}


# ------------------------------------------------------------------------------
# Distributed Utils
# ------------------------------------------------------------------------------

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        if not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                    rank=rank, world_size=world_size)
    else:
        try:
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            gpu = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            rank, world_size, gpu = 0, 1, 0
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    return rank, world_size, gpu


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------------------------------------------------------
# Model Loading (timm unified)
# ------------------------------------------------------------------------------

def load_dinov2_model(model_size, device):
    import timm
    model_map = {
        'base':  ('vit_base_patch14_dinov2.lvd142m', 768),
        'large': ('vit_large_patch14_dinov2.lvd142m', 1024),
        'giant': ('vit_giant_patch14_dinov2.lvd142m', 1536),
    }
    if model_size not in model_map:
        raise ValueError(f"Unsupported model size: {model_size}")

    model_name, dim = model_map[model_size]
    print(f"🔹 Loading DINOv2-{model_size} via timm (hf-hub:timm/{model_name})")
    model = timm.create_model(f"hf-hub:timm/{model_name}", pretrained=True, dynamic_img_size=True)
    model = model.to(device).eval()
    print(f"✅ Successfully loaded DINOv2-{model_size} ({dim} dim).")
    return model, dim


# ------------------------------------------------------------------------------
# Feature pooling
# ------------------------------------------------------------------------------

def global_pool(features):
    """Average patch tokens (exclude CLS)."""
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            features = features["x_norm_patchtokens"]
        else:
            features = list(features.values())[0]
    return features[:, 1:].mean(1)


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def scan_noise_levels(preprocessed_dir):
    path = Path(preprocessed_dir)
    noise_dirs = []
    for subdir in path.iterdir():
        if subdir.is_dir() and subdir.name.startswith('noise_'):
            m = re.match(r'noise_(\d+\.\d+)', subdir.name)
            if m:
                noise_dirs.append((float(m.group(1)), subdir))
    noise_dirs.sort(key=lambda x: x[0])
    return noise_dirs


def gather_results(features, names, trans, rank, world_size, dim):
    if world_size > 1:
        gf, gn, gt = [None]*world_size, [None]*world_size, [None]*world_size
        dist.all_gather_object(gf, features)
        dist.all_gather_object(gn, names)
        dist.all_gather_object(gt, trans)
        if rank == 0:
            f_list, n_list, t_list = [], [], []
            for f, n, t in zip(gf, gn, gt):
                if isinstance(f, torch.Tensor) and f.numel() > 0:
                    f_list.append(f)
                n_list.extend(n or [])
                t_list.extend(t or [])
            features = torch.cat(f_list, dim=0) if f_list else torch.empty(0, dim)
            names, trans = n_list, t_list
        else:
            features, names, trans = torch.empty(0, dim), [], []
    return features, names, trans


# ------------------------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------------------------

def extract_features_clean(args, model, device, rank, world_size, dim):
    dataset = CleanImageDataset(args.input_dir)
    sampler = DistributedSampler(dataset, world_size, rank, False) if world_size > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0)):
            imgs = batch['image'].to(device)
            f = global_pool(model.forward_features(imgs))
            feats.append(f.float().cpu())
            names.extend(batch['name'])
            trans.extend([{'mode': 'clean'}] * len(batch['name']))
    if len(feats) == 0:
        return torch.empty(0, dim), [], []
    return torch.cat(feats, 0), names, trans


def extract_features_equivariance(args, model, device, rank, world_size, dim):
    json_path = Path(args.input_dir) / "equivariance_transforms.json"
    if not json_path.exists():
        return torch.empty(0, dim), [], []

    dataset = EquivarianceDataset(args.input_dir, json_path)
    sampler = DistributedSampler(dataset, world_size, rank, False) if world_size > 1 else None
    loader = DataLoader(dataset, 1, sampler=sampler, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    feats, names, trans = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0)):
            img = batch['image'].to(device)
            f = global_pool(model.forward_features(img))
            feats.append(f.float().cpu())
            names.extend(batch['name'])
            trans.extend([dict(batch['transformation'], mode='equivariance')])
    if len(feats) == 0:
        return torch.empty(0, dim), [], []
    return torch.cat(feats, 0), names, trans


def extract_features_noise(args, model, device, rank, world_size, dim, noise_dir, noise_level):
    dataset = NoiseImageDataset(noise_dir)
    sampler = DistributedSampler(dataset, world_size, rank, False) if world_size > 1 else None
    loader = DataLoader(dataset, args.batch_size_per_gpu, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)
    feats, names, trans = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=(rank != 0)):
            imgs = batch['image'].to(device)
            f = global_pool(model.forward_features(imgs))
            feats.append(f.float().cpu())
            names.extend(batch['name'])
            trans.extend([{'mode': 'noise', 'noise_level': noise_level}] * len(batch['name']))
    if len(feats) == 0:
        return torch.empty(0, dim), [], []
    return torch.cat(feats, 0), names, trans


# ------------------------------------------------------------------------------
# Main Logic
# ------------------------------------------------------------------------------

@torch.no_grad()
def extract_features(args):
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Using device: {device}, mode: {args.mode}")
    model, dim = load_dinov2_model(args.model_size, device)

    if args.mode == 'clean':
        f, n, t = extract_features_clean(args, model, device, rank, world_size, dim)
    elif args.mode == 'equivariance':
        f, n, t = extract_features_equivariance(args, model, device, rank, world_size, dim)
    elif args.mode == 'noise':
        for lvl, ndir in scan_noise_levels(args.input_dir):
            if rank == 0:
                print(f"Processing noise level: {lvl}")
            f, n, t = extract_features_noise(args, model, device, rank, world_size, dim, ndir, lvl)
            f, n, t = gather_results(f, n, t, rank, world_size, dim)
            if rank == 0:
                idx = sorted(range(len(n)), key=lambda i: n[i])
                f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
                save_path = Path(args.output_dir) / f"{args.output_prefix}_noise_{lvl:.3f}.pt"
                torch.save({'features': f, 'names': n, 'transformations': t}, save_path)
                print(f"Saved {len(n)} noise={lvl:.3f} features → {save_path}")
            if world_size > 1:
                dist.barrier()
        cleanup_distributed()
        return
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    f, n, t = gather_results(f, n, t, rank, world_size, dim)
    if rank == 0:
        idx = sorted(range(len(n)), key=lambda i: n[i])
        f, n, t = f[idx], [n[i] for i in idx], [t[i] for i in idx]
        save_path = Path(args.output_dir) / f"{args.output_prefix}_{args.mode}.pt"
        torch.save({'features': f, 'names': n, 'transformations': t}, save_path)
        print(f"Saved {len(n)} {args.mode} features → {save_path}")
    cleanup_distributed()


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features (base/large/giant)")
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--output-prefix', type=str, required=True)
    parser.add_argument('--model-size', type=str, choices=['base', 'large', 'giant'], default='large')
    parser.add_argument('--mode', type=str, choices=['clean', 'noise', 'equivariance'], default='clean')
    parser.add_argument('--batch-size-per-gpu', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    extract_features(args)


if __name__ == '__main__':
    main()
