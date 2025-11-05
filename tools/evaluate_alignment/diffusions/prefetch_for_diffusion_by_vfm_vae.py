# ---------------------------------------------------------------------------------------------
# Extract DINOv2 cls tokens and VFM-VAE latents for diffusion analysis.
# Input: folder with files named as synset_number.png (e.g., n01440764_001.png)
# Output: safetensors files with DINOv2 cls tokens, VAE latents, labels, and image names.
# ---------------------------------------------------------------------------------------------


import sys
sys.path.append('.')
import os
import yaml
import dnnlib
import warnings
import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
warnings.filterwarnings("ignore")


# -------------------------------
# ImageNet label mapping
# -------------------------------
def load_imagenet_mapping():
    """Load ImageNet synset to label mapping from online JSON only."""
    try:
        import json
        import urllib.request
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        with urllib.request.urlopen(url) as response:
            class_index = json.loads(response.read().decode())
        synset_to_label, label_to_synset = {}, {}
        for label_str, (synset, class_name) in class_index.items():
            label = int(label_str)
            synset_to_label[synset] = label
            label_to_synset[label] = synset
        print(f"✅ Loaded ImageNet mapping from online JSON: {len(synset_to_label)} classes")
        return synset_to_label, label_to_synset
    except Exception as e:
        print(f"❌ Failed to load ImageNet mapping: {e}")
        print("⚠️ Using auto-generated labels instead.")
        return None, None


# -------------------------------
# Preprocessing functions
# -------------------------------
def preprocess_batch_dinov2(images: torch.Tensor, device, target_size=224):
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    images = F.interpolate(images, size=(target_size, target_size),
                           mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def preprocess_batch_vae(images: torch.Tensor):
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    return images


# -------------------------------
# Dataset
# -------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, input_dir, image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.input_dir = input_dir
        self.samples = []
        self.synset_to_label, self.label_to_synset = load_imagenet_mapping()
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
        if self.synset_to_label:
            self._process_with_mapping(image_files)
        else:
            self._process_auto(image_files)

    def _process_with_mapping(self, image_files):
        for f in image_files:
            path = os.path.join(self.input_dir, f)
            name = os.path.splitext(f)[0]
            parts = name.split('_')
            synset = '_'.join(parts[:-1]) if len(parts) >= 2 else name
            if synset in self.synset_to_label:
                label = self.synset_to_label[synset]
                self.samples.append({'path': path, 'label': label, 'class_name': synset, 'name': name})
        self.class_to_label = {s['class_name']: s['label'] for s in self.samples}

    def _process_auto(self, image_files):
        classes = sorted(list({os.path.splitext(f)[0].split('_')[0] for f in image_files}))
        self.class_to_label = {cls: i for i, cls in enumerate(classes)}
        for f in image_files:
            path = os.path.join(self.input_dir, f)
            name = os.path.splitext(f)[0]
            cls = name.split('_')[0]
            label = self.class_to_label[cls]
            self.samples.append({'path': path, 'label': label, 'class_name': cls, 'name': name})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = PIL.Image.open(s['path']).convert('RGB')
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        return {'image': torch.from_numpy(arr), 'label': s['label'],
                'name': s['name'], 'class_name': s['class_name']}


# -------------------------------
# Feature extraction
# -------------------------------
@torch.no_grad()
def run_feature_extraction_folder(
    dinov2_model, vae_encoder, input_dir, output_dir, rank, device,
    batch_size_per_gpu=32, images_per_safetensor=1000,
    max_images_per_gpu=None, extract_dinov2=True, extract_vae=True):

    os.makedirs(output_dir, exist_ok=True)
    
    dataset = ImageFolderDataset(input_dir)
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    loader = DataLoader(dataset, batch_size=batch_size_per_gpu,
                        sampler=sampler, num_workers=4,
                        pin_memory=True, drop_last=False)

    all_dinov2, all_vae, all_labels, all_names = [], [], [], []
    total = 0
    for batch in tqdm(loader, desc=f'[Rank {rank}] Processing'):
        if max_images_per_gpu and total >= max_images_per_gpu:
            break
        imgs, labs, names = batch['image'], batch['label'], batch['name']
        if max_images_per_gpu:
            remain = max_images_per_gpu - total
            imgs, labs, names = imgs[:remain], labs[:remain], names[:remain]
        imgs = imgs.to(device, non_blocking=True)

        if extract_dinov2:
            x = preprocess_batch_dinov2(imgs, device)
            feats = dinov2_model.forward_features(x)['x_norm_clstoken']
            all_dinov2.append(feats.cpu())

        if extract_vae:
            x = preprocess_batch_vae(imgs)
            lat = vae_encoder.encode(x)
            all_vae.append(lat.cpu())

        all_labels.append(labs.cpu())
        all_names.extend(names)
        total += len(labs)

    dinov2_tensor = torch.cat(all_dinov2) if extract_dinov2 else None
    vae_tensor = torch.cat(all_vae) if extract_vae else None
    labels_tensor = torch.cat(all_labels)

    if total > 0:
        start = 0; file_idx = 0
        while start < total:
            end = min(start + images_per_safetensor, total)
            save_batch_to_safetensor(
                dinov2_tensor[start:end] if extract_dinov2 else None,
                vae_tensor[start:end] if extract_vae else None,
                labels_tensor[start:end],
                all_names[start:end],
                output_dir, rank, file_idx,
                extract_dinov2, extract_vae
            )
            start = end; file_idx += 1

    if dist.is_initialized():
        dist.barrier()
    return total


def save_batch_to_safetensor(dino, vae, labels, names, outdir, rank, idx, do_dino, do_vae):
    path = os.path.join(outdir, f'latents_rank{rank:02d}_batch{idx:03d}.safetensors')
    enc = [n.encode('utf-8') for n in names]
    maxlen = max(len(n) for n in enc)
    names_tensor = torch.zeros(len(enc), maxlen, dtype=torch.uint8)
    for i, n in enumerate(enc):
        nb = torch.frombuffer(n, dtype=torch.uint8)
        names_tensor[i, :len(nb)] = nb
    d = {'labels': labels.contiguous(), 'names': names_tensor}
    if do_dino: d['dinov2_cls_tokens'] = dino.contiguous()
    if do_vae: d['latents'] = vae.contiguous()
    save_file(d, path)
    print(f"💾 Saved {path} ({len(names)} samples)")


# -------------------------------
# DINOv2 loader
# -------------------------------
def load_dinov2_model(device):
    import torch.hub
    import torch.distributed as dist

    print("🔄 Loading DINOv2 model (dinov2_vitb14) from torch.hub ...")
    try:
        # Only rank 0 downloads the model
        if not dist.is_initialized() or dist.get_rank() == 0:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
            torch.save(model.state_dict(), "/tmp/dinov2_vitb14.pth")
            print("✅ DINOv2 model downloaded and cached.")
        if dist.is_initialized():
            dist.barrier()

        # All ranks load locally from cache
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
        state_dict = torch.load("/tmp/dinov2_vitb14.pth", map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        print("✅ DINOv2 model loaded successfully (from cache).")
        return model

    except Exception as e:
        print(f"❌ Failed to load DINOv2 model: {e}")
        return None


# -------------------------------
# Main
# -------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--vae-pth', type=str, required=True)
    p.add_argument('--use-config', type=str, required=True)
    p.add_argument('--batch-size-per-gpu', type=int, default=32)
    p.add_argument('--images-per-safetensor', type=int, default=1000)
    p.add_argument('--max-images-per-gpu', type=int, default=None)
    p.add_argument('--no-dinov2', action='store_true')
    p.add_argument('--no-vae', action='store_true')
    args = p.parse_args()

    # Load YAML config
    with open(args.use_config, "r") as f:
        full_cfg = yaml.safe_load(f)

    vae_kwargs = full_cfg.get("G_kwargs", {})
    vae_kwargs["label_dim"] = 1000  # ImageNet 1k

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    print(f'Rank {rank} of {world_size} initialized.')

    # Build model
    vae_encoder = dnnlib.util.construct_class_by_name(**vae_kwargs).to(device)
    vae_encoder.requires_grad_(False)

    # Load checkpoint (.pth)
    print(f"Loading checkpoint: {args.vae_pth}")
    checkpoint = torch.load(args.vae_pth, map_location=device)
    if isinstance(checkpoint, dict) and "G_ema" in checkpoint:
        print("Found 'G_ema' in checkpoint, loading its weights...")
        state_dict = checkpoint["G_ema"]
    elif isinstance(checkpoint, dict):
        print("No 'G_ema' found, loading full state dict.")
        state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    vae_encoder.load_state_dict(state_dict, strict=False)
    vae_encoder.eval()

    # DINOv2
    do_dino, do_vae = not args.no_dinov2, not args.no_vae
    dinov2_model = load_dinov2_model(device) if do_dino else None

    # Run extraction
    total_processed = run_feature_extraction_folder(
        dinov2_model=dinov2_model,
        vae_encoder=vae_encoder,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rank=rank,
        device=device,
        batch_size_per_gpu=args.batch_size_per_gpu,
        images_per_safetensor=args.images_per_safetensor,
        max_images_per_gpu=args.max_images_per_gpu,
        extract_dinov2=do_dino,
        extract_vae=do_vae,
    )

    print(f"✅ Rank {rank} processed {total_processed} images.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
