# -----------------------------------------------------------------------------------------------
# Extract DINOv2 cls tokens and SD-VAE latents for diffusion analysis.
# Input: folder with files named as synset_number.png (e.g., n01440764_001.png)
# Output: safetensors files with SD-VAE latents, DINOv2 cls tokens, labels, and image names.
# Also saves SD-VAE normalization stats (mean=0, std=1/0.18215).
# -----------------------------------------------------------------------------------------------


import sys
sys.path.append('.')
import os
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
import PIL.Image
import numpy as np

from tqdm import tqdm
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
warnings.filterwarnings("ignore")


# ================================================================
# Distributed utils
# ================================================================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        rank, world_size, gpu_id = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    return rank, world_size, gpu_id


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ================================================================
# ImageNet mapping
# ================================================================
def load_imagenet_mapping():
    """Load ImageNet synset to label mapping from online JSON only."""
    try:
        import json, urllib.request
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        with urllib.request.urlopen(url) as response:
            class_index = json.loads(response.read().decode())
        synset_to_label, label_to_synset = {}, {}
        for label_str, (synset, _) in class_index.items():
            label = int(label_str)
            synset_to_label[synset] = label
            label_to_synset[label] = synset
        print(f"✅ Loaded ImageNet mapping from online JSON: {len(synset_to_label)} classes")
        return synset_to_label, label_to_synset
    except Exception as e:
        print(f"❌ Failed to load mapping: {e}")
        return None, None


# ================================================================
# SD-VAE loader
# ================================================================
class SDVAE(torch.nn.Module):
    def __init__(self, weight_path: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        from diffusers.models import AutoencoderKL
        print(f"🔄 Loading SD-VAE from: {weight_path}")
        self.model = AutoencoderKL.from_pretrained(weight_path)
        self.scaling_factor = self.model.config.scaling_factor  # usually 0.18215

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Return unscaled latents (do not divide by scaling_factor)
        latents = self.model.encode(x).latent_dist.sample()
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents).sample


# ================================================================
# DINOv2 loader (rank-safe)
# ================================================================
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


def preprocess_batch_dinov2(images: torch.Tensor, device, target_size=224):
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    images = F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - mean) / std


# ================================================================
# Dataset
# ================================================================
class ImageFolderDataset(Dataset):
    def __init__(self, input_dir, image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.samples = []
        self.synset_to_label, _ = load_imagenet_mapping()
        for fn in os.listdir(input_dir):
            if any(fn.lower().endswith(ext) for ext in image_extensions):
                name = os.path.splitext(fn)[0]
                parts = name.split('_')
                synset = '_'.join(parts[:-1]) if len(parts) >= 2 else name
                label = self.synset_to_label.get(synset, 0) if self.synset_to_label else 0
                self.samples.append({
                    "path": os.path.join(input_dir, fn),
                    "label": label,
                    "name": name
                })
        print(f"Found {len(self.samples)} images in {input_dir}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = PIL.Image.open(s["path"]).convert("RGB")
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        return {"image": torch.from_numpy(arr), "label": s["label"], "name": s["name"]}


# ================================================================
# Extraction
# ================================================================
@torch.no_grad()
def run_feature_extraction_folder(
    vae_model,
    dinov2_model,
    input_dir,
    output_dir,
    rank,
    device,
    batch_size_per_gpu=32,
    images_per_safetensor=1000,
):
    os.makedirs(output_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    dataset = ImageFolderDataset(input_dir)
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=sampler, num_workers=4)

    all_latents, all_dino, all_labels, all_names = [], [], [], []
    for batch in tqdm(dataloader, desc=f"[Rank {rank}]"):
        imgs = batch["image"].to(device)

        # SD-VAE preprocessing
        imgs_vae = imgs.float() / 255.0
        imgs_vae = F.interpolate(imgs_vae, size=(256, 256), mode="bilinear", align_corners=False)
        imgs_vae = imgs_vae * 2.0 - 1.0
        latents = vae_model.encode(imgs_vae)
        all_latents.append(latents.cpu())

        # DINOv2 CLS token
        x_dino = preprocess_batch_dinov2(imgs, device)
        feats = dinov2_model.forward_features(x_dino)["x_norm_clstoken"]
        all_dino.append(feats.cpu())

        all_labels.append(torch.tensor(batch["label"]))
        all_names.extend(batch["name"])

    latents = torch.cat(all_latents, dim=0)
    dinov2 = torch.cat(all_dino, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"Rank {rank}: got {latents.shape[0]} latents, {dinov2.shape[0]} dino tokens")

    # Save chunked .safetensors
    for i in range(0, latents.shape[0], images_per_safetensor):
        j = min(i + images_per_safetensor, latents.shape[0])
        enc = [n.encode('utf-8') for n in all_names[i:j]]
        maxlen = max(len(n) for n in enc)
        names_tensor = torch.zeros(len(enc), maxlen, dtype=torch.uint8)
        for k, n in enumerate(enc):
            nb = torch.frombuffer(n, dtype=torch.uint8)
            names_tensor[k, :len(nb)] = nb
        chunk = {
            "latents": latents[i:j].contiguous(),
            "dinov2_cls_tokens": dinov2[i:j].contiguous(),
            "labels": labels[i:j].contiguous(),
            "names": names_tensor.contiguous(),
        }
        save_file(chunk, os.path.join(output_dir, f"latents_rank{rank:02d}_batch{i//images_per_safetensor:03d}.safetensors"))
    print(f"💾 Saved features to {output_dir}")


# ================================================================
# Save fixed SD-VAE stats
# ================================================================
def save_sdvae_stats(output_dir):
    mean = torch.zeros(1, 4, 1, 1)
    std = torch.full((1, 4, 1, 1), 1.0 / 0.18215)
    stats = {"mean": mean, "std": std}
    path = os.path.join(output_dir, "latents_stats.pt")
    torch.save(stats, path)
    print(f"✅ Saved SD-VAE stats (mean=0, std=1/0.18215) to {path}")


# ================================================================
# Main
# ================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size-per-gpu", type=int, default=32)
    parser.add_argument("--images-per-safetensor", type=int, default=1000)
    args = parser.parse_args()

    rank, world_size, gpu_id = setup_distributed()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    vae = SDVAE().to(device).eval()
    dinov2 = load_dinov2_model(device)

    run_feature_extraction_folder(
        vae, dinov2, args.input_dir, args.output_dir, rank, device,
        batch_size_per_gpu=args.batch_size_per_gpu,
        images_per_safetensor=args.images_per_safetensor,
    )

    if rank == 0:
        save_sdvae_stats(args.output_dir)

    cleanup_distributed()


if __name__ == "__main__":
    main()
