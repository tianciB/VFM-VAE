# -----------------------------------------------------------------------------------------------
# Extract VA-VAE latents for diffusion analysis.
# Input: folder with files named as synset_number.png (e.g., n01440764_001.png)
# Output: safetensors files with VA-VAE latents, labels, and image names.
# -----------------------------------------------------------------------------------------------


import sys
sys.path.append('.')
import os
import warnings
import numpy as np
import torch
import torch.distributed as dist
import PIL.Image
from tqdm import tqdm
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tokenizer.vavae import VA_VAE

os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
warnings.filterwarnings("ignore")


# ================== ImageNet mapping ==================
def load_imagenet_mapping():
    try:
        import json, urllib.request
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        with urllib.request.urlopen(url) as response:
            class_index = json.loads(response.read().decode())
        synset_to_label = {v[0]: int(k) for k, v in class_index.items()}
        print(f"✅ Loaded ImageNet mapping: {len(synset_to_label)} classes")
        return synset_to_label
    except Exception as e:
        print(f"⚠️ Failed to load ImageNet mapping: {e}")
        return None


# ================== Dataset ==================
class ImageFolderDataset(Dataset):
    def __init__(self, input_dir, tokenizer, image_exts=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.samples = []
        self.tokenizer = tokenizer
        mapping = load_imagenet_mapping()
        for fn in os.listdir(input_dir):
            if not fn.lower().endswith(image_exts):
                continue
            name = os.path.splitext(fn)[0]
            parts = name.split('_')
            synset = '_'.join(parts[:-1]) if len(parts) >= 2 else name
            label = mapping.get(synset, 0) if mapping else 0
            self.samples.append({
                "path": os.path.join(input_dir, fn),
                "label": label,
                "name": name,
            })
        print(f"📂 Found {len(self.samples)} images in {input_dir}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = PIL.Image.open(s["path"]).convert("RGB")
        img = self.tokenizer.img_transform(p_hflip=0.0)(img)
        return {"image": img, "label": s["label"], "name": s["name"]}


# ================== Distributed utils ==================
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu_id = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
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


# ================== Model loader ==================
def load_va_vae_model(cfg_path):
    print(f"🔄 Loading VA-VAE from config: {cfg_path}")
    model = VA_VAE(cfg_path)
    print("✅ VA-VAE loaded successfully.")
    return model


# ================== Extraction ==================
@torch.no_grad()
def run_feature_extraction_folder(
    vae_model, input_dir, output_dir, rank, device,
    batch_size_per_gpu=32, images_per_safetensor=1000, max_images_per_gpu=None
):
    os.makedirs(output_dir, exist_ok=True)
    dataset = ImageFolderDataset(input_dir, vae_model)
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    loader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=False)

    all_latents, all_labels, all_names = [], [], []
    total = 0

    for batch in tqdm(loader, desc=f"[Rank {rank}] Processing"):
        if max_images_per_gpu and total >= max_images_per_gpu:
            break
        imgs, labs, names = batch["image"], batch["label"], batch["name"]
        if max_images_per_gpu:
            remain = max_images_per_gpu - total
            imgs, labs, names = imgs[:remain], labs[:remain], names[:remain]
        imgs = imgs.to(device, non_blocking=True)
        latents = vae_model.encode_images(imgs)

        all_latents.append(latents.cpu())
        all_labels.append(labs.cpu())
        all_names.extend(names)
        total += len(labs)

    if total == 0:
        print(f"[Rank {rank}] No images processed.")
        return 0

    latents = torch.cat(all_latents)
    labels = torch.cat(all_labels)
    print(f"Rank {rank}: {total} latents collected.")

    for i in range(0, total, images_per_safetensor):
        j = min(i + images_per_safetensor, total)
        names_enc = [n.encode("utf-8") for n in all_names[i:j]]
        maxlen = max(len(n) for n in names_enc)
        names_tensor = torch.zeros(len(names_enc), maxlen, dtype=torch.uint8)
        for k, n in enumerate(names_enc):
            nb = torch.frombuffer(n, dtype=torch.uint8)
            names_tensor[k, :len(nb)] = nb

        save_dict = {
            "latents": latents[i:j].contiguous(),
            "labels": labels[i:j].contiguous(),
            "names": names_tensor.contiguous(),
        }
        out_path = os.path.join(output_dir, f"latents_rank{rank:02d}_batch{i//images_per_safetensor:03d}.safetensors")
        save_file(save_dict, out_path)
        print(f"💾 Saved {out_path} ({j - i} samples)")

    if dist.is_initialized():
        dist.barrier()
    return total


# ================== Main ==================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-config", required=True)
    parser.add_argument("--batch-size-per-gpu", type=int, default=32)
    parser.add_argument("--images-per-safetensor", type=int, default=1000)
    parser.add_argument("--max-images-per-gpu", type=int, default=None)
    args = parser.parse_args()

    rank, world_size, gpu_id = setup_distributed()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 VA-VAE feature extraction started (rank {rank}/{world_size})")

    vae_model = load_va_vae_model(args.use_config)

    total = run_feature_extraction_folder(
        vae_model, args.input_dir, args.output_dir, rank, device,
        args.batch_size_per_gpu, args.images_per_safetensor, args.max_images_per_gpu,
    )
    print(f"🎯 Rank {rank}: Finished {total} samples")

    cleanup_distributed()


if __name__ == "__main__":
    main()
