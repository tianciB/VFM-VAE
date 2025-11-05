# ------------------------------------------------------------------------------
# Multi-GPU Latent Prefetching for LightningDIT using WebDataset
# ------------------------------------------------------------------------------


import os
import yaml
import dnnlib
import warnings
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
import PIL.Image
import functools

from functools import partial
from glob import glob
from tqdm import tqdm
from safetensors.torch import save_file
from safetensors import safe_open

os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Dataset for Latents
# ------------------------------------------------------------------------------
class ImgLatentDataset(torch.utils.data.Dataset):
    """Dataset wrapper for reading safetensors latents and computing mean/std."""

    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        print(f"[ImgLatentDataset] Found {len(self.img_to_file_map)} images across {len(self.files)} files in {data_dir}")

        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()

    def get_img_to_safefile_map(self):
        """Map each image index to its corresponding safetensor file and index inside file."""
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                num_imgs = labels.get_shape()[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len + i] = {'safe_file': safe_file, 'idx_in_file': i}
        return img_to_file

    def get_latent_stats(self):
        """Load cached mean/std of latents or compute if not available."""
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']

    def compute_latent_stats(self):
        """Compute mean/std over randomly sampled 10k latents."""
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices, desc="Computing latent stats"):
            img_info = self.img_to_file_map[idx]
            with safe_open(img_info['safe_file'], framework="pt", device="cpu") as f:
                features = f.get_slice('latents')
                latents.append(features[img_info['idx_in_file']:img_info['idx_in_file']+1])
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        print("Latent mean/std computed:", latent_stats)
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        """Return (latent, label) with optional normalization."""
        img_info = self.img_to_file_map[idx]
        with safe_open(img_info['safe_file'], framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_info['idx_in_file']:img_info['idx_in_file']+1]
            label = labels[img_info['idx_in_file']:img_info['idx_in_file']+1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier
        return feature.squeeze(0), label.squeeze(0)


# ------------------------------------------------------------------------------
# Preprocessing functions
# ------------------------------------------------------------------------------
def log_and_continue(exn) -> bool:
    """Log webdataset error and continue processing."""
    print(f"Handling webdataset error ({type(exn).__name__}): {exn}")
    return True


def center_crop_imagenet(image_size: int, arr: np.ndarray):
    """Center cropping from ADM preprocessing."""
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


def make_transform(transform, output_width, output_height):
    """Create transform function based on type."""
    if transform == 'center-crop-dhariwal':
        if output_width != output_height:
            raise ValueError('Width and height must match for Dhariwal cropping')
        return functools.partial(center_crop_imagenet, output_width)
    raise ValueError(f'Unknown transform: {transform}')


def preprocess_image_adm(img: PIL.Image.Image, resolution: int) -> np.ndarray:
    """ADM-style preprocessing: center crop and resize."""
    transform_func = make_transform('center-crop-dhariwal', resolution, resolution)
    img_array = np.array(img)
    processed = transform_func(img_array)
    processed = processed.transpose(2, 0, 1)
    return processed.astype(np.uint8)


def create_adm_wds_dataloader(
    train_data: list[str],
    *,
    batch_size_per_gpu: int,
    resolution: int = 256,
    workers: int = 4,
    one_epoch: bool = True,
    base_seed: int = 0,
) -> wds.WebLoader:
    """Create WebDataset dataloader with ADM-style preprocessing."""
    source = wds.SimpleShardList(train_data) if one_epoch else wds.ResampledShards(train_data)
    pipeline = [
        source,
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", label="cls"),
        wds.map_dict(
            image=partial(preprocess_image_adm, resolution=resolution),
            label=lambda x: int(x),
        ),
        wds.to_tuple("image", "label"),
        wds.batched(batch_size_per_gpu),
    ]
    dataset = wds.DataPipeline(*pipeline)
    worker_init_fn = (lambda worker_id: np.random.seed(base_seed + worker_id)) if base_seed is not None else None
    return wds.WebLoader(dataset, batch_size=None, num_workers=workers, worker_init_fn=worker_init_fn)


def preprocess_batch_vae(images: torch.Tensor) -> torch.Tensor:
    """Preprocess batch for VAE encoder."""
    if images.dtype == torch.uint8:
        images = images.float().div(255.0)
    return images


# ------------------------------------------------------------------------------
# Latent Extraction
# ------------------------------------------------------------------------------
@torch.no_grad()
def run_latent_extraction_wds(
    vae_encoder,
    data_path,
    output_dir,
    rank,
    device,
    resolution,
    batch_size_per_gpu,
    max_images_per_gpu=None,
):
    """Run latent extraction with optional image limit per GPU."""
    urls = sorted(glob(os.path.join(data_path, "*.tar"))) if os.path.isdir(data_path) else [data_path]
    print(f"Rank {rank}: Found {len(urls)} tar files")

    loader = create_adm_wds_dataloader(
        train_data=urls,
        batch_size_per_gpu=batch_size_per_gpu,
        resolution=resolution,
        workers=4,
        one_epoch=True,
        base_seed=0,
    )

    latents, latents_flip, labels_all = [], [], []
    saved_files, total_seen = 0, 0

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    for image_batch, label_batch in tqdm(loader, desc=f'[Rank {rank}] Extracting'):
        if max_images_per_gpu is not None and total_seen >= max_images_per_gpu:
            break

        # Trim batch if near the limit
        if max_images_per_gpu is not None:
            remain = max_images_per_gpu - total_seen
            if remain <= 0:
                break
            image_batch = image_batch[:remain]
            label_batch = label_batch[:remain]

        images = preprocess_batch_vae(image_batch).to(device, non_blocking=True)
        labels_idx = torch.tensor(label_batch, dtype=torch.long)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            z1 = vae_encoder.encode(images)
            images_flipped = torch.flip(images, dims=[-1])
            z2 = vae_encoder.encode(images_flipped)

        latents.append(z1.detach().cpu())
        latents_flip.append(z2.detach().cpu())
        labels_all.append(labels_idx.cpu())
        total_seen += images.shape[0]

        # Save every ~10k samples
        if sum(x.shape[0] for x in latents) >= 10000:
            latents_cat = torch.cat(latents, dim=0).contiguous()
            latents_flip_cat = torch.cat(latents_flip, dim=0).contiguous()
            labels_cat = torch.cat(labels_all, dim=0).contiguous()
            save_path = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file({'latents': latents_cat, 'latents_flip': latents_flip_cat, 'labels': labels_cat}, save_path)
            if rank == 0:
                print(f"Saved {save_path}")
            latents, latents_flip, labels_all = [], [], []
            saved_files += 1

    # Save remaining samples
    if latents:
        latents_cat = torch.cat(latents, dim=0).contiguous()
        latents_flip_cat = torch.cat(latents_flip, dim=0).contiguous()
        labels_cat = torch.cat(labels_all, dim=0).contiguous()
        save_path = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file({'latents': latents_cat, 'latents_flip': latents_flip_cat, 'labels': labels_cat}, save_path)
        if rank == 0:
            print(f"Saved {save_path}")

    dist.barrier()
    return total_seen


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',          type=str, required=True, help='Path to WebDataset tar files')
    parser.add_argument('--output-dir',         type=str, required=True, help='Directory to save extracted latents')
    parser.add_argument('--vae-pth',            type=str, required=True, help='Path to the VAE checkpoint (.pth)')
    parser.add_argument('--use-config',         type=str, required=True, help='Path to YAML config')
    parser.add_argument('--resolution',         type=int, default=256, help='Image resolution for preprocessing')
    parser.add_argument('--batch-size-per-gpu', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--max-images-per-gpu', type=int, default=None, help='Optional image limit per GPU')
    args = parser.parse_args()

    # Load YAML config
    with open(args.use_config, "r") as f:
        full_cfg = yaml.safe_load(f)

    vae_kwargs = full_cfg.get("G_kwargs", {})
    vae_kwargs["label_dim"] = 1000                  # default for ImageNet
    vae_kwargs["img_resolution"] = args.resolution  # set resolution
    vae_kwargs["conditional"] = False               # reconstruction is unconditional
    vae_kwargs["label_type"] = "cls2text"           # dummy, not used

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

    # Run extraction
    total_processed = run_latent_extraction_wds(
        vae_encoder=vae_encoder,
        data_path=args.data_path,
        output_dir=args.output_dir,
        rank=rank,
        device=device,
        resolution=args.resolution,
        batch_size_per_gpu=args.batch_size_per_gpu,
        max_images_per_gpu=args.max_images_per_gpu,
    )

    print(f"✅ Rank {rank} processed {total_processed} images.")

    if rank == 0:
        dataset = ImgLatentDataset(args.output_dir, latent_norm=True)
        print("Latent stats saved at", os.path.join(args.output_dir, "latents_stats.pt"))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
