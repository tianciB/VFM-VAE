# ==============================================================
# Extract intermediate DiT features from LightningDiT.
# ==============================================================


import os
import sys
import yaml
import torch
import argparse
import warnings
import torch.distributed as dist

from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ==============================================================
# Environment Setup
# ==============================================================
os.environ["TORCH_COMPILE_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

# Local modules
sys.path.append(".")
from models.lightningdit import LightningDiT_models
from transport import create_transport


# ==============================================================
# Config Loader
# ==============================================================
def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================
# Dataset Definition
# ==============================================================
class LatentDataset(Dataset):
    """Dataset for loading latent tensors and metadata from .safetensors files."""

    def __init__(self, input_dir, latent_norm=True, latent_multiplier=1.0):
        self.input_dir = input_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        # Collect safetensor files
        self.files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
        print(f"Found {len(self.files)} safetensor files.")

        # Build index mapping
        self.img_to_file_map = self._build_index_map()

        # Load normalization stats if required
        if latent_norm:
            self._latent_mean, self._latent_std = self._load_latent_stats()
            print("✅ Using latent normalization.")
        else:
            print("❌ Skipping latent normalization.")

        # Extract image names
        self.image_names = self._extract_image_names()

    def _build_index_map(self):
        """Map each image index to its safetensor file and position."""
        mapping = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                try:
                    latents = f.get_slice("latents")
                    num_imgs = latents.get_shape()[0]
                except Exception:
                    labels = f.get_slice("labels")
                    num_imgs = labels.get_shape()[0]

                base_idx = len(mapping)
                for i in range(num_imgs):
                    mapping[base_idx + i] = {"safe_file": safe_file, "idx_in_file": i}
        return mapping

    def _load_latent_stats(self):
        """Load latent normalization statistics."""
        stats_path = os.path.join(self.input_dir, "latents_stats.pt")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Missing latents_stats.pt in {self.input_dir}")
        stats = torch.load(stats_path, map_location="cpu")
        return stats["mean"], stats["std"]

    def _extract_image_names(self):
        """Read image names embedded in safetensors metadata."""
        names = []
        for idx in range(len(self.img_to_file_map)):
            info = self.img_to_file_map[idx]
            with safe_open(info["safe_file"], framework="pt", device="cpu") as f:
                try:
                    name_tensor = f.get_slice("names")[info["idx_in_file"]]
                    if isinstance(name_tensor, torch.Tensor) and name_tensor.dtype == torch.uint8:
                        name_bytes = name_tensor.cpu().numpy()
                        name_bytes = name_bytes[name_bytes != 0]
                        name = bytes(name_bytes).decode("utf-8")
                    else:
                        name = str(name_tensor)
                    name = os.path.splitext(name)[0]
                except Exception:
                    name = f"image_{idx:06d}"
            names.append(name)
        return names

    def __len__(self):
        return len(self.img_to_file_map)

    def __getitem__(self, idx):
        info = self.img_to_file_map[idx]
        with safe_open(info["safe_file"], framework="pt", device="cpu") as f:
            latents = f.get_slice("latents")[info["idx_in_file"] : info["idx_in_file"] + 1].squeeze(0)
            try:
                labels = f.get_slice("labels")[info["idx_in_file"] : info["idx_in_file"] + 1].squeeze(0)
            except Exception:
                labels = torch.tensor(0)

        if self.latent_norm:
            latents = (latents - self._latent_mean.squeeze(0)) / self._latent_std.squeeze(0)
        latents *= self.latent_multiplier

        return {"latent": latents, "label": labels, "name": self.image_names[idx], "idx": idx}


# ==============================================================
# Distributed Setup Utilities
# ==============================================================
def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("⚠️ Running in non-distributed mode.")
        return 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.barrier()
    return rank, world_size, local_rank


def cleanup_distributed():
    """Terminate distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ==============================================================
# Model Loading
# ==============================================================
def load_model_from_config(config, checkpoint_path, device):
    """Build and load LightningDiT model from checkpoint."""
    latent_size = config["data"]["image_size"] // config["vae"].get("downsample_ratio", 16)

    model = LightningDiT_models[config["model"]["model_type"]](
        input_size=latent_size,
        num_classes=config["data"]["num_classes"],
        use_qknorm=config["model"]["use_qknorm"],
        use_swiglu=config["model"].get("use_swiglu", False),
        use_rope=config["model"].get("use_rope", False),
        use_rmsnorm=config["model"].get("use_rmsnorm", False),
        wo_shift=config["model"].get("wo_shift", False),
        in_channels=config["model"].get("in_chans", 4),
        learn_sigma=config["model"].get("learn_sigma", False),
    )

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded successfully.")

    return model.to(device).eval()


# ==============================================================
# Feature Extraction Functions
# ==============================================================
def forward_with_features(model, x, t, y):
    """Extract intermediate block features from LightningDiT."""
    features = {}
    x = model.x_embedder(x) + model.pos_embed
    t_emb = model.t_embedder(t)
    y_emb = model.y_embedder(y, train=False)
    c = t_emb + y_emb

    features["embedder"] = x.mean(dim=1)
    for i, block in enumerate(model.blocks):
        x = block(x, c, model.feat_rope)
        features[f"block_{i}"] = x.mean(dim=1)
    x = model.final_layer(x, c)
    features["final_layer"] = x.mean(dim=1)
    return features


def add_noise_to_latent(latent, t, transport):
    """Apply controlled noise to latents via interpolation with random noise."""
    noise = torch.randn_like(latent)
    t_exp = t.view(-1, 1, 1, 1).expand_as(latent)
    return (1 - t_exp) * latent + t_exp * noise


# ==============================================================
# Multi-GPU Aggregation and Saving
# ==============================================================
def save_features(features_dict, names, indices, output_prefix, timestep, rank, world_size, device):
    """Aggregate features across ranks and save only on rank 0."""
    if world_size > 1:
        gathered_features = {}
        gathered_names = [None for _ in range(world_size)]
        gathered_indices = [None for _ in range(world_size)]

        dist.all_gather_object(gathered_names, names)
        dist.all_gather_object(gathered_indices, indices)

        for fname, local_feat in features_dict.items():
            if local_feat.device != device:
                local_feat = local_feat.to(device)
            temp = [torch.zeros_like(local_feat) for _ in range(world_size)]
            dist.all_gather(temp, local_feat)
            gathered_features[fname] = torch.cat([f.cpu() for f in temp], dim=0)
    else:
        gathered_features = features_dict
        gathered_names, gathered_indices = [names], [indices]

    if rank == 0:
        all_names = [n for sub in gathered_names for n in sub]
        all_indices = [i for sub in gathered_indices for i in sub]
        combined = list(zip(all_names, all_indices, range(len(all_names))))
        combined.sort(key=lambda x: x[0])
        sorted_names, _, pos = zip(*combined)

        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        for fname, feats in gathered_features.items():
            sorted_feats = torch.stack([feats[p] for p in pos])
            save_dict = {
                "features": sorted_feats,
                "names": list(sorted_names),
                "feature_name": fname,
                "timestep": timestep,
                "num_images": len(sorted_names),
            }
            out_path = f"{output_prefix}_{fname}_t{timestep:.3f}.pt"
            torch.save(save_dict, out_path)
            print(f"✅ Saved {fname} → {out_path}, shape={sorted_feats.shape}")


# ==============================================================
# Main Extraction Procedure
# ==============================================================
@torch.no_grad()
def extract_features(config, args, rank, world_size, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    input_dir = args.input_dir
    latent_norm = config["data"].get("latent_norm", True)
    latent_multiplier = config["data"].get("latent_multiplier", 1.0)
    checkpoint = config.get("ckpt_path")
    if not checkpoint:
        raise ValueError("No checkpoint path provided.")

    model = load_model_from_config(config, checkpoint, device)
    transport = create_transport(**config["transport"])

    dataset = LatentDataset(input_dir, latent_norm, latent_multiplier)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    dataloader = DataLoader(dataset, args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    all_features, all_names, all_indices = {}, [], []
    if rank == 0:
        print(f"🚀 Processing {len(dataset)} samples with timestep={args.timestep}")

    for batch in tqdm(dataloader, desc=f"Rank {rank}", disable=(rank != 0)):
        latents = batch["latent"].to(device)
        labels = batch["label"].to(device) if args.use_label else torch.full(
            (latents.size(0),), config["data"]["num_classes"], dtype=torch.long, device=device)
        t = torch.full((latents.size(0),), args.timestep, device=device, dtype=torch.float32)
        noisy_latents = add_noise_to_latent(latents, t, transport)
        feats = forward_with_features(model, noisy_latents, 1 - t, labels)

        for k, v in feats.items():
            all_features.setdefault(k, []).append(v.cpu())
        all_names.extend(batch["name"])
        all_indices.extend(batch["idx"].tolist())

    for k in all_features:
        all_features[k] = torch.cat(all_features[k], dim=0)

    save_features(all_features, all_names, all_indices, args.output_prefix, args.timestep, rank, world_size, device)


# ==============================================================
# Entry Point
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Extract intermediate LightningDiT features.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--use-config", type=str, required=True)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-label", action="store_true", default=False)
    args = parser.parse_args()

    config = load_config(args.use_config)
    print(f"Loaded config: {args.use_config}")

    rank, world_size, local_rank = setup_distributed()
    try:
        extract_features(config, args, rank, world_size, local_rank)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
