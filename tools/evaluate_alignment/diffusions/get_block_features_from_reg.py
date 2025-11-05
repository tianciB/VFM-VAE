# ==============================================================
# Extract intermediate DiT features from REG.
# ==============================================================


import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from glob import glob
from safetensors import safe_open

# Disable torch.compile to avoid symbolic_shapes warnings
os.environ['TORCH_COMPILE_DISABLE'] = '1'
warnings.filterwarnings("ignore", category=FutureWarning)

# Import local modules
sys.path.append('.')
from models.sit import SiT_models


# ==============================================================
# Dataset Definition
# ==============================================================
class LatentDataset(Dataset):
    """Dataset for loading both VAE latents and DINOv2 cls tokens from safetensors with image names."""
    
    def __init__(self, input_dir):
        self.input_dir = input_dir
        
        # Get all safetensor files
        self.files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
        print(f"Found {len(self.files)} safetensor files")
        
        # Build mapping from index to file and position
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        # Load statistics for normalization
        self._latent_mean, self._latent_std = self.get_latent_stats()
        print("✅ Using latent norm: True")
            
        # Extract image names for sorting
        self.image_names = self.extract_image_names()
        
    def get_img_to_safefile_map(self):
        """Create mapping from image index to safetensor file and position."""
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                try:
                    latents = f.get_slice('latents')
                    num_imgs = latents.get_shape()[0]
                except:
                    try:
                        dinov2_tokens = f.get_slice('dinov2_cls_tokens')
                        num_imgs = dinov2_tokens.get_shape()[0]
                    except:
                        labels = f.get_slice('labels')
                        num_imgs = labels.get_shape()[0]
                
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len + i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file
    
    def get_latent_stats(self):
        """Load latent statistics from cache file."""
        latent_stats_cache_file = os.path.join(self.input_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            vae_stats_cache_file = os.path.join(self.input_dir, "vae_stats.pt")
            if os.path.exists(vae_stats_cache_file):
                latent_stats_cache_file = vae_stats_cache_file
            else:
                raise FileNotFoundError(f"VAE latent stats not found in {self.input_dir}")
        
        latent_stats = torch.load(latent_stats_cache_file, map_location='cpu')
        return latent_stats['mean'], latent_stats['std']
    
    def extract_image_names(self):
        """Extract image names from all safetensor files."""
        names = []
        for idx in range(len(self.img_to_file_map)):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                try:
                    name_tensor = f.get_slice('names')
                    name = name_tensor[img_idx]
                    
                    if isinstance(name, torch.Tensor) and name.dtype == torch.uint8:
                        bytes_data = name.cpu().numpy()
                        bytes_data = bytes_data[bytes_data != 0]
                        name = bytes(bytes_data).decode('utf-8')
                    
                    name = os.path.splitext(name)[0]
                except:
                    name = f"image_{idx:06d}"
            
            names.append(name)

        return names
    
    def __len__(self):
        return len(self.img_to_file_map)
    
    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        
        result = {
            'name': self.image_names[idx],
            'idx': idx
        }
        
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            try:
                latents = f.get_slice('latents')
                vae_feature = latents[img_idx:img_idx+1].squeeze(0)
                vae_feature = (vae_feature - self._latent_mean.squeeze(dim=0)) / self._latent_std.squeeze(dim=0)
                result['latent'] = vae_feature
            except Exception as e:
                print(f"Warning: Could not load VAE latents from {safe_file}: {e}")
                result['latent'] = None
            
            try:
                dinov2_tokens = f.get_slice('dinov2_cls_tokens')
                dinov2_feature = dinov2_tokens[img_idx:img_idx+1].squeeze(0)
                result['dinov2_cls'] = dinov2_feature
            except Exception as e:
                print(f"Warning: Could not load DINOv2 cls tokens from {safe_file}: {e}")
                result['dinov2_cls'] = None
            
            try:
                labels = f.get_slice('labels')
                label = labels[img_idx:img_idx+1].squeeze(0)
                result['label'] = label
            except:
                result['label'] = torch.tensor(0)

        return result


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_from_checkpoint(args, device):
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=args.latent_size,
        in_channels=args.vae_latent_dim,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    )
    
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    return model


def forward_with_features(model, x, t, y, cls_token, print_shapes=False):
    features = {}
    x = model.x_embedder(x)
    
    if cls_token is not None:
        cls_token = model.cls_projectors2(cls_token)
        cls_token = model.wg_norm(cls_token)
        cls_token = cls_token.unsqueeze(1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + model.pos_embed
    else:
        x = x + model.pos_embed[:, 1:]
    
    N, T, D = x.shape
    
    if cls_token is not None:
        embedder_feature = x[:, 1:].mean(dim=1)
    else:
        embedder_feature = x.mean(dim=1)
    features['embedder'] = embedder_feature
    
    t_embed = model.t_embedder(t)
    y_embed = model.y_embedder(y, model.training)
    c = t_embed + y_embed

    for i, block in enumerate(model.blocks):
        x = block(x, c)
        if cls_token is not None:
            block_feature = x[:, 1:].mean(dim=1)
        else:
            block_feature = x.mean(dim=1)
        features[f'block_{i}'] = block_feature
        if (i + 1) == model.encoder_depth:
            zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in model.projectors]
            for j, z in enumerate(zs):
                projector_feature = z.mean(dim=1)
                features[f'projector_{j}'] = projector_feature
                
    final_output, cls_output = model.final_layer(x, c, cls=cls_token)
    final_feature = final_output.mean(dim=1)
    features['final_layer'] = final_feature
    
    return features


def add_noise_to_latent(latent, t, path_type="linear"):
    noise = torch.randn_like(latent)
    t_expanded = t.view(-1, 1, 1, 1).expand_as(latent)
    
    if path_type == "linear":
        alpha_t = 1 - t_expanded
        sigma_t = t_expanded
        noisy_latent = alpha_t * latent + sigma_t * noise
    elif path_type == "cosine":
        alpha_t = torch.cos(t_expanded * np.pi / 2)
        sigma_t = torch.sin(t_expanded * np.pi / 2)
        noisy_latent = alpha_t * latent + sigma_t * noise
    else:
        raise NotImplementedError(f"Path type {path_type} not implemented")
    
    return noisy_latent


def add_noise_to_cls_token(cls_token, t, path_type="linear"):
    noise = torch.randn_like(cls_token)
    t_expanded = t.view(-1, 1).expand_as(cls_token)
    
    if path_type == "linear":
        alpha_t = 1 - t_expanded
        sigma_t = t_expanded
        noisy_cls_token = alpha_t * cls_token + sigma_t * noise
    elif path_type == "cosine":
        alpha_t = torch.cos(t_expanded * np.pi / 2)
        sigma_t = torch.sin(t_expanded * np.pi / 2)
        noisy_cls_token = alpha_t * cls_token + sigma_t * noise
    else:
        raise NotImplementedError(f"Path type {path_type} not implemented")
    
    return noisy_cls_token


def save_features_separately(features_dict, names, indices, output_prefix, timestep, rank, world_size, device):
    if world_size > 1:
        gathered_features = {}
        gathered_names = [None for _ in range(world_size)]
        gathered_indices = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_names, names)
        dist.all_gather_object(gathered_indices, indices)
        for feature_name, local_features in features_dict.items():
            if local_features.device != device:
                local_features = local_features.to(device)
            gathered_feature_list = [torch.zeros_like(local_features) for _ in range(world_size)]
            dist.all_gather(gathered_feature_list, local_features)
            gathered_features[feature_name] = torch.cat([f.cpu() for f in gathered_feature_list], dim=0)
        if rank == 0:
            all_names, all_indices = [], []
            for names_list, indices_list in zip(gathered_names, gathered_indices):
                all_names.extend(names_list)
                all_indices.extend(indices_list)
        else:
            return
    else:
        gathered_features = features_dict
        all_names = names
        all_indices = indices
    
    if rank == 0:
        combined_data = list(zip(all_names, all_indices, range(len(all_names))))
        combined_data.sort(key=lambda x: x[0])
        sorted_names, sorted_indices, sorted_positions = zip(*combined_data)
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        for feature_name, features in gathered_features.items():
            sorted_features = torch.stack([features[pos] for pos in sorted_positions])
            save_dict = {
                'features': sorted_features,
                'names': list(sorted_names),
                'num_images': sorted_features.shape[0],
                'sorted': True,
                'sorted_by': 'name',
                'original_indices': list(sorted_indices),
                'timestep': timestep,
                'feature_name': feature_name,
                'feature_dim': sorted_features.shape[1],
            }
            output_path = f"{output_prefix}_{feature_name}_t{timestep:.3f}.pt"
            torch.save(save_dict, output_path)
            print(f"Saved {feature_name} features to {output_path}")
            print(f"Feature shape: {sorted_features.shape}")
        print(f"\n=== Verification: First 5 image names (sorted alphabetically) ===")
        for i, name in enumerate(sorted_names[:5]):
            print(f"{i+1}. {name}")
        print(f"Total images processed: {len(sorted_names)}")


@torch.no_grad()
def extract_features(args, rank, world_size, local_rank):
    # ✅ Set random seed for reproducibility
    import random
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{local_rank}")
    model = load_model_from_checkpoint(args, device)
    dataset = LatentDataset(
        input_dir=args.input_dir,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    all_features, all_names, all_indices = {}, [], []
    if rank == 0:
        print(f"Processing {len(dataset)} samples with timestep t={args.timestep}")
    
    for batch in tqdm(dataloader, desc=f"Rank {rank}", disable=(rank != 0)):
        batch_size = len(batch['name'])
        names, indices = batch['name'], batch['idx']
        
        if args.use_label:
            labels = batch['label'].to(device)
        else:
            labels = torch.full(
                (batch_size,),
                args.num_classes,
                dtype=torch.long,
                device=device
            )
        
        if 'latent' in batch and batch['latent'][0] is not None:
            latents = torch.stack([latent for latent in batch['latent'] if latent is not None]).to(device)
        else:
            latents = torch.randn(batch_size, args.vae_latent_dim, args.latent_size, args.latent_size, device=device)
        
        if 'dinov2_cls' in batch and batch['dinov2_cls'][0] is not None:
            cls_tokens = torch.stack([token for token in batch['dinov2_cls'] if token is not None]).to(device)
        else:
            raise ValueError("DINOv2 cls tokens are required but not found in the dataset.")
        
        t = torch.full((batch_size,), args.timestep, device=device, dtype=torch.float32)
        noisy_latents = add_noise_to_latent(latents, t, args.path_type)
        noisy_cls_tokens = add_noise_to_cls_token(cls_tokens, t, args.path_type)
        print_shapes = len(all_features) == 0 and rank == 0
        batch_features = forward_with_features(model, noisy_latents, t, labels, noisy_cls_tokens, print_shapes)
        for feature_name, features in batch_features.items():
            if feature_name not in all_features:
                all_features[feature_name] = []
            all_features[feature_name].append(features.cpu())
        all_names.extend(names)
        all_indices.extend(indices.tolist())
    
    for feature_name in all_features:
        all_features[feature_name] = torch.cat(all_features[feature_name], dim=0)
    save_features_separately(
        all_features, all_names, all_indices, args.output_prefix,
        args.timestep, rank, world_size, device
    )


def main():
    parser = argparse.ArgumentParser(description="Extract REG model features from VAE latents and DINOv2 cls tokens for CKNNA computation")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--vae-latent-dim", type=int, default=4)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-label", action="store_true", default=False, help="Use dataset labels instead of null label")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    rank, world_size, local_rank = setup_distributed()
    try:
        extract_features(args, rank, world_size, local_rank)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
