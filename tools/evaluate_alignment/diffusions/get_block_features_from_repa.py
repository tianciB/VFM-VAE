# ==============================================================
# Extract intermediate DiT features from REPA.
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


class LatentDataset(Dataset):
    """Dataset for loading latent features from safetensors with image names."""
    
    def __init__(self, input_dir):
        self.input_dir = input_dir
        
        # Get all safetensor files
        self.files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
        print(f"Found {len(self.files)} safetensor files")
        
        # Build mapping from index to file and position
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        # Load latent statistics for normalization
        self._latent_mean, self._latent_std = self.get_latent_stats()
        print("✅ Using latent norm: True")
            
        # Extract image names for sorting
        self.image_names = self.extract_image_names()
        
    def get_img_to_safefile_map(self):
        """Create mapping from image index to safetensor file and position."""
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                # Try to get the number of items in the file
                try:
                    latents = f.get_slice('latents')
                    num_imgs = latents.get_shape()[0]
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
            raise FileNotFoundError(f"latents_stats.pt not found in {self.input_dir}")
        
        latent_stats = torch.load(latent_stats_cache_file, map_location='cpu')
        return latent_stats['mean'], latent_stats['std']
    
    def extract_image_names(self):
        """Extract image names from all safetensor files."""
        names = []
        for idx in range(len(self.img_to_file_map)):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                # Try to get names directly
                try:
                    name_tensor = f.get_slice('names')
                    name = name_tensor[img_idx]
                    
                    # Convert bytes to string
                    if isinstance(name, torch.Tensor) and name.dtype == torch.uint8:
                        # Remove padding zeros and convert to string
                        bytes_data = name.cpu().numpy()
                        bytes_data = bytes_data[bytes_data != 0]
                        name = bytes(bytes_data).decode('utf-8')
                    
                    # Remove file extension
                    name = os.path.splitext(name)[0]
                    
                except:
                    # Fallback: use index as name
                    name = f"image_{idx:06d}"
            
            names.append(name)

        return names
    
    def __len__(self):
        return len(self.img_to_file_map)
    
    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            # Load latent feature
            latents = f.get_slice('latents')
            feature = latents[img_idx:img_idx+1].squeeze(0)  # Remove batch dimension
            
            # Load label if available
            try:
                labels = f.get_slice('labels')
                label = labels[img_idx:img_idx+1].squeeze(0)
            except:
                label = torch.tensor(0)  # Default label

        feature = (feature - self._latent_mean.squeeze(dim=0)) / self._latent_std.squeeze(dim=0)
        
        return {
            'latent': feature,
            'label': label,
            'name': self.image_names[idx],
            'idx': idx
        }


def setup_distributed():
    """Setup distributed training environment."""
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
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_from_checkpoint(model_name, checkpoint_path, device, resolution, **model_kwargs):
    """Load REPA SiT model from checkpoint."""
    # Create model with REPA configuration
    latent_size = resolution // 8
    block_kwargs = {"fused_attn": model_kwargs.get("fused_attn", False), 
                   "qk_norm": model_kwargs.get("qk_norm", False)}
    
    # First load checkpoint to infer z_dims from projector layers
    z_dims = [0]  # Default value
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if "ema" in checkpoint:
            state_dict = checkpoint["ema"]
            print("✅ Using EMA weights from checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("✅ Using model weights from checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ Using state_dict weights from checkpoint")
        else:
            state_dict = checkpoint
            print("✅ Using direct weights from checkpoint")
        
        # Remove 'module.' prefix if present (from DDP training)
        if isinstance(state_dict, dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Infer z_dims from projector layers in checkpoint
        z_dims = []
        projector_keys = [k for k in state_dict.keys() if k.startswith('projectors.') and k.endswith('.4.weight')]
        for key in sorted(projector_keys):
            z_dim = state_dict[key].shape[0]
            z_dims.append(z_dim)
        
        if not z_dims:
            z_dims = [0]  # Fallback if no projectors found
        
        print(f"📊 Inferred z_dims from checkpoint: {z_dims}")
    
    # Create model with inferred z_dims
    model = SiT_models[model_name](
        input_size=latent_size,
        num_classes=model_kwargs.get("num_classes", 1000),
        use_cfg=model_kwargs.get("use_cfg", False),
        z_dims=z_dims,
        encoder_depth=model_kwargs.get("encoder_depth", 8),
        **block_kwargs
    )
    
    # Load checkpoint weights
    if checkpoint_path and os.path.isfile(checkpoint_path):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            
        print("✅ Model checkpoint loaded successfully!")
    else:
        print("⚠️  No checkpoint provided, using random weights")
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model moved to {device} and set to eval mode")
    
    return model


def forward_with_features(model, x, t, y, print_shapes=False):
    """Forward pass through REPA SiT model and extract features from each stage."""
    features = {}
    
    # Initial embedding
    x = model.x_embedder(x) + model.pos_embed  # (N, T, D)
    N, T, D = x.shape
    
    # Timestep and class embedding
    t_emb = model.t_embedder(t)                # (N, D)
    y_emb = model.y_embedder(y, train=False)   # (N, D) - use train=False for inference
    c = t_emb + y_emb                          # (N, D)
    
    if print_shapes:
        print(f"\n=== Feature Shapes Before Mean Pooling ===")
        print(f"Initial embedding (x_embedder + pos_embed): {x.shape}")
        print(f"Timestep embedding (t_embedder): {t_emb.shape}")
        print(f"Label embedding (y_embedder): {y_emb.shape}")
        print(f"Combined conditioning (c): {c.shape}")
    
    # Save embedder output (apply spatial mean pooling)
    if print_shapes:
        print(f"Embedder output before mean: {x.shape} -> after mean: {x.mean(dim=1).shape}")
    features['embedder'] = x.mean(dim=1)  # (N, D) - spatial mean of patches    
    
    # Pass through each block and collect features
    for i, block in enumerate(model.blocks):
        x = block(x, c)  # REPA SiT blocks
        
        if print_shapes and i < 3:  # Print first 3 blocks
            print(f"Block {i} output before mean: {x.shape} -> after mean: {x.mean(dim=1).shape}")
        elif print_shapes and i == 3:
            print(f"... (blocks 3-{len(model.blocks)-1} have same shape pattern)")
        
        # Apply spatial mean pooling: (N, T, D) -> (N, D)
        block_feature = x.mean(dim=1)  # Average over spatial dimension
        features[f'block_{i}'] = block_feature
        
        # Extract projection features at encoder depth if requested and available
        if (i + 1) == model.encoder_depth and hasattr(model, 'projectors') and len(model.projectors) > 0:
            # Check if projectors have valid dimensions
            if any(proj[4].out_features > 0 for proj in model.projectors if len(proj) > 4):
                zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in model.projectors]
                # Save projection features
                for j, z in enumerate(zs):
                    proj_feature = z.mean(dim=1)  # (N, z_dim)
                    features[f'projection_{j}'] = proj_feature
                if print_shapes:
                    print(f"Projection features: {[z.shape for z in zs]} -> after mean: {[z.mean(dim=1).shape for z in zs]}")
    
    # Final layer
    final_output = model.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
    
    if print_shapes:
        print(f"Final layer output before mean: {final_output.shape} -> after mean: {final_output.mean(dim=1).shape}")
        print(f"=== End of Feature Shapes ===\n")
    
    # Save final layer output (apply spatial mean pooling)
    final_feature = final_output.mean(dim=1)  # (N, patch_size ** 2 * out_channels)
    features['final_layer'] = final_feature
    
    return features


def add_noise_with_interpolant(latent, t, path_type="linear"):
    """Add noise to latent at given timestep t using REPA's interpolant."""
    noise = torch.randn_like(latent)
    
    if path_type == "linear":
        alpha_t = 1 - t
        sigma_t = t
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
    else:
        raise NotImplementedError(f"Path type {path_type} not implemented")
    
    # Expand dimensions to match latent shape
    alpha_t = alpha_t.view(-1, 1, 1, 1).expand_as(latent)
    sigma_t = sigma_t.view(-1, 1, 1, 1).expand_as(latent)
    
    noisy_latent = alpha_t * latent + sigma_t * noise
    return noisy_latent


def save_features_separately(features_dict, names, indices, output_prefix, timestep, rank, world_size, device):
    """Save each feature type separately."""
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
            all_names = []
            all_indices = []
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
                'timestep': timestep,
                'feature_name': feature_name,
                'feature_dim': sorted_features.shape[1],
            }
            
            output_path = f"{output_prefix}_{feature_name}_t{timestep:.3f}.pt"
            torch.save(save_dict, output_path)
            
            print(f"Saved {feature_name} features to {output_path}")
            print(f"Feature shape: {sorted_features.shape}")
        
        print(f"First 5 image names (sorted): {sorted_names[:5]}")
        print(f"Total images processed: {len(sorted_names)}")


@torch.no_grad()
def extract_features(args, rank, world_size, local_rank):
    """Extract features from REPA SiT model."""
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"🚀 Starting REPA SiT Feature Extraction")
        print(f"{'='*60}")
    
    model = load_model_from_checkpoint(
        model_name=args.model,
        checkpoint_path=args.checkpoint_path,
        device=device,
        resolution=args.resolution,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        encoder_depth=args.encoder_depth,
        fused_attn=args.fused_attn,
        qk_norm=args.qk_norm
    )
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        has_projectors = hasattr(model, 'projectors') and len(model.projectors) > 0
        print(f"📊 Model statistics:")
        print(f"   - Architecture: {args.model}")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Number of transformer blocks: {len(model.blocks)}")
        print(f"   - Hidden size: {model.blocks[0].norm1.normalized_shape[0] if model.blocks else 'N/A'}")
        if has_projectors:
            z_dims_str = ', '.join([str(proj[4].out_features) for proj in model.projectors if len(proj) > 4])
            print(f"   - Projector dimensions: [{z_dims_str}]")
        else:
            print(f"   - Projectors: None")
        print(f"✅ Model setup complete!\n")
    
    dataset = LatentDataset(
        input_dir=args.input_dir,
    )
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    all_features = {}
    all_names = []
    all_indices = []
    
    if rank == 0:
        print(f"📋 Processing Configuration:")
        print(f"   - Dataset: {len(dataset)} samples")
        print(f"   - Timestep t: {args.timestep}")
        print(f"   - Batch size: {args.batch_size}")
        print(f"   - Num workers: {args.num_workers}")
        print(f"   - Checkpoint: {args.checkpoint_path}")
        print(f"   - Data path: {args.input_dir}")
        print(f"   - Path type: {args.path_type}")
        print(f"   - Using labels: {'Yes' if args.use_label else 'No (null labels)'}")
        print(f"   - Output prefix: {args.output_prefix}")
        print(f"\n🔄 Starting feature extraction...")
        print(f"{'='*60}")
    
    for batch in tqdm(dataloader, desc=f"Rank {rank}", disable=(rank != 0)):
        latents = batch['latent'].to(device)
        
        # Use label or null label based on configuration
        if args.use_label:
            labels = batch['label'].to(device)
        else:
            labels = torch.full((latents.size(0),), args.num_classes, dtype=torch.long, device=device)
        
        names = batch['name']
        indices = batch['idx']
        
        batch_size = latents.shape[0]
        t = torch.full((batch_size,), args.timestep, device=device, dtype=torch.float32)
        
        # Add noise using REPA's interpolant
        noisy_latents = add_noise_with_interpolant(latents, t, args.path_type)
        
        print_shapes = len(all_features) == 0 and rank == 0
        batch_features = forward_with_features(
            model, noisy_latents, t, labels, 
            print_shapes=print_shapes
        )
        
        for feature_name, features in batch_features.items():
            if feature_name not in all_features:
                all_features[feature_name] = []
            all_features[feature_name].append(features.cpu())
        
        all_names.extend(names)
        all_indices.extend(indices.tolist())
    
    for feature_name in all_features:
        all_features[feature_name] = torch.cat(all_features[feature_name], dim=0)
    
    save_features_separately(
        all_features, 
        all_names, 
        all_indices, 
        args.output_prefix, 
        args.timestep, 
        rank, 
        world_size,
        device
    )
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"🎉 Feature extraction completed successfully!")
        print(f"✅ Extracted features from {len(all_features)} different layers")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Extract REPA SiT block features from latents")
    
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action="store_true", default=True, help="Use fused attention")
    parser.add_argument("--qk-norm", action="store_true", default=False, help="Use QK normalization")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])    
    parser.add_argument("--timestep", type=float, default=0.5, help="Timestep t for adding noise (0.0 to 1.0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--use-label", action="store_true", default=False, help="Whether to use labels from dataset")
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    try:
        extract_features(args, rank, world_size, local_rank)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()