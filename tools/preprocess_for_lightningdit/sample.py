# ------------------------------------------------------------------------------
# Multi-GPU latent sampling for LightningDiT models with Accelerate
# ------------------------------------------------------------------------------


import os
import math
import argparse
import yaml
import torch
import numpy as np
from time import strftime
from tqdm import tqdm
from safetensors.torch import save_file
from accelerate import Accelerator

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def print_with_prefix(*messages):
    """Print messages with a unified LightningDiT prefix and timestamp."""
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    print(f"{prefix}: {' '.join(map(str, messages))}")


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# ------------------------------------------------------------------------------
# Sampling core
# ------------------------------------------------------------------------------
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None):
    """Perform multi-GPU latent sampling."""
    folder_name = (
        f"{train_config['model']['model_type'].replace('/', '-')}-"
        f"ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-"
        f"{train_config['sample']['sampling_method']}-"
        f"{train_config['sample']['num_sampling_steps']}"
    ).lower()

    timestep_shift = train_config['sample'].get('timestep_shift', 0)
    folder_name += f"-shift{timestep_shift:.2f}"
    cfg_scale = cfg_scale or train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample'].get('cfg_interval_start', 0)
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}-cfg{cfg_scale:.2f}"

    sample_folder_dir = os.path.join(
        train_config['train']['output_dir'],
        train_config['train']['exp_name'],
        folder_name
    )

    if accelerator.process_index == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print_with_prefix('Saving latent .safetensors to', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)

    # Environment setup
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."
    torch.set_grad_enabled(False)

    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval().to(device)

    # Create transport and sampler
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss=train_config['transport'].get('use_cosine_loss', False),
        use_lognorm=train_config['transport'].get('use_lognorm', False),
    )

    sampler = Sampler(transport)
    assert train_config['sample']['mode'] == "ODE"
    sample_fn = sampler.sample_ode(
        sampling_method=train_config['sample']['sampling_method'],
        num_steps=train_config['sample']['num_sampling_steps'],
        atol=train_config['sample']['atol'],
        rtol=train_config['sample']['rtol'],
        reverse=train_config['sample']['reverse'],
        timestep_shift=timestep_shift,
    )

    # ------------------------------------------------------------------------------
    # Sampling setup
    # ------------------------------------------------------------------------------
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = total_samples // accelerator.num_processes
    iterations = samples_needed_this_gpu // n

    latent_size = train_config['data']['image_size'] // train_config['vae'].get('downsample_ratio', 16)

    # Load latent normalization stats
    stats_path = os.path.join(train_config['data']['data_path'], "latents_stats.pt")
    assert os.path.exists(stats_path), f"Missing latent stats at {stats_path}"
    stats = torch.load(stats_path)
    latent_mean = stats['mean'].to(device)
    latent_std = stats['std'].to(device)
    latent_multiplier = train_config['data'].get('latent_multiplier', 0.18215)

    using_cfg = cfg_scale > 1.0
    if using_cfg and accelerator.process_index == 0:
        print_with_prefix('Using classifier-free guidance (CFG).')

    if rank == 0:
        print_with_prefix(f"Total number of samples to generate: {total_samples}")
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)

    # ------------------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------------------
    latent_buffer, label_buffer = [], []
    save_chunk = 10000
    shard_id = 0
    saved_sample_count = 0

    for i in pbar:
        # Sample random latent noise and class labels
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, train_config['data']['num_classes'], (n,), device=device)

        # Apply CFG duplication if enabled
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward

        # Perform ODE sampling
        samples = sample_fn(z, model_fn, **model_kwargs)[-1]

        # Split CFG batch back
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
            y, _ = y.chunk(2, dim=0)

        # Denormalize to original latent scale
        samples = (samples * latent_std) / latent_multiplier + latent_mean

        latent_buffer.append(samples.cpu())
        label_buffer.append(y.cpu())

        # Save every `save_chunk` samples
        if sum(x.size(0) for x in latent_buffer) >= save_chunk or i == iterations - 1:
            all_latents = torch.cat(latent_buffer, dim=0)
            all_labels = torch.cat(label_buffer, dim=0).to(torch.int)
            for start in range(0, all_latents.size(0), save_chunk):
                latent_chunk = all_latents[start:start + save_chunk]
                label_chunk = all_labels[start:start + save_chunk]
                if latent_chunk.size(0) == 0:
                    continue
                filename = f"samples_rank{rank:02d}_shard{shard_id:03d}.safetensors"
                save_file({"latents": latent_chunk, "labels": label_chunk},
                          os.path.join(sample_folder_dir, filename))
                if accelerator.process_index == 0:
                    print_with_prefix(f"Saved {latent_chunk.size(0)} samples to {filename}")
                saved_sample_count += latent_chunk.size(0)
                shard_id += 1
            latent_buffer.clear()
            label_buffer.clear()
        accelerator.wait_for_everyone()

    print_with_prefix(f"[Rank {rank}] Total saved samples: {saved_sample_count}")
    return sample_folder_dir


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed latent sampling for LightningDiT models")
    parser.add_argument('--config', type=str, required=True, help='Path to sampling YAML config file')
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    train_config = load_config(args.config)

    ckpt_path = train_config['ckpt_path']
    if accelerator.process_index == 0:
        print_with_prefix('Using checkpoint:', ckpt_path)

    latent_size = train_config['data']['image_size'] // train_config['vae'].get('downsample_ratio', 16)

    # Build model from config
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model'].get('use_swiglu', False),
        use_rope=train_config['model'].get('use_rope', False),
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
        wo_shift=train_config['model'].get('wo_shift', False),
        in_channels=train_config['model'].get('in_chans', 4),
        learn_sigma=train_config['model'].get('learn_sigma', False),
    )

    # Launch sampling
    do_sample(train_config, accelerator, ckpt_path=ckpt_path, model=model)
