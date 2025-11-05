# ------------------------------------------------------------------------------
# Multi-GPU Latent Sampling for SiT models
# ------------------------------------------------------------------------------


import os
import math
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
from safetensors.torch import save_file

# Local imports
from models.sit import SiT_models
from samplers import euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model


def main(args):
    # ------------------- Setup -------------------
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling requires GPU (use DDP)."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"[Rank {rank}] Seed={seed}, world_size={world_size}")

    # ------------------- Load model -------------------
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.latent_size
    latent_dim = args.vae_latent_dim

    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=latent_dim,   # number of channels in VAE latent
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    if args.ckpt is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=False)['ema']

    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
        )
    model.load_state_dict(state_dict)
    model.eval()

    # ------------------- Load latent stats -------------------
    stats_path = os.path.join(args.latents_stats_dir, "latents_stats.pt")
    assert os.path.exists(stats_path), f"Missing latent stats at {stats_path}"
    stats = torch.load(stats_path, map_location="cpu")
    latents_mean = stats["mean"].view(1, -1, 1, 1).to(device)
    latents_std = stats["std"].view(1, -1, 1, 1).to(device)

    # ------------------- Output folder -------------------
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-f{args.latent_size}d{args.vae_latent_dim}-" \
                    f"cfg-{args.cfg_scale}-cls-cfg-{args.cls_cfg_scale}-{args.mode}-{args.guidance_low}-{args.guidance_high}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"[Rank {rank}] Saving latents at {sample_folder_dir}")
    dist.barrier()

    # ------------------- Sampling setup -------------------
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = total_samples // world_size
    iterations = samples_needed_this_gpu // n
    if rank == 0:
        print(f"Total samples: {total_samples}, Per-GPU: {samples_needed_this_gpu}, Iterations: {iterations}")

    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)

    # ------------------- Buffers -------------------
    latent_buffer = []
    label_buffer = []
    save_chunk = 10000
    shard_id, saved_count = 0, 0

    # ------------------- Main loop -------------------
    for _ in pbar:
        # Random inputs
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        cls_z = torch.randn(n, args.cls, device=device)

        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            cls_latents=cls_z,
            args=args,
        )

        with torch.no_grad():
            if args.mode == "sde":
                latents = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError("Only SDE sampling implemented")

        # De-normalize back to VAE latent space
        latents = latents * latents_std + latents_mean

        latent_buffer.append(latents.cpu())
        label_buffer.append(y.cpu())

        # Save shard if buffer is large enough
        if sum(x.size(0) for x in latent_buffer) >= save_chunk:
            all_latents = torch.cat(latent_buffer, dim=0)
            all_labels = torch.cat(label_buffer, dim=0).to(torch.int)
            filename = f"samples_rank{rank:02d}_shard{shard_id:03d}.safetensors"
            save_file(
                {"latents": all_latents, "labels": all_labels},
                os.path.join(sample_folder_dir, filename)
            )
            if rank == 0:
                print(f"Saved {all_latents.size(0)} latents -> {filename}")
            saved_count += all_latents.size(0)
            shard_id += 1
            latent_buffer.clear()
            label_buffer.clear()

    # Flush remaining buffer
    if latent_buffer:
        all_latents = torch.cat(latent_buffer, dim=0)
        all_labels = torch.cat(label_buffer, dim=0).to(torch.int)
        filename = f"samples_rank{rank:02d}_shard{shard_id:03d}.safetensors"
        save_file(
            {"latents": all_latents, "labels": all_labels},
            os.path.join(sample_folder_dir, filename)
        )
        if rank == 0:
            print(f"Saved {all_latents.size(0)} latents -> {filename}")
        saved_count += all_latents.size(0)

    print(f"[Rank {rank}] Total saved: {saved_count}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # Logging / saving
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--latents-stats-dir", type=str, required=True, help="Path to folder containing latents_stats.pt")

    # Model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--latent-size", type=int, default=16, help="Spatial size of VAE latent (H=W)")
    parser.add_argument("--vae-latent-dim", type=int, default=4, help="Number of VAE latent channels")
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # Number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # Sampling
    parser.add_argument("--mode", type=str, default="sde", choices=["sde"])
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cls-cfg-scale", type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument("--cls", default=768, type=int)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    main(args)
