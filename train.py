# ------------------------------------------------------------------------------
# Copyright (c) 2025, Tianci Bi, Xi'an Jiaotong University.
# This version reads all training configurations from a YAML file instead of CLI.
# ------------------------------------------------------------------------------

"""
Train VFM-VAE using YAML-based configuration (.pth checkpoints).
"""

import os
import json
import yaml
import click
import torch
import dnnlib

from glob import glob
from training import training_loop
from torch_utils import custom_ops
from torch_utils import distributed as dist


def find_latest_network_snapshot(run_dir: str):
    """Find the latest .pth snapshot in the output directory."""
    if not os.path.exists(run_dir):
        return None
    snapshots = glob(os.path.join(run_dir, "network-snapshot-*.pth"))
    if not snapshots:
        return None
    latest_snapshot, latest_kimg = None, -1
    for path in snapshots:
        try:
            kimg = int(
                os.path.basename(path)
                .replace("network-snapshot-", "")
                .replace(".pth", "")
            )
            if kimg > latest_kimg:
                latest_kimg, latest_snapshot = kimg, path
        except ValueError:
            continue
    return latest_snapshot


def to_easydict(obj):
    """Recursively convert nested dicts to EasyDicts."""
    if isinstance(obj, dict):
        return dnnlib.EasyDict({k: to_easydict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_easydict(v) for v in obj]
    else:
        return obj


@click.command("train", context_settings={'show_default': True})
@click.option("--config", type=str, required=True, help="Path to YAML config file.")
def main(config):
    # -------------------------------------------------------------
    # 1. Load YAML configuration
    # -------------------------------------------------------------
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    c = to_easydict(cfg)  # recursive EasyDict conversion

    # -------------------------------------------------------------
    # 2. Basic defaults and inheritance
    # -------------------------------------------------------------
    if "one_epoch" not in c:
        c.one_epoch = c.training_set_kwargs.get("one_epoch", False)

    if "resume_kimg" not in c:
        c.resume_kimg = 0

    if "resume_path" not in c:
        c.resume_path = None

    if "G_kwargs" in c:
        if "resolution" not in c.G_kwargs and "resolution" in c.training_set_kwargs:
            c.G_kwargs.img_resolution = c.training_set_kwargs.get("resolution", None)

        if "conditional" not in c.G_kwargs and "conditional" in c.training_set_kwargs:
            c.G_kwargs.conditional = c.training_set_kwargs.get("conditional", False)

        if "label_type" not in c.G_kwargs and "label_type" in c.training_set_kwargs:
            c.G_kwargs.label_type = c.training_set_kwargs.get("label_type", None)

        if "use_kl_loss" not in c.G_kwargs and "kl_loss_weight" in c.loss_kwargs:
            c.G_kwargs.use_kl_loss = c.loss_kwargs.get("kl_loss_weight", 0.0) > 0.0
        
        if "use_vf_loss" not in c.G_kwargs and "vf_loss_weight" in c.loss_kwargs:
            c.G_kwargs.use_vf_loss = c.loss_kwargs.get("vf_loss_weight", 0.0) > 0.0

        if "use_adaptive_vf_loss" not in c.G_kwargs and "use_adaptive_vf_loss" in c.loss_kwargs:
            c.G_kwargs.use_adaptive_vf_loss = c.loss_kwargs.get("use_adaptive_vf_loss", False)

        if "use_equivariance_regularization" not in c.G_kwargs and "use_equivariance_regularization" in c.loss_kwargs:
            c.G_kwargs.use_equivariance_regularization = c.loss_kwargs.get("use_equivariance_regularization", False)

    if "D_kwargs" in c:
        if "vfm_name" not in c.D_kwargs and "G_kwargs" in c:
            c.D_kwargs.vfm_name = c.G_kwargs.get("vfm_name", None)

    if "loss_kwargs" in c:
        if "vfm_name" not in c.loss_kwargs and "G_kwargs" in c:
            c.loss_kwargs.vfm_name = c.G_kwargs.get("vfm_name", None)

        if "compression_mode" not in c.loss_kwargs and "compression_mode" in c.G_kwargs:
            c.loss_kwargs.compression_mode = c.G_kwargs.get("compression_mode", None)

        if "resume_kimg" not in c.loss_kwargs:
            c.loss_kwargs.resume_kimg = c.get("resume_kimg", 0)

    # -------------------------------------------------------------
    # 3. Initialize distributed environment
    # -------------------------------------------------------------
    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init()
    world_size = dist.get_world_size()
    dist.print0(f"[INFO] Distributed init: rank {dist.get_rank()}/{world_size}")

    # -------------------------------------------------------------
    # 4. Auto compute total batch size
    # -------------------------------------------------------------
    if "accumulate_gradients" not in c:
        c.accumulate_gradients = 1

    c.batch_size = c.batch_size
    assert c.batch_size % (world_size * c.accumulate_gradients) == 0, \
        f"batch_size {c.batch_size} must be divisible by (world_size {world_size} * accumulate_gradients {c.accumulate_gradients})"
    c.batch_gpu = c.batch_size // (world_size * c.accumulate_gradients)
    dist.print0(f"[INFO] Auto-computed batch size={c.batch_size}, batch gpu={c.batch_gpu}, gpu count={world_size}, accumulate gradients={c.accumulate_gradients}")

    # -------------------------------------------------------------
    # 5. Random seed & CUDNN setup
    # -------------------------------------------------------------
    seed = c.get("random_seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    dist.print0(f"[INFO] Random seed set to {seed}")

    # -------------------------------------------------------------
    # 6. Auto-resume from latest .pth snapshot
    # -------------------------------------------------------------
    if c.resume_path is None:
        latest_snapshot = find_latest_network_snapshot(c.run_dir)
        if latest_snapshot and os.path.getsize(latest_snapshot) > 1000:
            c.resume_path = latest_snapshot
            try:
                c.resume_kimg = int(
                    os.path.basename(latest_snapshot)
                    .replace("network-snapshot-", "")
                    .replace(".pth", "")
                )
            except ValueError:
                c.resume_kimg = 0
            dist.print0(f"[INFO] Auto-resume from {latest_snapshot}, resume_kimg={c.resume_kimg}")
        else:
            dist.print0("[INFO] No valid snapshot found, starting from scratch.")

    # -------------------------------------------------------------
    # 7. Create directories and logger
    # -------------------------------------------------------------
    c.train_sample_dir = os.path.join(c.run_dir, "train_samples")

    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        os.makedirs(c.train_sample_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        dnnlib.util.Logger(
            file_name=os.path.join(c.run_dir, "log.txt"),
            file_mode="a",
            should_flush=True,
        )
        dist.print0(f"[INFO] Logs and configs saved to {c.run_dir}")
    else:
        custom_ops.verbosity = "none"

    torch.distributed.barrier()  # Sync all ranks

    # -------------------------------------------------------------
    # 8. Print configuration summary
    # -------------------------------------------------------------
    dist.print0()
    dist.print0(f"[INFO] Loaded config from {config}")
    dist.print0(json.dumps(cfg, indent=2))
    dist.print0(f"[INFO] Output directory: {c.run_dir}")
    dist.print0(f"[INFO] Number of GPUs: {world_size}")
    dist.print0()

    # -------------------------------------------------------------
    # 9. Validate required fields
    # -------------------------------------------------------------
    for field in ["training_set_kwargs", "G_kwargs"]:
        if field not in c:
            raise ValueError(f"Missing required field '{field}' in YAML config.")

    # -------------------------------------------------------------
    # 10. Launch training
    # -------------------------------------------------------------
    dist.print0("[INFO] Starting training...")
    training_loop.training_loop(**c)
    dist.print0("[INFO] Training finished.")


if __name__ == "__main__":
    main()
