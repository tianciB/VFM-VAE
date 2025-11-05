# ------------------------------------------------------------------------------
# Training Codes of LightningDiT together with VA-VAE.
# It envolves advanced training methods, sampling methods, 
# architecture design methods, computation methods. We achieve
# state-of-the-art FID 1.35 on ImageNet 256x256.
# 
# Original work by Maple (Jingfeng Yao), HUST-VL.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Modifications Copyright (c) 2025, Tianci Bi, Xi'an Jiaotong University.
# This version includes minor modifications to support gradient accumulation.
# ------------------------------------------------------------------------------


import os
import re
import uuid
import wandb
import yaml
import logging
import argparse
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from models.lightningdit import LightningDiT_models
from transport import create_transport
from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset


def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    device = accelerator.device
    # ==== unified: gradient accumulation from config ====
    grad_accum_steps = int(train_config['train'].get('grad_accum_steps', 1))

    # ==== Output dirs & logging (main process only) ====
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = os.path.join(f"{train_config['train']['output_dir']}", exp_name)
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, is_main=True)
        logger.info(f"Experiment directory created at {experiment_dir}")

        wandb.init(
            project=train_config['train'].get('wandb_project', 'lightningdit'),
            name=exp_name,
            config=train_config,
            id=str(uuid.uuid4()),
            resume="never",
        )
    else:
        # dummy logger for non-main ranks
        exp_name = train_config['train']['exp_name']
        experiment_dir = os.path.join(f"{train_config['train']['output_dir']}", exp_name)
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        logger = create_logger(experiment_dir, is_main=False)

    # Create model
    downsample_ratio = train_config['vae'].get('downsample_ratio', 16)
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by VAE downsample ratio."
    latent_size = train_config['data']['image_size'] // downsample_ratio

    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model'].get('use_swiglu', False),
        use_rope=train_config['model'].get('use_rope', False),
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
        wo_shift=train_config['model'].get('wo_shift', False),
        in_channels=train_config['model'].get('in_chans', 4),
        use_checkpoint=train_config['model'].get('use_checkpoint', False),
    )

    ema = deepcopy(model).to(device)

    # load pretrained
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location='cpu')
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=accelerator.local_process_index)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=accelerator.local_process_index)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")

    requires_grad(ema, False)
    model = model.to(device)

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss=train_config['transport'].get('use_cosine_loss', False),
        use_lognorm=train_config['transport'].get('use_lognorm', False),
    )

    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f"Use lognorm sampling: {train_config['transport'].get('use_lognorm', False)}")
        logger.info(f"Use cosine loss: {train_config['transport'].get('use_cosine_loss', False)}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['optimizer']['lr'],
        weight_decay=0,
        betas=(0.9, train_config['optimizer']['beta2'])
    )

    # ==== Dataset & DataLoader with DistributedSampler ====
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data'].get('latent_norm', False),
        latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215),
    )

    world_size = accelerator.num_processes
    assert train_config['train']['global_batch_size'] % (world_size * grad_accum_steps) == 0, \
        f"global_batch_size must be divisible by world_size*grad_accum_steps; got {train_config['train']['global_batch_size']} vs {world_size}*{grad_accum_steps}"
    batch_size_per_gpu = train_config['train']['global_batch_size'] // (world_size * grad_accum_steps)
    global_batch_size = batch_size_per_gpu * world_size * grad_accum_steps

    train_sampler = DistributedSampler(dataset, drop_last=True, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        sampler=train_sampler,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    valid_loader = None
    if 'valid_path' in train_config['data']:
        valid_dataset = ImgLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data'].get('latent_norm', False),
            latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215),
        )
        valid_sampler = DistributedSampler(valid_dataset, drop_last=True, shuffle=True)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, grad_accum_steps={grad_accum_steps}, effective global batch size {global_batch_size}")

    # ---- Prepare with accelerate ----
    model, opt, loader, valid_loader = accelerator.prepare(model, opt, loader, valid_loader)

    # ==== EMA/init ====
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    # ==== Resume ====
    train_config['train']['resume'] = train_config['train'].get('resume', False)
    optimizer_steps = 0

    def _strip_or_add_module_prefix(sd: dict, want_module_prefix: bool):
        """Make checkpoint keys compatible with (un)wrapped model."""
        if not sd:
            return sd
        keys = list(sd.keys())
        has_module = keys[0].startswith("module.")
        if want_module_prefix and not has_module:
            return {f"module.{k}": v for k, v in sd.items()}
        if (not want_module_prefix) and has_module:
            return {k[len("module."):]: v for k, v in sd.items()}
        return sd

    if train_config['train']['resume']:
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            # checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            def _extract_step(fname):
                num = re.findall(r"(\d+)", os.path.basename(fname))
                return int(num[0]) if num else -1

            checkpoint_files.sort(key=lambda x: _extract_step(x))      
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')

            # --- Model ---
            unwrapped = accelerator.unwrap_model(model)
            model_sd = checkpoint.get('model', {})
            model_sd = _strip_or_add_module_prefix(model_sd, want_module_prefix=False)  # target key: no module prefix
            missing, unexpected = unwrapped.load_state_dict(model_sd, strict=False)

            # --- EMA ---
            if 'ema' in checkpoint and checkpoint['ema'] is not None:
                ema_sd = _strip_or_add_module_prefix(checkpoint['ema'], want_module_prefix=False)
                ema.load_state_dict(ema_sd, strict=False)

            # --- Optimizer ---
            if 'opt' in checkpoint and checkpoint['opt'] is not None:
                try:
                    opt.load_state_dict(checkpoint['opt'])
                except Exception:
                    pass
            
            # optimizer_steps is saved in the filename
            optimizer_steps = int(os.path.basename(latest_checkpoint).split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resumed from checkpoint: {latest_checkpoint} (optimizer_steps={optimizer_steps})")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting from scratch.")

    # ==== Train loop ====
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    use_checkpoint = train_config['train'].get('use_checkpoint', True)
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    epoch_id = 0

    def set_samplers_epoch(epoch):
        ts = getattr(loader, 'sampler', None)
        vs = getattr(valid_loader, 'sampler', None) if valid_loader is not None else None
        if hasattr(ts, 'set_epoch'):
            ts.set_epoch(epoch)
        if vs is not None and hasattr(vs, 'set_epoch'):
            vs.set_epoch(epoch)

    # ==== unified stop: max_steps means number of optimizer updates ====
    max_steps = int(train_config['train']['max_steps'])

    while True:
        set_samplers_epoch(epoch_id)
        epoch_id += 1

        for x, y in loader:
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y if not torch.is_tensor(y) else y.to(device)
            else:
                x = x.to(device)
                y = y if not torch.is_tensor(y) else y.to(device)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=y)
                loss_dict = transport.training_losses(model, x, model_kwargs)
                if 'cos_loss' in loss_dict:
                    mse_loss = loss_dict["loss"].mean()
                    loss = loss_dict["cos_loss"].mean() + mse_loss
                else:
                    loss = loss_dict["loss"].mean()

                accelerator.backward(loss)

                if 'max_grad_norm' in train_config['optimizer'] and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])

                if accelerator.sync_gradients:
                    opt.step()
                    opt.zero_grad()
                    update_ema(ema, model)
                    optimizer_steps += 1

            running_loss += (mse_loss if 'cos_loss' in loss_dict else loss).item()
            log_steps += 1

            # ---- PERIODIC LOG (only at update steps; main process only) ----
            if accelerator.sync_gradients and (optimizer_steps % train_config['train']['log_every'] == 0) and optimizer_steps > 0:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time()
                updates_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction='mean')

                if accelerator.is_main_process:
                    logger.info(f"(update={optimizer_steps:07d}) Train Loss: {avg_loss.item():.4f}, Updates/Sec: {updates_per_sec:.2f}")
                    eff_global_bs = batch_size_per_gpu * accelerator.num_processes * grad_accum_steps
                    logger.info(f"Effective Global Batch Size = {eff_global_bs}  (micro={batch_size_per_gpu*accelerator.num_processes}, grad_accum_steps={grad_accum_steps})")
                    wandb.log({'Loss/train': avg_loss.item()}, step=optimizer_steps)

                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # ---- CHECKPOINT + VALID (only at update steps; main process logs only) ----
            if accelerator.sync_gradients and (optimizer_steps % train_config['train']['ckpt_every'] == 0) and optimizer_steps > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process and use_checkpoint:
                    checkpoint = {
                        "model": accelerator.get_state_dict(model),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{optimizer_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                accelerator.wait_for_everyone()

                # Validation
                if valid_loader is not None:
                    was_training = model.training
                    model.eval()
                    val_loss = evaluate(model, valid_loader, device, transport, (0.0, 1.0))
                    val_loss = accelerator.reduce(val_loss, reduction='mean')
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss.item():.4f}")
                        wandb.log({'Loss/validation': val_loss.item()}, step=optimizer_steps)
                    if was_training:
                        model.train()

            # ---- EXIT CONDITION: updates ----
            if optimizer_steps >= max_steps:
                break

        if optimizer_steps >= max_steps:
            break

    if accelerator.is_main_process:
        logger.info("Done!")
        wandb.finish()

    return accelerator


def load_weights_with_shape_check(model, checkpoint, rank=0):
    model_state_dict = model.state_dict()
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                          f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    model.load_state_dict(model_state_dict, strict=False)
    return model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_logger(logging_dir, is_main: bool):
    """Create a logger that writes to a log file and stdout for main rank; dummy otherwise."""
    logger = logging.getLogger(logging_dir)
    logger.handlers.clear()
    if is_main:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[\x1b[34m%(asctime)s\x1b[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(logging_dir, 'log.txt'))
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def evaluate(model, valid_loader, device, transport, t_range=(0.0, 1.0)):
    """A simple validation loop that returns mean loss tensor on this rank."""
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y if not torch.is_tensor(y) else y.to(device)
            loss_dict = transport.training_losses(model, x, dict(y=y))
            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict["loss"].mean()
                loss = loss_dict["cos_loss"].mean() + mse_loss
            else:
                loss = loss_dict["loss"].mean()
            total += loss.item()
            count += 1
    return torch.tensor(total / max(count, 1), device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    train_config = load_config(args.config)
    grad_accum_steps = int(train_config['train'].get('grad_accum_steps', 1))

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=grad_accum_steps)

    if accelerator.is_main_process:
        print(f"[pre] world={accelerator.num_processes} rank={accelerator.process_index} "
              f"local_rank={accelerator.local_process_index} device={accelerator.device}", flush=True)
    do_train(train_config, accelerator)
