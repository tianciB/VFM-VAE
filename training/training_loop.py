# ------------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Modifications Copyright (c) 2025, Tianci Bi, Xi'an Jiaotong University.
# This version includes substantial modifications based on NVIDIA's StyleGAN-T.
# ------------------------------------------------------------------------------


"""Main training loop."""

import os
import uuid
import math
import time
import copy
import json
import PIL.Image
import psutil
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import dnnlib

from torch_utils import training_stats
from torch_utils import misc
from torch_utils import distributed as dist
from torch_utils.ops import conv2d_gradfix
from metrics import metric_main
from training.data_wds import wds_dataloader
from torchvision.utils import save_image
from tqdm import tqdm
from typing import Union, Iterator, Optional, Any
from networks.utils.dataclasses import GeneratorForwardOutput


# -----------------------------------------------------------------# 
# Visualization                                                    #
# -----------------------------------------------------------------# 


def setup_snapshot_image_grid(
    training_set: Any,
    random_seed: int = 0,
    gw: Optional[int] = None,
    gh: Optional[int] = None,
) -> tuple[tuple[int,int], np.ndarray, np.ndarray]:

    rnd = np.random.RandomState(random_seed)
    if gw is None:
        gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    if gh is None:
        gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    elif training_set.labels_are_text:
        all_indices = list(range(len(training_set)))
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data, the extras are not used.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


def save_image_grid(
    img: torch.Tensor,
    drange: tuple[int, int],
    grid_size: tuple[int, int],
    fname: str = '',
) -> Optional[np.ndarray]:
    """Build image grid, save if fname is given"""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    # Save or return.
    if fname:
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
        else:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    else:
        return img


def save_prompts_dict(
    prompts_dict: dict,
    path: str,
    figsize: tuple[int, int] = (100,50),
) -> None:
    _, axs = plt.subplots(len(prompts_dict), 1, constrained_layout=True, figsize=figsize)
    for i, (prompt, imgs) in enumerate(prompts_dict.items()):
        axs[i].set_title(prompt)
        axs[i].imshow(imgs)
        axs[i].axis('off')
    plt.savefig(path, bbox_inches='tight')


@torch.no_grad()
def save_reconstructions(
    G_ema: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    data_prefix: str,   # train_ / val_ / test_ ...
    train_sample_dir: str,
    device: torch.device,
    cur_nimg: int,
) -> None:
    # Lists to accumulate all batches.
    real_list = []
    gen_list = []

    # Iterate over all batches and collect outputs.
    for images, labels in tqdm(data_loader, desc=f'Reconstructions at {cur_nimg // 1000} kimg'):
        # Move inputs to device.
        images = torch.from_numpy(np.stack(images)).to(device)
        labels = labels if isinstance(labels[0], str) else torch.from_numpy(np.stack(labels)).to(device)

        # Generate reconstructions.
        gen_images, *_ = G_ema(images, labels, validation=True)

        # Normalize real images to [0, 1].
        real_images = images.float() / 255.0
        if real_images.shape[-2:] != gen_images.shape[-2:]:
            real_images = F.interpolate(real_images, size=gen_images.shape[-2:], mode='area').clamp(0, 1)

        # Normalize generated images from [-1,1] to [0,1].
        gen_images = gen_images.add(1).div(2).clamp(0, 1)

        # Move to CPU and store.
        real_list.append(real_images.cpu())
        gen_list.append(gen_images.cpu())

    # Concatenate all batches along the batch dimension.
    all_reals = torch.cat(real_list, dim=0)
    all_gens  = torch.cat(gen_list, dim=0)

    # Prepare file paths
    real_image_path = os.path.join(train_sample_dir, f"{data_prefix}reals.png")
    gen_image_path = os.path.join(train_sample_dir, f"{data_prefix}gens_{cur_nimg//1000:08d}.png")

    # Save real images once if not existing.
    if not os.path.exists(real_image_path):
        # Here nrow controls how many images per row; adjust as needed
        save_image(all_reals, real_image_path, nrow=8)

    # Always save generated images.
    save_image(all_gens, gen_image_path, nrow=8)


def network_summaries(G: nn.Module, D: nn.Module, training_set: dnnlib.EasyDict, device: torch.device) -> None:
    """ Print network summaries. """
    
    # Fake image and label.
    img = torch.zeros([1, 3, 256, 256], device=device)

    if training_set.__class__.__name__ == 'WdsWrapper':
        if training_set.label_type in ['text', 'cls2text']:
            c = ["flamingo",]
        elif training_set.label_type == 'cls2id':
            c = torch.zeros([1, training_set.label_dim], device=device)
        else:
            raise ValueError(f"Unsupported label_type: {training_set.label_type}")
    
    if training_set.__class__.__name__ == 'ImageFolderDataset':
        if training_set.label_dim == 1:
            c = ["flamingo",]
        elif training_set.label_dim > 1:
            c = torch.zeros([1, training_set.label_dim], device=device)
        else:
            raise ValueError(f"Unsupported label_dim: {training_set.label_dim}")

    generator_output: GeneratorForwardOutput = misc.print_module_summary(G, [img, c, 1.0, True])
    gen_img = generator_output.gen_img
    c_enc = generator_output.global_text_tokens
    misc.print_module_summary(D, [gen_img, c_enc if training_set.label_type in ['text', 'cls2text'] else c])


# -----------------------------------------------------------------# 
# Distributed                                                      #
# -----------------------------------------------------------------# 


def sharded_all_mean(tensor: torch.Tensor, shard_size: int = 2**23) -> torch.Tensor:
    assert tensor.dim() == 1
    shards = tensor.tensor_split(math.ceil(tensor.numel() / shard_size))
    for shard in shards:
        torch.distributed.all_reduce(shard)
    tensor = torch.cat(shards) / dist.get_world_size()
    return tensor


def sync_grads(network: nn.Module, gain: Optional[int] = None) -> None:
    params = [param for param in network.parameters() if param.grad is not None]
    flat_grads = torch.cat([param.grad.flatten() for param in params])
    flat_grads = sharded_all_mean(flat_grads)
    flat_grads = flat_grads if gain is None else flat_grads * gain
    torch.nan_to_num(flat_grads, nan=0, posinf=1e5, neginf=-1e5, out=flat_grads)
    grads = flat_grads.split([param.numel() for param in params])
    for param, grad in zip(params, grads):
        param.grad = grad.reshape(param.size()).to(param.dtype)


# -----------------------------------------------------------------# 
# Data                                                             #
# -----------------------------------------------------------------# 


def split(arr: Union[list, np.ndarray, torch.Tensor], chunk_size: int, dim: int = 0) -> list:
    ''' equivalent to torch.Tensor.split, works for np/torch/list'''
    if isinstance(arr, list):
        return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]
    else:
        splits = int(np.ceil(len(arr) / chunk_size))
        return np.array_split(arr, splits, dim)


def preprocess_image(image, device):
    return image.to(device).to(torch.float32) / 255.  # Normalize to [0, 1] range


def fetch_data(
    training_set_iterator: Iterator,
    device: torch.device,
    batch_gpu: int,
) -> tuple[torch.Tensor, Union[list[str], torch.Tensor]]:
    # Fetch images and labels (necessary for training).
    real_img, real_cs = next(training_set_iterator)
    real_img = preprocess_image(real_img, device) # uint8 [0, 255] -> float32 [0, 1]
    real_cs = real_cs if isinstance(real_cs[0], str) else real_cs.to(device)

    # Split images and labels for phases.
    real_img = split(real_img, batch_gpu)
    real_cs = split(real_cs, batch_gpu)
    return real_img, real_cs


# -----------------------------------------------------------------#
# Auxiliary functions for WDS resampling                           #
# -----------------------------------------------------------------# 


def _make_resample_iterator(wds_wrapper, batch_size_per_rank, base_seed):
    """Return an *infinite* iterator on the same dataset using ResampledShards."""
    from training.data_wds import wds_dataloader
    return iter(
        wds_dataloader(
            wds_wrapper.urls,                         # the same urls as in the original wds_wrapper
            resolution=wds_wrapper.resolution,
            label_type=wds_wrapper.label_type,
            filter_keys_path=wds_wrapper.filter_keys_path,
            cls_to_text_path=wds_wrapper.cls_to_text_path,
            data_augmentation=wds_wrapper.data_augmentation,
            one_epoch=False,                          # infinite iterator
            batch_size=batch_size_per_rank,
            base_seed=base_seed,                      
        )
    )


def _sync_all_done(local_done_flag: bool, device):
    """check if all ranks have finished processing their data."""
    flag = torch.tensor(int(local_done_flag), device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
    return bool(flag.item() == torch.distributed.get_world_size())


# -----------------------------------------------------------------#
# Final-step helper                                                #
# -----------------------------------------------------------------#


def _finalize_run(
    *,                         # only keyword arguments to avoid confusion
    cur_nimg: int,
    device: torch.device,
    G: nn.Module, D: nn.Module, G_ema: nn.Module,
    run_dir: str,
    train_sample_dir: str,
    validation_flag: bool,
    validation_loader,
    metrics: list,
    training_set_kwargs: dict,
    stats_collector,
    stats_metrics: dict,
    stats_jsonl,
    snapshot_prefix: str = "network-snapshot",
):
    """
    Do *one and only one* final save / evaluation / logging pass.

    This is invoked **after** every rank has left the main training loop,
    regardless of the exit path (total_kimg reached, one-epoch exhausted,
    external early-stop, etc.).  Calling it twice is harmless because all
    I/O ops inside are idempotent (they first check file existence).
    """
    dist.print0("[finalize] starting …")

    # 1. optional reconstruction visuals
    if dist.get_rank() == 0 and validation_flag:
        save_reconstructions(
            G_ema, validation_loader,
            data_prefix='val_',
            train_sample_dir=train_sample_dir,
            device=device,
            cur_nimg=cur_nimg,
        )

    # 2. network snapshot (rank-0 writes, others sync)
    snapshot_pth = os.path.join(run_dir, f"{snapshot_prefix}-{cur_nimg//1000:08d}.pth")
    torch.save({
        "G": G.state_dict(),
        "D": D.state_dict(),
        "G_ema": G_ema.state_dict(),
        "training_set_kwargs": dict(training_set_kwargs),
    }, snapshot_pth)
    dist.print0(f"[finalize] snapshot saved → {snapshot_pth}")

    torch.distributed.barrier()

    # 3. final metric evaluation
    if metrics:
        dist.print0("[finalize] evaluating metrics …")
        for m in metrics:
            res = metric_main.calc_metric(
                metric=m, G=G_ema,
                dataset_kwargs=training_set_kwargs,
                num_gpus=dist.get_world_size(),
                rank=dist.get_rank(),
                device=device,
            )
            if dist.get_rank() == 0:
                metric_main.report_metric(res, run_dir=run_dir, snapshot_path=snapshot_pth)
            stats_metrics.update(res.results)

    # 4. stats flush (same pattern as in main loop)
    stats_collector.update()
    stats_dict = stats_collector.as_dict()
    timestamp = time.time()
    if stats_jsonl is not None:
        stats_jsonl.write(json.dumps(dict(stats_dict, timestamp=timestamp)) + '\n')
        stats_jsonl.flush()
    if dist.get_rank() == 0 and wandb.run is not None:
        global_step = int(cur_nimg / 1e3)
        for k, v in stats_dict.items():
            wandb.log({k: v.mean}, step=global_step)
        for k, v in stats_metrics.items():
            wandb.log({f"Metrics/{k}": v}, step=global_step)

    dist.print0("[finalize] done.")


# ----------------------------------------------------------------# 
# Training                                                        # 
# ----------------------------------------------------------------#


def partial_freeze(phase: dnnlib.EasyDict) -> None:
    """ Freeze the layers of the network that are not trainable. """

    if phase.name == 'D':
        if 'dino' in phase.module._modules:
            phase.module.dino.requires_grad_(False)

    elif phase.name == 'G':
        phase.module.requires_grad_(False)

        trainable_layers = phase.module.trainable_layers
        for name, layer in phase.module.named_modules():
            should_train = any(layer_type in name for layer_type in trainable_layers)
            layer.requires_grad_(should_train)


def training_loop(
    run_dir                             = '.',       # Output directory.
    train_sample_dir                    = None,      # Directory for training samples.
    wandb_project_name                  = None,      # WandB project name.
    wandb_run_name                      = None,      # WandB run name.
    training_set_kwargs                 = {},        # Options for training set.
    training_data_loader_kwargs         = {},        # Options for torch.utils.data.DataLoader of training set.
    validation_set_kwargs               = {},        # Options for validation set.
    validation_data_loader_kwargs       = {},        # Options for torch.utils.data.DataLoader of validation set.
    G_kwargs                            = {},        # Options for generator network.
    G_opt_kwargs                        = {},        # Options for generator optimizer.
    D_kwargs                            = {},        # Options for discriminator network.
    D_opt_kwargs                        = {},        # Options for discriminator optimizer.
    loss_kwargs                         = {},        # Options for loss function.
    metrics                             = [],        # Metrics to evaluate during training.
    random_seed                         = 0,         # Global random seed.
    batch_size                          = 4,         # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu                           = 4,         # Number of samples processed at a time by one GPU.
    ema_kimg                            = 10,        # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup                          = 0.05,      # EMA ramp-up coefficient. None = no rampup.
    total_kimg                          = 25000,     # Total length of the training, measured in thousands of real images.
    kimg_per_tick                       = 4,         # Progress snapshot interval.
    image_snapshot_ticks                = 50,        # How often to save image snapshots? None = disable.
    network_snapshot_ticks              = 50,        # How often to save network snapshots? None = disable.
    resume_path                         = None,      # Network path to resume training from.
    resume_kimg                         = 0,         # First kimg to report when resuming training.
    resume_discriminator                = True,      # Whether to resume discriminator weights.
    cudnn_benchmark                     = True,      # Enable torch.backends.cudnn.benchmark?
    abort_fn                            = None,      # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn                         = None,      # Callback function for updating training progress. Called for all ranks.
    one_epoch                           = False,     # Whether to train for one epoch.
    device                              = torch.device('cuda'),
) -> None:

    # Initialize.
    start_time = time.time()
    base_seed = random_seed * dist.get_world_size() + dist.get_rank()
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    n_batch_acc = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * n_batch_acc * dist.get_world_size()

    # Load training set. Choose between WDS and zip dataloader.
    dist.print0('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    if training_set_kwargs.class_name == 'training.data_wds.WdsWrapper':
        training_set_iterator = iter(wds_dataloader(
                training_set.urls, 
                resolution=training_set.resolution, 
                label_type=training_set.label_type,
                filter_keys_path=training_set.filter_keys_path,
                cls_to_text_path=training_set.cls_to_text_path,
                data_augmentation=training_set.data_augmentation,
                one_epoch=training_set.one_epoch,
                processed_tar_read_dir= training_set.processed_tar_read_dir,
                processed_tar_write_dir=training_set.processed_tar_write_dir,
                batch_size=batch_size//dist.get_world_size(),
                base_seed=base_seed,
        ))

        dist.print0('Num images:', len(training_set))
        dist.print0('Image shape:', training_set.image_shape)
        dist.print0('Label type:',  training_set.label_type)
        dist.print0('Label shape:', training_set.label_shape)
        dist.print0('Data augmentation:', training_set.data_augmentation)
        dist.print0()
    
    else:
        training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=random_seed)
        training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//dist.get_world_size(), **training_data_loader_kwargs))

        dist.print0('Num images:', len(training_set))
        dist.print0('Image shape:', training_set.image_shape)
        dist.print0('Label shape:', training_set.label_shape)
        dist.print0()
        
    # Load validation set if needed.
    validation_flag = False
    if dist.get_rank() == 0 and validation_set_kwargs != {}:
        dist.print0('Loading validation set...')
        if validation_set_kwargs.class_name == 'training.data_wds.WdsWrapper':
            raise NotImplementedError('WDS validation set is not implemented yet.')
        else:
            validation_flag = True
            validation_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs)
            validation_set.label_shape = training_set.label_shape # important for WDS to setting the label shape
            validation_set_data_loader = torch.utils.data.DataLoader(dataset=validation_set, shuffle=False, batch_size=min(batch_gpu, len(validation_set)), **validation_data_loader_kwargs)    

        dist.print0('Num images: ', len(validation_set))
        dist.print0('Image shape:', validation_set.image_shape)
        dist.print0('Label shape:', validation_set.label_shape)
        dist.print0()

    else:
        validation_set_data_loader = None

    # One epoch training.
    if one_epoch:
        dist.print0(f'One epoch mode is enabled. Training for one epoch only.')
    else:
        dist.print0(f'One epoch mode is disabled. Training for {total_kimg} kimg.')

    # Construct networks.
    dist.print0('Constructing networks...')
    G = dnnlib.util.construct_class_by_name(label_dim=training_set.label_dim, **G_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval() # EMA model, always in eval mode
    D = dnnlib.util.construct_class_by_name(c_dim=G.c_dim, **D_kwargs).train().requires_grad_(False).to(device)

    # Check for existing checkpoint
    if resume_path is not None and dist.get_rank() == 0:
        dist.print0(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if resume_discriminator and "D" in checkpoint:
            D.load_state_dict(checkpoint["D"], strict=False)
            dist.print0("Discriminator weights loaded from .pth checkpoint.")
        else:
            dist.print0("Skipping discriminator weights loading.")
        if "G" in checkpoint:
            G.load_state_dict(checkpoint["G"], strict=False)
        if "G_ema" in checkpoint:
            G_ema.load_state_dict(checkpoint["G_ema"], strict=False)
        dist.print0("Generator and EMA weights loaded from .pth checkpoint.")

    # Print network summary tables.
    if dist.get_rank() == 0:
        network_summaries(G_ema, D, training_set, device)

    # Distribute across GPUs.
    dist.print0(f'Distributing across {dist.get_world_size()} GPUs...')
    for module in [G, D, G_ema]:
        if module is not None and dist.get_world_size() > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    dist.print0('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, **loss_kwargs)
    phases = []

    for name, module, opt_kwargs in [('D', D, D_opt_kwargs), ('G', G, G_opt_kwargs)]:
        opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
        phases += [dnnlib.EasyDict(name=name, module=module, opt=opt, interval=1)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if dist.get_rank() == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    dist.print0('Exporting sample images...')
    if dist.get_rank() == 0:
        if training_set_kwargs.class_name == 'training.data_wds.WdsWrapper':
            dist.print0('WDS opening speed is slow, so we skip the sample images.')
        else:
            grid_size, images, _ = setup_snapshot_image_grid(training_set)
            save_image_grid(images, drange=[0, 255], grid_size=grid_size, fname=os.path.join(train_sample_dir, "reals.png"))

    # Initialize logs.
    dist.print0('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if dist.get_rank() == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')

    # Initialize wandb if needed.
    if dist.get_rank() == 0 and wandb_project_name is not None and wandb_run_name is not None:
        wandb.init(
            project=wandb_project_name,
            name=wandb_run_name,
            id=str(uuid.uuid4()),
            resume="never",
            config={
                "batch_size_per_gpu": batch_gpu,
                "accumulation_steps": n_batch_acc,
                "gpu_count": dist.get_world_size(),
                "lr of G": G_opt_kwargs['lr'],
                "lr of D": D_opt_kwargs['lr'],
                "total_kimg": total_kimg,
            },
        )

    # Train.
    dist.print0(f'Training started at {resume_kimg} kimg.')
    dist.print0()

    # Prepare for training loop.
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)

    # For one epoch training, we need to handle the iterator differently.
    one_epoch_exhausted = False          # check if the training set is exhausted at this rank
    infinite_iter       = None           # resampled iterator for training set

    while True:
        try:
            phase_real_img, phase_real_c = fetch_data(training_set_iterator if not one_epoch_exhausted else infinite_iter, device, batch_gpu)
        except StopIteration:
            if not one_epoch_exhausted:
                one_epoch_exhausted = True
                print(f"[one-epoch] rank {dist.get_rank():02d}: training set exhausted, switching to resampled iterator.")
                infinite_iter = _make_resample_iterator(training_set, batch_size // dist.get_world_size(), base_seed=base_seed)
                phase_real_img, phase_real_c = fetch_data(infinite_iter, device, batch_gpu)
            else:
                continue
        
        # Check if the iterator is exhausted.
        all_done = _sync_all_done(one_epoch_exhausted, device)
        if all_done:
            break
        
        # Execute training phases.
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Enable/disable gradients.
            phase.module.requires_grad_(True)
            partial_freeze(phase)

            # Accumulate gradients.
            for real_img, real_c in zip(phase_real_img, phase_real_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            params = [param for param in phase.module.parameters() if param.grad is not None]
            if len(params) > 0:
                sync_grads(network=phase.module, gain=n_batch_acc)
            phase.opt.step()
            phase.opt.zero_grad(set_to_none=True)

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(G_ema.buffers(), G.buffers()):
            b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save image snapshot.
        if (dist.get_rank() == 0) and validation_flag and (image_snapshot_ticks is not None) and (done or (cur_tick % image_snapshot_ticks == 0 and cur_tick > 0)):
            save_reconstructions(G_ema, validation_set_data_loader, data_prefix='val_', train_sample_dir=train_sample_dir, device=device, cur_nimg=cur_nimg)

        # Save network snapshot.
        data = None
        if (network_snapshot_ticks is not None) and (done or (cur_tick % network_snapshot_ticks == 0 and cur_tick > 0)):
            data = dict(G=G, D=D, G_ema=G_ema, training_set_kwargs=dict(training_set_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    for param in misc.params_and_buffers(value):
                        torch.distributed.broadcast(param, src=0)
                    data[key] = value.cpu()
                del value  # conserve memory

            if dist.get_rank() == 0:
                snapshot_pth = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:08d}.pth')
                torch.save({
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "G_ema": G_ema.state_dict(),
                    "training_set_kwargs": dict(training_set_kwargs),
                }, snapshot_pth)
                dist.print0(f"snapshot saved → {snapshot_pth}")

        torch.distributed.barrier()  # ensure all ranks are synchronized before evaluating metrics

        # Evaluate metrics.
        if cur_tick and (data is not None) and (len(metrics) > 0):
            dist.print0('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
                if dist.get_rank() == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_path=snapshot_pth)
                stats_metrics.update(result_dict.results)
        
        del data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        
        # Here, we only record the stats on rank 0.
        if dist.get_rank() == 0 and wandb.run is not None:  # ensure wandb is initialized
            global_step = int(cur_nimg / 1e3)
            for name, value in stats_dict.items():
                wandb.log({name: value.mean}, step=global_step)
            for name, value in stats_metrics.items():
                wandb.log({f'Metrics/{name}': value}, step=global_step)
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
    
    # Finalize run.
    torch.distributed.barrier()
    _finalize_run(
        cur_nimg=cur_nimg,
        device=device,
        G=G, D=D, G_ema=G_ema,
        run_dir=run_dir,
        train_sample_dir=train_sample_dir,
        validation_flag=validation_flag,
        validation_loader=validation_set_data_loader,
        metrics=metrics,
        training_set_kwargs=training_set_kwargs,
        stats_collector=stats_collector,
        stats_metrics=stats_metrics,
        stats_jsonl=stats_jsonl
    )
    
    # Done.
    dist.destroy_process_group()
    dist.print0()
    dist.print0('Exiting...')
    return
