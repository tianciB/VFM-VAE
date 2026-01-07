# ------------------------------------------------------------------------------
# REG (Original) Training Codes of LightningDiT integrated with VA-VAE
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Modifications Copyright (c) 2025, Tianci Bi, Xi'an Jiaotong University.
# This version includes minor modifications to support:
#    - parse_args for latent dimensions
#    - auto-resume from latest checkpoint
#    - support qk-norm and deepspeed options
# ------------------------------------------------------------------------------


import os
import math
import json
import copy
import logging
import argparse
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm.auto import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders
from dataset import CustomDataset

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_mean, latents_std):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return (z - latents_mean) / latents_std


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # Set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    deepspeed_plugin = None
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=args.deepspeed_stage,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        deepspeed_plugin=deepspeed_plugin,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model    
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.latent_size
    latent_dim = args.vae_latent_dim

    # Use raw images with encoders
    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=latent_dim,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Setup data
    train_dataset = CustomDataset(
        args.data_dir, 
        image_folder=args.image_folder, 
        vae_folder=args.vae_folder
        )
    # Get latent stats for VAE normalization
    latents_stats_path = os.path.join(args.data_dir, args.vae_folder, "latents_stats.pt")
    if os.path.exists(latents_stats_path):
        latents_stats = torch.load(latents_stats_path, map_location='cpu')
        latents_mean = latents_stats['mean'].reshape(1, args.vae_latent_dim, 1, 1).to(device)
        latents_std = latents_stats['std'].reshape(1, args.vae_latent_dim, 1, 1).to(device)
        print(f"Loaded VAE latents stats from  {latents_stats_path}")
    else:
        # Fallback to default values
        latents_mean = torch.tensor([0.] * args.vae_latent_dim).view(1, args.vae_latent_dim, 1, 1).to(device)
        latents_std = torch.tensor([1 / 0.18215] * args.vae_latent_dim).view(1, args.vae_latent_dim, 1, 1).to(device)
        print(f"No latents stats found at {latents_stats_path}, using default mean/std.")

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # per-micro forward batch size
    if args.batch_size_per_gpu is None:
        local_batch_size = int(args.batch_size // accelerator.num_processes)
    else:
        local_batch_size = int(args.batch_size_per_gpu)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    ckpt_path = None
    checkpoint_dir = os.path.join(args.output_dir, args.exp_name, "checkpoints")

    if args.resume_step > 0:
        # resume from specified step
        ckpt_name = str(args.resume_step).zfill(7) + ".pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    else:
        # auto resume from latest
        if os.path.isdir(checkpoint_dir):
            ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
            if len(ckpts) > 0:
                # extract step number from filenames like 0001000.pt
                steps = [int(os.path.splitext(f)[0]) for f in ckpts if os.path.splitext(f)[0].isdigit()]
                if len(steps) > 0:
                    latest_step = max(steps)
                    ckpt_path = os.path.join(checkpoint_dir, f"{latest_step:07d}.pt")

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        global_step = ckpt["steps"]
    else:
        print("No checkpoint found, training from scratch.")

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="VFM-VAE",
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": f"{args.exp_name}",
                    "group": "LDM-diffusion, REG",
                    "dir": save_dir,
                }
            },
        )

        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

        
    for epoch in range(args.epochs):
        model.train()
        for batch_data in train_dataloader:
            # CustomDataset returns (raw_image, vae_features, labels)
            raw_image, x, y = batch_data
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
        
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            
            with torch.no_grad():
                # Sample VAE posterior
                x = sample_posterior(x, latents_mean, latents_std)

                # Process encoders
                zs = []
                cls_token = None
                # Use encoders to process raw images
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'dinov2' in encoder_type:
                            dense_z = z['x_norm_patchtokens']
                            cls_token = z['x_norm_clstoken']
                            dense_z = torch.cat([cls_token.unsqueeze(1), dense_z], dim=1)
                        else:
                            exit()
                        zs.append(dense_z)

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    model_kwargs = dict(y=labels)
                    loss1, proj_loss1, time_input, noises, loss2 = loss_fn(model, x, model_kwargs, zs=zs,
                                                                        cls_token=cls_token,
                                                                        time_input=None, noises=None)
                    loss_mean = loss1.mean()
                    loss_mean_cls = loss2.mean() * args.cls
                    proj_loss_mean = proj_loss1.mean() * args.proj_coeff
                    loss = loss_mean + proj_loss_mean + loss_mean_cls

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    if not torch.is_tensor(grad_norm):
                        grad_norm = torch.tensor(grad_norm, device=device)
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                logging.info("Generating EMA samples done.")

            logs = {
                "loss_final": accelerator.gather(loss).mean().detach().item(),
                "loss_mean": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "loss_mean_cls": accelerator.gather(loss_mean_cls).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }

            log_message = ", ".join(f"{key}: {value:.6f}" for key, value in logs.items())
            logging.info(f"Step: {global_step}, Training Logs: {log_message}")

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # deepspeed
    parser.add_argument("--use-deepspeed", action="store_true")
    parser.add_argument("--deepspeed-stage", type=int, default=2, choices=[2, 3])

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ops-head", type=int, default=16)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../reg/imagenet_train_256x256/")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--image-folder", type=str, default="images", help="Raw images folder")
    parser.add_argument("--vae-folder", type=str, default="vae_latents", help="VAE latents folder")
    
    # latent dimensions
    parser.add_argument("--latent-size", type=int, default=16, help="Latent spatial size")
    parser.add_argument("--vae-latent-dim", type=int, default=32, help="VAE latent dimension")

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=1000000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cls", type=float, default=0.03)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
