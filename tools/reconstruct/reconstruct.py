# ------------------------------------------------------------------------------
# Multi-GPU image reconstruction for VFM-VAE
# ------------------------------------------------------------------------------

import os
import yaml
import torch
import dnnlib
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from networks.utils.dataclasses import GeneratorForwardOutput


# --------------------------- Dataset --------------------------- #
class ImageFolderWithNames(Dataset):
    """Dataset returning (image_tensor, filename)."""
    def __init__(self, root, transform=None, exts=('.jpg', '.jpeg', '.png')):
        self.root = root
        self.transform = transform
        self.samples = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(exts)
        ]
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(path)
        return img, filename


# --------------------------- Utilities --------------------------- #
def safe_save(tensor, path):
    img = to_pil_image(tensor.clamp(0, 1))
    img.save(path)
    img.close()


@torch.no_grad()
def run_rfid_reconstruction(
    vae_encoder,
    validation_set,
    output_dir,
    batch_size_per_gpu,
    rank,
    world_size,
    device,
):
    recon_input_dir = os.path.join(output_dir, "inputs")
    recon_output_dir = os.path.join(output_dir, "outputs")
    os.makedirs(recon_input_dir, exist_ok=True)
    os.makedirs(recon_output_dir, exist_ok=True)

    sampler = DistributedSampler(
        validation_set, rank=rank, num_replicas=world_size, shuffle=False, seed=42
    )
    loader = DataLoader(
        validation_set, sampler=sampler, batch_size=batch_size_per_gpu,
        num_workers=4, pin_memory=True, drop_last=False
    )

    for images, names in tqdm(loader, desc=f"[Rank {rank}] Reconstruction"):
        images = images.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            generate_output: GeneratorForwardOutput = vae_encoder(images, names, validation=True)
        gen_images = generate_output.gen_img  # [-1, 1]

        real = images
        gen = gen_images.add(1).div(2).clamp(0, 1)  # [0, 1]

        for i, name in enumerate(names):
            base, _ = os.path.splitext(name)
            safe_save(real[i].cpu(), os.path.join(recon_input_dir, f"{base}.png"))
            safe_save(gen[i].cpu(), os.path.join(recon_output_dir, f"{base}.png"))


# --------------------------- Main --------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',          type=str, required=True, help='Path to input image folder')
    parser.add_argument('--output-dir',         type=str, required=True, help='Path to output folder')
    parser.add_argument('--vae-pth',            type=str, required=True, help='Path to VAE model (.pth)')
    parser.add_argument('--use-config',         type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resolution',         type=int, default=256, help='Input image resolution')
    parser.add_argument('--batch-size-per-gpu', type=int, default=32, help='Batch size per GPU')
    args = parser.parse_args()

    # 1. Load YAML config (shared training/inference config)
    with open(args.use_config, "r") as f:
        full_cfg = yaml.safe_load(f)

    # Extract model construction parameters
    vae_kwargs = full_cfg.get("G_kwargs", {})
    vae_kwargs["label_dim"] = 1000                  # default for ImageNet
    vae_kwargs["img_resolution"] = args.resolution  # set resolution
    vae_kwargs["conditional"] = False               # reconstruction is unconditional
    vae_kwargs["label_type"] = "cls2text"           # dummy, not used
    vae_kwargs["use_kl_loss"] = False               # disable KL loss for validation
    vae_kwargs["use_vf_loss"] = False               # disable VF loss for validation
    vae_kwargs["num_fp16_res"] = 0                  # disable fp16 for validation

    # 2. Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 3. Build model
    vae_encoder = dnnlib.util.construct_class_by_name(**vae_kwargs).to(device)
    vae_encoder.requires_grad_(False)

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

    # 4. Dataset
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])
    val_set = ImageFolderWithNames(root=args.input_dir, transform=transform)

    # 5. Reconstruction
    run_rfid_reconstruction(
        vae_encoder=vae_encoder,
        validation_set=val_set,
        output_dir=args.output_dir,
        batch_size_per_gpu=args.batch_size_per_gpu,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
