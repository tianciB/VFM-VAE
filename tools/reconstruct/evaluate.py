# ------------------------------------------------------------------------------
# Computes LPIPS, PSNR, and SSIM between paired images in two folders
# ------------------------------------------------------------------------------

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from training.lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class PairedImageDataset(Dataset):
    """Dataset for loading corresponding image pairs from two folders."""

    def __init__(self, ref_dir: str, pred_dir: str, transform=None):
        self.ref_dir = ref_dir
        self.pred_dir = pred_dir
        self.transform = transform or (lambda x: x)

        ref_files = set(os.listdir(ref_dir))
        pred_files = set(os.listdir(pred_dir))
        self.filenames = sorted(ref_files & pred_files)
        if not self.filenames:
            raise ValueError(f"No overlapping files found between {ref_dir} and {pred_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        ref_path = os.path.join(self.ref_dir, name)
        pred_path = os.path.join(self.pred_dir, name)

        # Faster C++ backend (torchvision.io) instead of PIL
        ref = read_image(ref_path).float() / 255.0
        prd = read_image(pred_path).float() / 255.0

        # Normalize to [-1, 1]
        ref = (ref - 0.5) / 0.5
        prd = (prd - 0.5) / 0.5
        return ref, prd


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_image_metrics(
    ref_dir: str,
    pred_dir: str,
    batch_size: int = 64,
    num_workers: int = 8,
):
    """Compute LPIPS, PSNR, and SSIM for paired image folders."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Initialize metrics
    lpips_metric = LPIPS().to(device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)

    dataset = PairedImageDataset(ref_dir, pred_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    lpips_sum = torch.tensor(0.0, device=device)
    psnr_sum = torch.tensor(0.0, device=device)
    ssim_sum = torch.tensor(0.0, device=device)
    total = 0

    for ref_batch, pred_batch in tqdm(loader, desc="Evaluating", ncols=100):
        ref_batch = ref_batch.to(device, non_blocking=True)
        pred_batch = pred_batch.to(device, non_blocking=True)
        bs = ref_batch.size(0)

        # Compute LPIPS / SSIM / PSNR
        lp = lpips_metric(pred_batch, ref_batch).mean()
        ss = ssim_metric(pred_batch, ref_batch)

        # Per-image PSNR (strict definition)
        psnrs = torch.stack([
            psnr_metric(pred.unsqueeze(0), ref.unsqueeze(0))
            for pred, ref in zip(pred_batch, ref_batch)
        ])
        pn = psnrs.mean()

        lpips_sum += lp * bs
        psnr_sum += pn * bs
        ssim_sum += ss * bs
        total += bs

    lpips_avg = (lpips_sum / total).item()
    psnr_avg = (psnr_sum / total).item()
    ssim_avg = (ssim_sum / total).item()

    print("\n===== Evaluation Results =====")
    print(f"Total Images : {total}")
    print(f"Average LPIPS: {lpips_avg:.4f}")
    print(f"Average PSNR : {psnr_avg:.4f}")
    print(f"Average SSIM : {ssim_avg:.4f}")


# ------------------------------------------------------------------------------
# CLI Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LPIPS / PSNR / SSIM between image pairs (optimized).")
    parser.add_argument("--ref-dir", type=str, required=True, help="Path to reference images.")
    parser.add_argument("--pred-dir", type=str, required=True, help="Path to predicted images.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU inference.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers.")
    args = parser.parse_args()

    evaluate_image_metrics(
        ref_dir=args.ref_dir,
        pred_dir=args.pred_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )