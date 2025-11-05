# ------------------------------------------------------------------------------
# Image Transformation Preprocessing (Equivariance + Noise)
# ------------------------------------------------------------------------------

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Any
import concurrent.futures


def apply_rotation(image: np.ndarray, angle: int) -> np.ndarray:
    """Apply rotation to image array [H, W, C]."""
    if angle == 0:
        return image
    elif angle == 90:
        return np.rot90(image, k=1, axes=(0, 1))
    elif angle == 180:
        return np.rot90(image, k=2, axes=(0, 1))
    elif angle == 270:
        return np.rot90(image, k=3, axes=(0, 1))
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")


def apply_noise(image: np.ndarray, noise_level: float, idx: int, seed: int) -> np.ndarray:
    """Apply Gaussian noise deterministically using (seed + idx)."""
    np.random.seed(seed + idx)
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise * 255.0, 0, 255)
    return noisy_image.astype(np.uint8)


def get_transformation_params(idx: int, seed: int) -> Dict[str, Any]:
    """Get deterministic transformation parameters for a given image index."""
    np.random.seed(seed + idx)
    rotation_angles = [0, 90, 180, 270]
    scale_factors = [1.0, 0.75, 0.5, 0.25]
    rotation = np.random.choice(rotation_angles)
    scale = np.random.choice(scale_factors)
    return {
        'rotation': int(rotation),
        'scale': float(scale)
    }


def process_equivariance(args):
    """Generate deterministic equivariance transformation records."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(input_dir.glob('*.png')), key=lambda p: p.stem)
    transformation_records = {}

    print(f"Processing {len(image_paths)} images for equivariance transformations...")

    for idx, image_path in enumerate(tqdm(image_paths)):
        transform_params = get_transformation_params(idx, args.seed)
        transformation_records[image_path.stem] = {**transform_params}

    json_path = output_dir / "equivariance_transforms.json"
    with open(json_path, 'w') as f:
        json.dump(transformation_records, f, indent=2)

    print(f"✅ Saved equivariance transformation records to {json_path}")
    return transformation_records


def process_single_noise(args, image_path, noise_level, idx):
    """Apply Gaussian noise to a single image."""
    image = Image.open(image_path).convert('RGB')
    if image.size != (args.resolution, args.resolution):
        image = image.resize((args.resolution, args.resolution), Image.LANCZOS)
    image_array = np.array(image, dtype=np.uint8)
    noisy_image_array = apply_noise(image_array, noise_level, idx, args.seed)
    noisy_image = Image.fromarray(noisy_image_array)
    return noisy_image, image_path.name


def process_noise(args):
    """Process images with different noise levels and save results."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_paths = sorted(list(input_dir.glob('*.png')), key=lambda p: p.stem)

    print(f"Processing {len(image_paths)} images for noise transformations...")

    for noise_level in args.noise_levels:
        noise_dir = output_dir / f"noise_{noise_level:.3f}"
        noise_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing noise level: {noise_level}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_single_noise, args, image_path, noise_level, idx)
                for idx, image_path in enumerate(image_paths)
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Noise {noise_level}"):
                noisy_image, filename = future.result()
                output_path = noise_dir / filename
                noisy_image.save(output_path)

        print(f"✅ Saved noise level {noise_level} images to {noise_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess images for equivariance and noise transformations")
    parser.add_argument('--input-dir', type=str, required=True, help="Directory containing original images")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory for processed images")
    parser.add_argument('--mode', type=str, choices=['equivariance', 'noise', 'all'], default='all', help="Processing mode")
    parser.add_argument('--noise-levels', type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2], help="Noise levels to apply")
    parser.add_argument('--resolution', type=int, default=256, help="Target image resolution")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.mode in ['equivariance', 'all']:
        process_equivariance(args)

    if args.mode in ['noise', 'all']:
        process_noise(args)

    print("✅ Image preprocessing completed!")


if __name__ == '__main__':
    main()
