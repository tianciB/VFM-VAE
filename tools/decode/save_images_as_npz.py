# ------------------------------------------------------------------------------
# Build .npz from folder of .png samples
# ------------------------------------------------------------------------------


import os
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm


def create_npz_from_sample_folder(sample_dir, output_npz, num):
    """
    Builds a .npz file from a folder of .png samples.
    Reads exactly `num` images, sorted by filename.
    """
    print(f"Preparing to read {num} images from '{sample_dir}'.")

    image_files = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))

    if len(image_files) < num:
        raise ValueError(f"Found only {len(image_files)} .png images, but need {num}.")

    selected_files = image_files[:num]

    samples = []
    for fname in tqdm(selected_files, desc="Building .npz file from samples"):
        img_path = os.path.join(sample_dir, fname)
        sample_pil = Image.open(img_path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)

    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3), f"Unexpected shape: {samples.shape}"

    output_dir = os.path.dirname(output_npz)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.savez(output_npz, arr_0=samples)
    print(f"Saved .npz file to {output_npz} [shape={samples.shape}].")
    return output_npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Path to folder with .png images")
    parser.add_argument("--output-npz", type=str, required=True, help="Path to save the .npz file")
    parser.add_argument("--num", type=int, required=True, help="Number of samples to read")
    args = parser.parse_args()

    create_npz_from_sample_folder(args.input_dir, args.output_npz, args.num)
