# ------------------------------------------------------------------------------
# Extract Images from WDS .tar Archives
# ------------------------------------------------------------------------------


import os
import tarfile
import argparse
from tqdm import tqdm


def extract_images(input_dir, output_dir):
    """Extract all .jpg images from .tar archives in input_dir to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    tar_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tar")])

    if not tar_files:
        print(f"No .tar files found in {input_dir}")
        return

    total_extracted = 0

    for tar_name in tar_files:
        tar_path = os.path.join(input_dir, tar_name)
        print(f"Extracting {tar_name} ...")
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar if m.isfile() and m.name.endswith(".jpg")]
            for member in tqdm(members, desc=f"  → {tar_name}", ncols=80):
                key_name = os.path.basename(member.name)
                out_path = os.path.join(output_dir, key_name)
                if not os.path.exists(out_path):
                    with tar.extractfile(member) as f_in, open(out_path, "wb") as f_out:
                        f_out.write(f_in.read())
                    total_extracted += 1

    print(f"\nDone! Extracted {total_extracted} images to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all images from WDS .tar archives (e.g., ImageNet-1k-WDS)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing .tar files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory for extracted images."
    )
    args = parser.parse_args()
    extract_images(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
