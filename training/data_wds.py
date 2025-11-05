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


"""Streaming images and labels from webdatasets."""


import os
import json
import torch
import random
import pickle
import logging
import functools
import PIL.Image
import numpy as np
import webdataset as wds

from glob import glob
from pathlib import Path
from functools import partial
from torch_utils import distributed as dist
from torch.utils.data import get_worker_info
from typing import Union, Optional


# ------------------------------------------------------------------#
# global seeding helpers                                            #
# ------------------------------------------------------------------#

DEFAULT_SEED = 42

def _safe_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _seed_worker(worker_id: int, base_seed: int):
    """Top-level worker init function for DataLoader."""
    seed = base_seed + _safe_rank() * 1_000 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_worker_init(base_seed: int):
    """
    Factory function to create a worker init function for DataLoader.
    """
    return functools.partial(_seed_worker, base_seed=base_seed)


# -----------------------------------------------------------------------------#
# ShardTracker — record fully consumed tar shards in one-epoch mode            #
# -----------------------------------------------------------------------------#


class ShardTracker:
    """
    Maintain a set of already‐seen tar URLs. When a sample’s tar URL is encountered
    for the first time, record it immediately to processed_tars_rankXX.txt.
    """

    def __init__(self, log_dir: str):
        self.rank = dist.get_rank() if dist.is_initialized() else 5
        self.log_dir = log_dir
        
        # Ensure the directory exists on this rank’s node
        os.makedirs(self.log_dir, exist_ok=True)

        # File path for this rank’s processed shards
        self.log_path = os.path.join(self.log_dir, f"processed_tars_rank{self.rank:02d}.txt")

        # Load any already‐recorded shards to avoid duplicates
        self.processed_set = set()
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.processed_set.add(line)

        print(f"[one-epoch] rank {self.rank:02d} ShardTracker initialized, log_path={self.log_path}")

    def __call__(self, sample: dict):
        """
        Called for each sample in the WebDataset pipeline.
        If the sample's tar URL has not been seen before, record it immediately.
        """

        # Extract the current shard URL from the sample
        url = sample.get("__url__") or sample.get("__src__")
        if url is None:
            return sample

        # If this tar hasn’t been seen yet, record it and add to the set
        if url not in self.processed_set:
            with open(self.log_path, "a") as f:
                f.write(url + "\n")
            self.processed_set.add(url)
            print(f"[one-epoch] rank {self.rank:02d} recorded shard: {url}")

        return sample


# -----------------------------------------------------------------------------#
# misc helpers                                                                 #
# -----------------------------------------------------------------------------#


def get_all_processed_tars(processed_tar_read_dir: str, workers: int) -> list[str]:
    """
    Read all 'processed_tars_rank*.txt' under processed_tar_read_dir,
    return a sorted list of all recorded shard URLs (deduplicated),
    but only keep the relative tail: 'capsfusion_120m_xx/xxxxx.tar'
    """
    processed = set()
    if processed_tar_read_dir and os.path.isdir(processed_tar_read_dir):
        files = glob(os.path.join(processed_tar_read_dir, "processed_tars_*.txt"))
        if _safe_rank() == 0:
            print(f"[one-epoch] Found {len(files)} processed shard files.")
        for txt_file in files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()[:-workers]  # skip last N lines for workers because they may not be fully processed
                for line in lines:
                    line = line.strip()
                    if line:
                        tail = os.path.join(os.path.basename(os.path.dirname(line)), os.path.basename(line))
                        processed.add(tail)
    if _safe_rank() == 0 and len(processed) > 0:
        print(f"[one-epoch] Total skipped shard names: {len(processed)}. Sample: {list(processed)[:3]}")
    return sorted(processed)


def get_tail(p):  # e.g. capsfusion_120m_09/00304.tar
    return os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))


def log_and_continue(exn) -> bool:
    logging.warning(f'Webdataset error ({repr(exn)}). Ignoring.')
    return True


# -----------------------------------------------------------------------------#
# basic preprocessing helpers                                                  #
# -----------------------------------------------------------------------------#


def preprocess_img(img: PIL.Image, resolution: int = 256) -> np.ndarray:
    img = np.array(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis] # HW => HWC
    # img = center_crop(resolution, resolution, img) # no transform for text label
    img = img.transpose(2, 0, 1) # HWC => CHW
    assert img.shape[-1] == resolution, f"Image width {img.shape[-1]} does not match expected resolution {resolution}."
    return img.astype(np.uint8)


def preprocess_txt(text: str) -> str:
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    return text.strip()


def filter_no_caption(sample: dict) -> bool:
    return 'text' in sample and isinstance(sample['text'], str) and sample['text'] != ""


def filter_by_filename(sample: dict, keep_set: list[str]) -> bool:
    if keep_set is None:
        return True
    return sample['__key__'] in keep_set


def center_crop(width: int, height: int, img: np.ndarray) -> np.ndarray:
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.LANCZOS)
    return np.array(img)


def transform_image(img: PIL.Image.Image, resolution: int, augment: bool):
    img = np.array(img)
    
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    h, w = img.shape[:2]
    crop_ratio = random.uniform(0.5, 1.0) if augment else 1.0
    crop_size = max(1, int(min(h, w) * crop_ratio))

    top = random.randint(0, h - crop_size) if augment else max((h - crop_size) // 2, 0)
    left = random.randint(0, w - crop_size) if augment else  max((w - crop_size) // 2, 0)

    img = img[top:top+crop_size, left:left+crop_size]
    img = PIL.Image.fromarray(img, 'RGB').resize((resolution, resolution), PIL.Image.LANCZOS)
    img = np.array(img)

    if augment and random.random() < 0.5:
        img = np.ascontiguousarray(np.flip(img, axis=1))

    img = img.transpose(2, 0, 1)
    
    return img.astype(np.uint8)


def to_text_label(label: int, cls2text: dict) -> str:
    return cls2text[str(label)] if cls2text is not None else str(label)


def to_one_hot(label: int, num_classes: int) -> np.ndarray:
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[int(label)] = 1.0
    return one_hot


# -----------------------------------------------------------------------------#
# wds_dataloader – build pipeline & return proper iterator                     #
# -----------------------------------------------------------------------------#


def wds_dataloader(
    train_data: list[str],
    *,
    batch_size: int,
    resolution: int,
    workers: int = 3,
    shard_shuffle_size: int = 50_000,
    sample_shuffle_size: int = 50_000,
    label_type: str = "text",
    filter_keys_path: Optional[str] = None,
    cls_to_text_path: Optional[str] = None,
    data_augmentation: bool = False,
    one_epoch: bool = False,
    processed_tar_read_dir: Optional[str] = None,
    processed_tar_write_dir: Optional[str] = None,
    base_seed: Optional[int] = None,
) -> wds.WebLoader:

    assert base_seed is not None, "base_seed must be provided for reproducibility."

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    # ---------- optional key-filter & cls→text mapping ----------------------
    keep_set = None
    if filter_keys_path and os.path.isfile(filter_keys_path):
        keep_set = set(pickle.load(open(filter_keys_path, "rb")))

    cls2text = None
    if cls_to_text_path and os.path.isfile(cls_to_text_path):
        cls2text = json.load(open(cls_to_text_path, "r", encoding="utf-8"))

    num_classes = len(cls2text) if cls2text else 0
    
    # ---------- choose shard source & tracker ------------------------------
    if one_epoch:
        if processed_tar_read_dir:
            skipped_tail = set(get_all_processed_tars(processed_tar_read_dir, workers))
            old_total = len(train_data)

            # Find the skipped shards and filter them out.
            skipped_full = [u for u in train_data if get_tail(u) in skipped_tail]
            train_data   = [u for u in train_data if get_tail(u) not in skipped_tail]

            dist.print0(f"[one-epoch] skipped {len(skipped_full)} shards, {len(train_data)} remain from {old_total}.")

            # Write the skipped shards to a log file for next resume.
            if processed_tar_write_dir:
                os.makedirs(processed_tar_write_dir, exist_ok=True)
                log_path = os.path.join(processed_tar_write_dir, f"processed_tars_rank{_safe_rank():02d}.txt")
                with open(log_path, "a") as f:
                    for u in skipped_full:
                        f.write(u.strip() + "\n")
        else:
            dist.print0(f"[one-epoch] {len(train_data)} shards to process, no skipped.")

        tracker = ShardTracker(processed_tar_write_dir) if processed_tar_write_dir else None
        random.shuffle(train_data) # shuffle before splitting by node/worker
        source = wds.SimpleShardList(train_data)

    else:
        tracker = None
        source  = wds.ResampledShards(train_data)   # infinite stream

    # ---------- common pipeline pieces -------------------------------------
    pipeline = [
        source,
        wds.split_by_node,
        wds.split_by_worker,
        wds.shuffle(shard_shuffle_size),
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.shuffle(sample_shuffle_size),
    ]

    if tracker:
        pipeline.append(wds.map(tracker))

    # ---------- branch by label type ---------------------------------------
    if label_type == "text":
        pipeline += [
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png", text="txt"),
            wds.map_dict(
                image=partial(preprocess_img, resolution=resolution),
                text=preprocess_txt),
            wds.select(filter_no_caption),
            wds.to_tuple("image", "text"),
            wds.batched(batch_size),
        ]
    
    elif label_type in ["cls2text", "cls2id"]:
        pipeline += [
            wds.select(partial(filter_by_filename, keep_set=keep_set)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png", label="cls"),
            wds.map_dict(
                image=partial(transform_image, resolution=resolution, 
                              augment=data_augmentation),
                label=partial(to_text_label, cls2text=cls2text) 
                              if label_type == "cls2text" 
                              else partial(to_one_hot, num_classes=num_classes),
            ),
            wds.to_tuple("image", "label"),
            wds.batched(batch_size),
        ]
    
    else:
        raise ValueError(f"Unsupported label_type: {label_type}")

    # ---------- build WebLoader --------------------------------------------
    web_loader = wds.WebLoader(
        wds.DataPipeline(pipeline),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=_make_worker_init(base_seed),
    )
    return web_loader


class WdsWrapper():
    """
    Lightweight wrapper so training code can call  len(dataset), dataset[i]  etc.
    Only used for image-level visualisation / stats; not critical for training.
    """

    def __init__(
        self,
        path: str,
        resolution: int,
        label_type: str = "text",
        filter_keys_path: Optional[str] = None,
        cls_to_text_path: Optional[str] = None,
        data_augmentation: bool = False,
        one_epoch: bool = False,
        processed_tar_read_dir: Optional[str] = None,
        processed_tar_write_dir: Optional[str] = None,
        **kwargs,
    ):
        self._root      = Path(path)
        self.resolution = resolution
        self.label_type = label_type
        self.filter_keys_path = filter_keys_path
        self.cls_to_text_path = cls_to_text_path
        self.data_augmentation = data_augmentation
        self.one_epoch  = one_epoch
        self.processed_tar_read_dir = processed_tar_read_dir
        self.processed_tar_write_dir = processed_tar_write_dir

        if cls_to_text_path and os.path.isfile(cls_to_text_path):
            self._cls2text = json.load(open(cls_to_text_path, "r", encoding="utf-8"))
            self.num_classes = len(self._cls2text)
        else:
            self._cls2text = None
            self.num_classes = 0

        self.urls = self._get_urls(path)

        # quick preview – optional
        self.first_images, self.first_labels = [], []
        # self.init_viz()

    def _get_urls(self, path: str) -> list[str]:
        '''
        Expected file structure:
        path/
        path/part1/0000.tar
        path/part1/0001.tar
        ...
        path/part2/0000.tar
        path/part2/0001.tar
        ...

        Dataloader can be used while the dataset is still downloading,
        only fully downloaded tars will be used.
        '''

        if self.label_type in ["cls2text", "cls2id"]:
            return sorted(glob(f'{path}/**/*.tar', recursive=True))
        elif self.label_type == "text":
            jsons = glob(f'{path}/**/*.json')
            urls = [p.replace('_stats.json', '.tar') for p in jsons]
            return urls
        else:
            raise ValueError(f"Unsupported label_type: {self.label_type}")

    def init_viz(self) -> None:
        gw = np.clip(7680 // self.resolution, 7, 32)
        gh = np.clip(4320 // self.resolution, 4, 32)
        dl = iter(wds_dataloader(
            self.urls[:1],              # only one shard for viz
            batch_size=gw*gh,
            workers=0,
            resolution=self.resolution,
            label_type=self.label_type,
            filter_keys_path=self.filter_keys_path,
            cls_to_text_path=self.cls_to_text_path,
            data_augmentation= self.data_augmentation,
            one_epoch=self.one_epoch,
            processed_tar_read_dir=self.processed_tar_read_dir,
            processed_tar_write_dir=self.processed_tar_write_dir,
            base_seed=DEFAULT_SEED,
        ))
        
        try:
            self.first_images, self.first_labels = next(dl)
        except StopIteration:
            logging.warning("WdsWrapper: Visualization sample loading failed.")
            self.first_images, self.first_labels = [], []

    def __len__(self) -> int:
        if self.label_type in ("cls2text", "cls2id"):
            if self.filter_keys_path and os.path.isfile(self.filter_keys_path):
                return len(set(pickle.load(open(self.filter_keys_path, "rb"))))
            return 1281167            # Imagenet-1k
        return len(self.urls) * 10000 # heuristic for WDS

    @property
    def image_shape(self) -> list[int]:
        return [3, self.resolution, self.resolution]

    @property
    def label_shape(self) -> list[int]:
        return [self.num_classes] if self.label_type in ("cls2text", "cls2id") else [1]
    
    @property
    def label_dim(self) -> int:
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def name(self) -> str:
        return self._root.name

    # dummy getter for visualisation
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str] :
        return self.first_images[idx], self.first_labels[idx]
