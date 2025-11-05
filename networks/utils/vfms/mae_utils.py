# ------------------------------------------------------------------------------
# MAE Encoder Wrapper
# ------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional
from torch_utils import distributed as dist
from torchvision.transforms.functional import normalize
from transformers import ViTMAEModel


__all__ = ["MAEEncoder"]


# --------------------------------------------------
# Helper utilities
# --------------------------------------------------

def _infer_patch_size(model_name: str) -> int:
    """MAE ViT models use patch size 16."""
    return 16

# --------------------------------------------------
# Main wrapper
# --------------------------------------------------

class MAEEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/vit-mae-large",
        scale_factor: float = 1.0,
        patch_from_layers: List[int] = [-1],   # 0=patchify, 1...N=encoder blocks, -1=last layer
        amp_dtype: torch.dtype = torch.bfloat16,
        amp_enabled: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.patch_from_layers = patch_from_layers
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

        self.patch_size = _infer_patch_size(model_name)

        # MAE Normalization (ImageNet mean/std).
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # Load vision model.
        self.vision_model = ViTMAEModel.from_pretrained(model_name)
        self.vision_model.eval().requires_grad_(False)

        # Hook storage.
        self._features = {}
        self._register_hooks()

        dist.print0(
            f"✅ MAEEncoder ready\n"
            f"  model                   : {model_name}\n"
            f"  text_encoder            : False\n"
            f"  scale_factor            : {scale_factor}\n"
            f"  patch_from_layers       : {self.patch_from_layers}\n"
            f"  patch_size              : {self.patch_size}\n"
            f"  amp_dtype               : {self.amp_dtype}\n"
            f"  amp_enabled             : {self.amp_enabled}\n"
        )

    # ---------------- Register hooks ----------------

    def _register_hooks(self):
        """Register forward hooks to extract patchify and block features."""

        def hook_patchify(module, input, output):
            self._features["patchify"] = output.detach() # embeddings output: [B, N+1, D]

        def hook_block(module, input, output):
            if "blocks" not in self._features:
                self._features["blocks"] = []
            self._features["blocks"].append(output.detach())

        # Patchify features (after embeddings).
        self.vision_model.embeddings.register_forward_hook(hook_patchify)

        # Encoder block outputs.
        for blk in self.vision_model.encoder.layer:
            blk.register_forward_hook(hook_block)

    # ---------------- Utility ----------------

    def _preprocess_image(self, img: torch.Tensor, eq_scale_factor: float, is_eq_prior: bool) -> torch.Tensor:
        """Preprocess input image with optional scaling for equivariance regularization."""
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        # First, resize for equivariance regularization.
        if is_eq_prior and eq_scale_factor < 1.0:
            img = F.interpolate(img, scale_factor=eq_scale_factor, mode="bilinear", align_corners=False)

        # Second, resize to target resolution.
        if self.scale_factor != 1.0:
            img = F.interpolate(img, scale_factor=self.scale_factor, mode="bilinear", align_corners=False,
                                antialias=(self.scale_factor < 1.0))

        img = normalize(img, mean=self._mean.to(img.device), std=self._std.to(img.device))
        return img

    # ---------------- Public API ----------------

    @torch.no_grad()
    def encode_image(
        self, img: torch.Tensor, eq_scale_factor: float=1.0, is_eq_prior: bool=False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return patch-level features and pooled embedding."""
        self._features.clear()
        img = self._preprocess_image(img, eq_scale_factor, is_eq_prior)
        inputs = {"pixel_values": img}

        with torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
            outputs = self.vision_model(**inputs, return_dict=True)

        patch_features = []
        for idx in self.patch_from_layers:
            if idx == 0:  # patchify features
                patch_features.append(self._features["patchify"][:, 1:, :].float())
            elif idx == -1:  # final layer
                patch_features.append(outputs.last_hidden_state[:, 1:, :].float())
            else:  # intermediate blocks (1-based indexing)
                patch_features.append(self._features["blocks"][idx - 1][:, 1:, :].float())

        # Global embedding = mean pooling over patches.
        pooled_output = outputs.last_hidden_state[:, 1:, :].mean(dim=1).float()

        return patch_features, pooled_output

    @torch.no_grad()
    def encode_text(
        self, text: List[str]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """MAE does not provide a text encoder."""
        return None, None, None
