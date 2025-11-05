# ------------------------------------------------------------------------------
# DINOv2 Encoder Wrapper
# ------------------------------------------------------------------------------


import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional
from torch_utils import distributed as dist
from torchvision.transforms.functional import normalize
from transformers import AutoModel


__all__ = ["DINOv2Encoder"]


# --------------------------------------------------
# Helper utilities
# --------------------------------------------------

def _infer_patch_size(model_name: str, default: int = 14) -> int:
    """DINOv2 large & giant are ViT-L/G with patch size 14."""
    return default


# --------------------------------------------------
# Main wrapper
# --------------------------------------------------

class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        scale_factor: float = 1.0,
        patch_from_layers: List[int] = [-1],
        amp_dtype: torch.dtype = torch.bfloat16,
        amp_enabled: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.patch_from_layers = patch_from_layers

        # AMP settings.
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

        # Model properties.
        self.patch_size = _infer_patch_size(model_name, default=14)

        # Normalization (ImageNet mean/std as used in DINOv2).
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # Load vision model.
        self.vision_model = AutoModel.from_pretrained(model_name)
        self.vision_model.eval().requires_grad_(False)

        dist.print0(
            f"✅ DINOv2Encoder ready\n"
            f"  model                   : {model_name}\n"
            f"  text_enabled            : False\n"
            f"  scale_factor            : {self.scale_factor}\n"
            f"  patch_from_layers       : {self.patch_from_layers}\n"
            f"  patch_size              : {self.patch_size}\n"
            f"  amp_dtype               : {self.amp_dtype}\n"
            f"  amp_enabled             : {self.amp_enabled}\n"
        )

    # ---------------- Utility ----------------

    def _preprocess_image(self, img: torch.Tensor, eq_scale_factor: float, is_eq_prior: bool) -> torch.Tensor:
        """Preprocess input image with optional scaling for equivariance regularization."""
        if img.dtype == torch.uint8:  # assume range [0, 255]
            img = img.float() / 255.0
        elif img.dtype == torch.float32:  # assume range [0, 1]
            pass

        # First downsample for equivariance regularization.
        if is_eq_prior and eq_scale_factor < 1.0:
            img = F.interpolate(img, scale_factor=eq_scale_factor, mode="bicubic", align_corners=False, antialias=True)

        # Second resize to match model's target resolution.
        if self.scale_factor != 1.0:
            img = F.interpolate(img, scale_factor=self.scale_factor, mode="bicubic", align_corners=False, 
                                antialias=(self.scale_factor < 1.0))
        
        img = normalize(img, mean=self._mean.to(img.device), std=self._std.to(img.device))
        return img

    # ---------------- Public API ----------------

    @torch.no_grad()
    def encode_image(
        self, img: torch.Tensor, eq_scale_factor: float, is_eq_prior: bool
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return patch-level features and global CLS embedding."""
        img = self._preprocess_image(img, eq_scale_factor, is_eq_prior)
        inputs = {"pixel_values": img}

        need_hs = any(idx != -1 for idx in self.patch_from_layers)
        with torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
            outputs = self.vision_model(
                **inputs,
                return_dict=True,
                output_hidden_states=need_hs,
            )

        # Collect patch features from hidden states.
        patch_features = []
        for idx in self.patch_from_layers:
            if idx == -1:  # final normalized output
                patch_features.append(outputs.last_hidden_state[:, 1:, :].float())
            elif idx >= 0:
                patch_features.append(outputs.hidden_states[idx][:, 1:, :].float())
            else:  # negative indexing from the end
                shifted = idx + 1
                patch_features.append(outputs.hidden_states[shifted][:, 1:, :].float())

        # CLS token is the first token in last_hidden_state.
        pooled_output = outputs.pooler_output.float()

        return patch_features, pooled_output


    @torch.no_grad()
    def encode_text(
        self, text: List[str]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        DINOv2 does not provide a text encoder.
        This method is intentionally unimplemented to keep interface consistency.
        """
        return None, None, None