# ------------------------------------------------------------------------------
# SigLIP2 Encoder Wrapper
# ------------------------------------------------------------------------------


import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List
from torch_utils import distributed as dist
from torchvision.transforms.functional import normalize
from transformers import SiglipVisionModel, AutoTokenizer, SiglipTextModel


__all__ = ["SigLIP2Encoder"]


# --------------------------------------------------
# Helper utilities
# --------------------------------------------------


def _infer_patch_size(model_name: str, default: int = 16) -> int:
    name = os.path.basename(model_name.rstrip("/")).lower()
    match = re.search(r"patch(\d+)", name)
    return int(match.group(1)) if match else default


# --------------------------------------------------
# Main wrapper
# --------------------------------------------------


class SigLIP2Encoder(nn.Module):
    def __init__(
            self, 
            model_name: str, 
            conditional: bool, 
            label_type: str, 
            scale_factor: float,
            patch_from_layers: List[int],
            amp_dtype: torch.dtype = torch.bfloat16, 
            amp_enabled: bool = True
        ):
        super().__init__()
        self.model_name = model_name
        self.conditional = conditional
        self.label_type = label_type
        self.scale_factor = scale_factor
        self.patch_from_layers = patch_from_layers
        
        # AMP settings.
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled

        # ------- Vision branch -------
        self.patch_size = _infer_patch_size(model_name, default=16)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.vision_model.eval().requires_grad_(False)

        # ------- Text branch -------
        if conditional and label_type in ["text", "cls2text"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = SiglipTextModel.from_pretrained(model_name)
            self.text_model.eval().requires_grad_(False)
            self.use_text = True
        else:
            self.tokenizer = None
            self.text_model = None
            self.use_text = False

        dist.print0(
            f"✅ SigLip2Encoder ready\n"
            f"  model                    : {model_name}\n"
            f"  text_enabled             : {self.use_text}\n"
            f"  scale_factor             : {self.scale_factor}\n"
            f"  patch_from_layers        : {self.patch_from_layers}\n"
            f"  patch_size               : {self.patch_size}\n"
            f"  amp_dtype                : {self.amp_dtype}\n"
            f"  amp_enabled              : {self.amp_enabled}\n"
        )

    # ---------------- Utility ----------------

    def _preprocess_image(self, img: torch.Tensor, eq_scale_factor: float, is_eq_prior: bool) -> torch.Tensor:
        if img.dtype == torch.uint8: # default in [0, 255]
            img = img.float() / 255.0
        elif img.dtype == torch.float32: # default in [0, 1]
            pass
        
        # Resize first to match EQ regularization.
        if is_eq_prior and eq_scale_factor < 1.0:
            img = F.interpolate(img, scale_factor=eq_scale_factor, mode="bilinear", align_corners=False, antialias=True)

        # Resize twice to match the target resolution.
        if self.scale_factor != 1.0:
            img = F.interpolate(img, scale_factor=self.scale_factor, mode="bilinear", align_corners=False, 
                                antialias=(self.scale_factor < 1.0))

        img = normalize(img, mean=self._mean.to(img.device), std=self._std.to(img.device))
        return img

    # ---------------- Public API ----------------

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor, eq_scale_factor: float, is_eq_prior: bool) -> Tuple[List[torch.Tensor], torch.Tensor]:
        img = self._preprocess_image(img, eq_scale_factor, is_eq_prior)
        inputs = {"pixel_values": img}

        need_hs = any(idx != -1 for idx in self.patch_from_layers)
        with torch.amp.autocast('cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
            outputs = self.vision_model(**inputs, return_dict=True, interpolate_pos_encoding=True, output_hidden_states=need_hs)

        # Get patch features.
        patch_features = []
        for idx in self.patch_from_layers:
            if idx == -1:                 # after post_layernorm
                patch_features.append(outputs.last_hidden_state.float())
            elif idx >= 0:                # 0 for patch_embed, 1, 2, ..., N for Transformer blocks
                patch_features.append(outputs.hidden_states[idx].float())
            else:                         # -2, -3, -4, ..., -2 is the last layer before post_layernorm
                shifted = idx + 1         # -2 → -1, -3 → -2, ...
                patch_features.append(outputs.hidden_states[shifted].float())
        
        # Get global token.
        pooled_output = outputs.pooler_output.float()

        return patch_features, pooled_output

    @torch.no_grad()
    def encode_text(
        self, text: List[str]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_text:
            return None, None, None

        # Important: make sure to set padding="max_length" as that's how the model was trained.
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        device = next(self.text_model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.amp.autocast('cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
            outputs = self.text_model(**inputs, return_dict=True)
            seq_tokens = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            text_mask = inputs["attention_mask"]

        return seq_tokens.float(), pooled_output.float(), text_mask.bool()
