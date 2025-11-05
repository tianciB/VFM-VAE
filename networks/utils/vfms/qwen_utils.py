# ------------------------------------------------------------------------------
# Qwen2.5-VL Encoder Wrapper
# ------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import List, Optional, Tuple
from torch_utils import distributed as dist
from torchvision.transforms.functional import normalize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


__all__ = ["QwenEncoder"]


# --------------------------------------------------
# Main wrapper
# --------------------------------------------------

# This is forward-hook function to save outputs from vision layers.
def _save_hook(idx: int, store: dict, _module, _inp, output):
    store[idx] = output


class QwenVisionEncoder(nn.Module):
    """Light-weight vision & (optional) text encoder for Qwen 2.5-VL.

    *   Images are resized to **448x448** (bicubic) → patch size **14** → **32x32** tokens.
    *   Default `merge_factor=2` averages 2x2 windows → **16x16 = 256** tokens.
    *   Text branch only keeps **token embedding layer** (no decoder LLM).
    """

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
        self.patch_size: int = 14           # default patch size
        self.merge_size: int = 2            # default merge size
        self.temporal_patch_size: int = 2   # default temporal patch size

        mean = [0.48145466, 0.4578275, 0.40821073]
        std  = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # Load full model to grab vision tower & token embeddings only.
        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
        full.eval().requires_grad_(False)
        
        # Different versions of transformers may have different attributes.
        if hasattr(full.model, "visual"):
            self.vision_model = full.model.visual
        elif hasattr(full, "visual"): # for transformers == 4.50.1
            self.vision_model = full.visual
        
        # Keep only token‑embedding layer for text (optional).
        if self.conditional and self.label_type in ["text", "cls2text"]:
            embed = full.get_input_embeddings()  # nn.Embedding(vocab, dim)
            self.token_embedding = nn.Embedding.from_pretrained(embed.weight.detach(), freeze=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if hasattr(full, "language_model"):
                self.text_model = full.language_model
            elif hasattr(full, "visual"): # for transformers == 4.50.1
                self.text_model = full.model
            self.use_text = True
            # Reference: https://github.com/QwenLM/Qwen2.5-VL/issues/951.
            dist.print0("⚠️ Warning: This implementation has not been thoroughly tested with text inputs.")
        else:
            self.token_embedding = None
            self.tokenizer = None
            self.use_text = False

        # Free the rest of the full model to save memory.
        del full

        dist.print0(
            f"\n✅ QwenVisionEncoder ready\n"
            f"  model                   : {model_name}\n"
            f"  text_enabled            : {self.use_text}\n"
            f"  scale_factor            : {scale_factor}\n"
            f"  patch_from_layers       : {self.patch_from_layers}\n"
            f"  patch_size              : {self.patch_size}\n"
            f"  merge_size              : {self.merge_size}\n"
            f"  temporal_patch_size     : {self.temporal_patch_size}\n"
            f"  amp_dtype               : {self.amp_dtype}\n"
            f"  amp_enabled             : {self.amp_enabled}\n"
        )

    # ---------------- Utility ----------------

    @torch.no_grad()
    def _preprocess_image(self, img: torch.Tensor, eq_scale_factor: float, is_eq_prior: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            img: (B, C, H, W), dtype uint8 in [0,255] or float32 in [0, 1]
        Output:
            flatten_patches: (B, N_patches, C * temporal_patch_size * patch_size * patch_size)
            grid_thw: (B, 3), torch.long, each row is (grid_t, grid_h, grid_w)
        """
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.dtype == torch.float32:
            pass

        # First, resize for equivariance regularization.
        if is_eq_prior and eq_scale_factor < 1.0:
            img = F.interpolate(img, scale_factor=eq_scale_factor, mode="bicubic", align_corners=False, antialias=True)

        # Second, resize to target resolution.
        if self.scale_factor != 1.0:
            img = F.interpolate(img, scale_factor=self.scale_factor, mode="bicubic", align_corners=False,
                                antialias=(self.scale_factor < 1.0))

        # Normalize.
        img = normalize(img, mean=self._mean.to(img.device), std=self._std.to(img.device))  # (B, C, H, W)

        '''
        # Compute grid_thw
        B, _, H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Resolution must be divisible by patch_size"
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        grid_t = 1  # single frame
        grid_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long, device=img.device).expand(B, 3)

        # Simulate temporal dim
        img = img.unsqueeze(2).repeat(1, 1, self.temporal_patch_size, 1, 1)  # (B, C, T, H, W)

        return img, grid_thw
        '''

        # Parameters.
        B, C, H, W = img.shape
        patch_size = self.patch_size                    # e.g., 16
        merge_size = self.merge_size                    # e.g., 2
        temporal_patch_size = self.temporal_patch_size  # e.g., 2

        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"

        # Simulate temporal dim: (B, T=1, C, H, W).
        img = img.unsqueeze(1)  # (B, 1, C, H, W)

        if temporal_patch_size > 1:
            T = img.shape[1]
            pad_T = (temporal_patch_size - T % temporal_patch_size) % temporal_patch_size
            if pad_T > 0:
                img = torch.cat([img, img[:, -1:].repeat(1, pad_T, 1, 1, 1)], dim=1)  # (B, T', C, H, W)

        B, T, C, H, W = img.shape
        grid_t = T // temporal_patch_size
        grid_h = H // patch_size
        grid_w = W // patch_size

        # Reshape and permute for patchify.
        img = img.reshape(
            B, grid_t, temporal_patch_size, C,
            grid_h // merge_size, merge_size, patch_size,
            grid_w // merge_size, merge_size, patch_size,
        )  # (B, gt, tp, C, ghg, mh, ph, gwg, mw, pw)

        img = img.permute(
            0, 1, 4, 7, 5, 8, 3, 2, 6, 9
        )  # (B, gt, ghg, gwg, mh, mw, C, tp, ph, pw)

        # Flatten to (B, N_patches, D).
        flatten_patches = img.reshape(
            B,
            grid_t * grid_h * grid_w,
            C * temporal_patch_size * patch_size * patch_size
        )  # (B, N_patches, D)

        # Construct grid_thw: (B, 3), long tensor.
        grid_thw = torch.tensor(
            [grid_t, grid_h, grid_w], dtype=torch.long, device=img.device
        ).expand(B, 3)

        return flatten_patches, grid_thw


    # ---------------- Public API ----------------
    @torch.no_grad()
    def encode_image(
        self, img: torch.Tensor, eq_scale_factor: float=1.0, is_eq_prior: bool=False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Preprocess the images.
        batch = img.shape[0]
        img, grid_thw = self._preprocess_image(img, eq_scale_factor, is_eq_prior)

        # Register hooks to save outputs from specified layers.
        feat_dict = {}
        hooks = []
        n_blocks = len(self.vision_model.blocks)

        # 0 for patch embedding output.
        if 0 in self.patch_from_layers:
            hooks.append(
                self.vision_model.patch_embed.register_forward_hook(
                    partial(_save_hook, 0, feat_dict)
                )
            )

        # 1 to N or -2 to -N for attention layers.
        for idx in self.patch_from_layers:
            if idx == 0 or idx == -1:
                continue

            if idx > 0:                         # 1 → block0 , 2 → block1 ...
                real_idx = idx - 1
            else:                               # -2 → last block, -3 → second-last ...
                real_idx = n_blocks + idx + 1

            if not (0 <= real_idx < n_blocks):
                raise ValueError(
                    f"Layer index {idx} (mapped to block {real_idx}) "
                    f"is out of range for {n_blocks} blocks."
                )

            hooks.append(
                self.vision_model.blocks[real_idx].register_forward_hook(
                    partial(_save_hook, idx, feat_dict)
                )
            )

        # Forward pass through the vision model.
        with torch.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
            patch_tokens = self.vision_model(img, grid_thw=grid_thw) # (B, N_patches, D)    
            patch_tokens = patch_tokens.reshape(batch, -1, patch_tokens.shape[-1])
            pooled_output = patch_tokens.mean(dim=1) # no cls token in qwen, use mean pooling
        
        # -1 for merger output.
        if -1 in self.patch_from_layers:
            feat_dict[-1] = patch_tokens # after final layernorm

        # Remove hooks.
        for h in hooks:
            h.remove()

        patch_features = [feat_dict[i].float() for i in self.patch_from_layers]
        return patch_features, pooled_output

    @torch.no_grad()
    def encode_text(
        self, text: List[str]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_text:
            return None, None, None
        
        # Important: truncate texts to max length of 512 tokens, and pad them to the longest sequence.
        tok = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tok = tok.to(next(self.text_model.parameters()).device)

        # Please mention the position_ids here! This is not tested well.
        with torch.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
            outputs = self.text_model(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                return_dict=True,
            )
        seq_tokens = outputs.last_hidden_state
        pooled_output = seq_tokens[:, 0]
        mask = tok.attention_mask
        return seq_tokens.float(), pooled_output.float(), mask.bool()
