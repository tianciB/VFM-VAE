# ------------------------------------------------------------------------------
# EVA-CLIP Encoder Wrapper
# ------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import timm
from typing import Tuple, List, Optional

__all__ = ["EVAEncoder"]


class EVAEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "eva02_large_patch14_clip_336",
        scale_factor: float = 1.0,
        patch_from_layers: List[int] = [-1],
        amp_dtype: torch.dtype = torch.bfloat16,
        amp_enabled: bool = True,
        dynamic_img_pad: bool = False,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.patch_from_layers = patch_from_layers
        self.amp_dtype = amp_dtype
        self.amp_enabled = amp_enabled
        
        self.dynamic_img_pad = dynamic_img_pad
        self.use_cls_token = use_cls_token
        self.patch_size = 14 if "patch14" in model_name.lower() else 16

        # CLIP normalization.
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std  = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("_mean", torch.tensor(clip_mean).view(1,3,1,1), persistent=False)
        self.register_buffer("_std",  torch.tensor(clip_std ).view(1,3,1,1), persistent=False)

        # Build timm vision model with pretrained weights.
        self.vision_model = timm.create_model(
            self.model_name,
            pretrained=True,
            dynamic_img_size=True,
            dynamic_img_pad=self.dynamic_img_pad,
        )
        self.vision_model.eval().requires_grad_(False)

        print(
            f"✅ EVAEncoder ready\n"
            f"  model                   : {self.model_name}\n"
            f"  text_enabled            : False\n"
            f"  scale_factor            : {self.scale_factor}\n"
            f"  patch_from_layers       : {self.patch_from_layers}\n"
            f"  patch_size              : {self.patch_size}\n"
            f"  amp_dtype               : {self.amp_dtype}\n"
            f"  amp_enabled             : {self.amp_enabled}\n"
            f"  dynamic_img_pad         : {self.dynamic_img_pad}\n"
            f"  global embedding        : {'CLS' if self.use_cls_token else 'mean(patches)'}\n"
        )

    # ---------------- utils ----------------

    def _preprocess_image(self, img: torch.Tensor, eq_scale_factor: float, is_eq_prior: bool) -> torch.Tensor:
        """Preprocess input image with optional scaling for equivariance regularization."""
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        # First interpolation: downscale for equivariance/prior experiments.
        if is_eq_prior and eq_scale_factor < 1.0:
            img = F.interpolate(img, scale_factor=eq_scale_factor, mode="bilinear", align_corners=False)

        # Second interpolation: force final resolution.
        if self.scale_factor != 1.0:
            img = F.interpolate(img, scale_factor=self.scale_factor, mode="bilinear", align_corners=False, 
                                antialias=(self.scale_factor < 1.0))

        img = normalize(img, mean=self._mean.to(img.device), std=self._std.to(img.device))
        return img

    # ---------------- API ----------------

    @torch.no_grad()
    def encode_image(
        self, img: torch.Tensor, eq_scale_factor: float = 1.0, is_eq_prior: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Encode an image and return patch features + pooled embedding.
        """
        vm = self.vision_model
        img = self._preprocess_image(img, eq_scale_factor, is_eq_prior)

        want_patchify = any(i == 0 for i in self.patch_from_layers)
        want_blocks   = sorted({i for i in self.patch_from_layers if isinstance(i, int) and i > 0})

        with torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
            x = vm.patch_embed(img)
            x, rot_pos_embed = vm._pos_embed(x)
            if want_patchify:
                patchify_tokens = x
            x = vm.norm_pre(x)

            cached = {}
            for i, blk in enumerate(vm.blocks, start=1):
                x = blk(x, rope=rot_pos_embed)
                if i in want_blocks:
                    cached[i] = x.detach()

        num_prefix = getattr(vm, "num_prefix_tokens", 1)
        patch_features: List[torch.Tensor] = []
        last_tokens: torch.Tensor = x

        for idx in self.patch_from_layers:
            if idx == 0:
                feats = patchify_tokens[:, num_prefix:, :]
            elif idx == -1:
                feats = last_tokens[:, num_prefix:, :]
            else:
                feats = cached[idx][:, num_prefix:, :]
            patch_features.append(feats.float())

        if self.use_cls_token and num_prefix >= 1:
            pooled = last_tokens[:, 0, :].float()
        else:
            pooled = last_tokens[:, num_prefix:, :].mean(dim=1).float()

        return patch_features, pooled

    @torch.no_grad()
    def encode_text(
        self, text: List[str]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Vision-only encoder; text path is not provided here."""
        return None, None, None
