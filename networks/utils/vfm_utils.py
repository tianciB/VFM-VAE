# ------------------------------------------------------------------------------
# VFM - Vision Foundation Model - Utilities
# ------------------------------------------------------------------------------


import torch
import torch.nn as nn

from typing import Optional, Tuple, List
from .vfms.qwen_utils import QwenVisionEncoder
from .vfms.siglip2_utils import SigLIP2Encoder
from .vfms.dinov2_utils import DINOv2Encoder
from .vfms.mae_utils import MAEEncoder
from .vfms.eva_utils import EVAEncoder


VFM2INTERPOLATION = {
    'siglip':   'bilinear',
    'qwen':     'bicubic',
    'dino':     'bicubic',
    'mae':      'bilinear',
    'eva':      'bicubic',
}


class VFMEncoder(nn.Module):
    """
    Vision-Feature-Model encoder that automatically dispatches to the
    correct backbone based on `model_name`.

    Index mapping (shared by both back-ends)
    ----------------------------------------
      0   : output right after `patch_embed`
      1…N : output of the first, second … N-th Transformer block
     -1   : final sequence (Qwen = merger+LN, SigLIP-2 = post-LN)
     -2   : last Transformer block
     -3   : second-last Transformer block
       …

    Parameters
    ----------
    model_name : str
        HF hub name or local path, e.g. "Qwen/Qwen2.5-VL-7B" or
        "google/siglip2-large-patch16-512".
    conditional / label_type : forwarded to the sub-encoder.
    patch_from_layers : List[int]
        Indices specified with the mapping above.
    amp_dtype / amp_enabled : mixed-precision settings.
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
    ) -> None:
        super().__init__()
        model_name_lower = model_name.lower()

        if "qwen" in model_name_lower:
            self.encoder = QwenVisionEncoder(
                model_name=model_name,
                conditional=conditional,
                label_type=label_type,
                scale_factor=scale_factor,
                patch_from_layers=patch_from_layers,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled
            )
        
        elif "siglip2" in model_name_lower:
            self.encoder = SigLIP2Encoder(
                model_name=model_name,
                conditional=conditional,
                label_type=label_type,
                scale_factor=scale_factor,
                patch_from_layers=patch_from_layers,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled
            )

        elif "dinov2" in model_name_lower:
            self.encoder = DINOv2Encoder(
                model_name=model_name,
                patch_from_layers=patch_from_layers,
                scale_factor=scale_factor,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled
            )
        elif "mae" in model_name_lower:
            self.encoder = MAEEncoder(
                model_name=model_name,
                patch_from_layers=patch_from_layers,
                scale_factor=scale_factor,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled
            )
        elif "eva" in model_name_lower:
            self.encoder = EVAEncoder(
                model_name=model_name,
                scale_factor=scale_factor,
                patch_from_layers=patch_from_layers,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled
            )

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size

    def encode_image(
        self, img: torch.Tensor, eq_scale_factor: float=1.0, is_eq_prior: bool=False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.encoder.encode_image(img, eq_scale_factor, is_eq_prior)

    def encode_text(
        self, text: List[str]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.encoder.encode_text(text)
