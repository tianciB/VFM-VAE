# ------------------------------------------------------------------------------
# Dataclasses for network outputs
# ------------------------------------------------------------------------------


import torch

from typing import Optional, List
from dataclasses import dataclass


# -------------------------------------------------------------------
# Output structure for LDMAdapter.encode()
# -------------------------------------------------------------------
@dataclass
class EncodeOutput:
    """
    Wrapper for the outputs of LDMAdapter.encode():

    Attributes
    ----------
    z : torch.Tensor
        Latent feature map with shape [B, C_latent, H, W].

    vf_loss : Optional[torch.Tensor]
        Vision foudation (VF) loss, if applicable.

    vf_last_layer: Optional[torch.Tensor]
        Last layer weight from the VF model, if applicable.
        
    kl_loss : Optional[torch.Tensor]
        KL-divergence loss (only populated in *continuous* mode).

    vq_loss : Optional[torch.Tensor]
        Vector-quantization loss (only populated in *discrete* mode).

    entropy_loss : Optional[torch.Tensor]
        Auxiliary entropy loss returned by the VQ tokenizer (discrete mode).

    usages : Optional[torch.Tensor]
        Code-book usage statistics from the VQ tokenizer (discrete mode).
    """
    z: torch.Tensor
    vf_loss:            Optional[torch.Tensor] = None
    vf_last_layer:      Optional[torch.Tensor] = None
    kl_loss:            Optional[torch.Tensor] = None
    vq_loss:            Optional[torch.Tensor] = None
    entropy_loss:       Optional[torch.Tensor] = None
    codebook_usages:    Optional[torch.Tensor] = None


# -------------------------------------------------------------------
# Output structure for Generator.forward()
# -------------------------------------------------------------------
@dataclass
class GeneratorForwardOutput:
    """
    Wrapper for the outputs of Generator.encode():

    Attributes
    ----------
    gen_img : torch.Tensor
        The main output image, shape ``[B, img_channels, img_resolution, img_resolution]``.

    gen_multiscale_imgs : Optional[List[torch.Tensor]]
        Additional images at intermediate resolutions when
        ``use_multiscale_output`` is ``True``; otherwise ``None``.
    
    vf_loss : Optional[torch.Tensor]
        Vision foundation (VF) loss, if applicable.

    vf_last_layer: Optional[torch.Tensor]
        Last layer weight from the VF model, if applicable.

    kl_loss / vq_loss / entropy_loss / usages :
        Same semantics as in *EncodeOutput*; these are forwarded so that
        later stages (e.g., loss computation) can access them directly.

    eq_scale_factor : float
        Scaling factor applied during equivariance regularization
        (1.0 means *no* scaling).

    eq_angle_factor : int
        Number of 90-degree rotations applied during equivariance regularization.  
        Values are 0 → 0°, 1 → 90°, 2 → 180°, 3 → 270°.

    global_text_tokens : Optional[torch.Tensor]
        Global text embeddings produced by the VFM text encoder when
        ``label_type`` is ``'text'`` or ``'cls2text'``.  ``None`` otherwise.
    """
    gen_img:                torch.Tensor
    gen_multiscale_imgs:    List[torch.Tensor]
    vf_loss:                Optional[torch.Tensor] = None
    vf_last_layer:          Optional[torch.Tensor] = None
    kl_loss:                Optional[torch.Tensor] = None
    vq_loss:                Optional[torch.Tensor] = None
    entropy_loss:           Optional[torch.Tensor] = None
    codebook_usages:        Optional[torch.Tensor] = None
    eq_scale_factor:        float = 1.0
    eq_angle_factor:        int = 0
    global_text_tokens:     Optional[torch.Tensor] = None


# -------------------------------------------------------------------
# Output structure for Discriminator.forward()
# -------------------------------------------------------------------
@dataclass
class DiscriminatorForwardOutput:
    """
    Unified output wrapper for multi-branch discriminators 
    (StyleGAN-T + PatchGAN).

    Attributes
    ----------
    stylegan_t_logits : torch.Tensor
        Logits returned by the StyleGAN-T ViT-based discriminator head.
        Shape: [B, N_heads], where N_heads = number of ViT hook layers.

    patchgan_logits : Optional[Union[List[torch.Tensor], List[List[torch.Tensor]]]]
        Outputs from the PatchGAN (multi-scale) discriminator.
        - If get_interm_feat = False: List of logits per scale, each [B, 1, H_i, W_i].
        - If get_interm_feat = True: List of List of activations per scale and layer:
            [[layer0_out, layer1_out, ..., logits], ..., scale_N].

        Can be None if PatchGAN branch is disabled.
    """
    stylegan_t_logits: torch.Tensor
    patchgan_logits: Optional[list[list[torch.Tensor]]] = None
