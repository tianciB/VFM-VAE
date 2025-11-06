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


"""Loss function."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from torch_utils import training_stats
from torch_utils import distributed as dist
from torch_utils.ops import upfirdn2d
from training.lpips import LPIPS
from typing import Tuple, Union, Optional
from networks.utils.vfms.clip_utils import CLIP
from networks.utils.vfm_utils import VFM2INTERPOLATION
from networks.utils.dataclasses import GeneratorForwardOutput, DiscriminatorForwardOutput
from torchvision.transforms import RandomCrop
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

SAFE_MARK, UNSAFE_MARK = 1, 0


class ImageTransform(nn.Module):
    def __init__(self, apply_equivariance: bool = False, interpolation: str = 'bilinear') -> None:
        super().__init__()
        self.apply_equivariance = apply_equivariance
        self.interpolation = interpolation

    def _interpolate(self, img: torch.Tensor, *, size=None, scale_factor=None) -> torch.Tensor:
        kwargs = dict(mode=self.interpolation)

        # Bilinear and bicubic interpolation require align_corners=False.
        if self.interpolation in ["bilinear", "bicubic"]:
            kwargs["align_corners"] = False

        # Antialias is only needed when downsampling with bilinear or bicubic.
        need_antialias = (
            self.interpolation in ["bilinear", "bicubic"]
            and ((scale_factor and scale_factor < 1.0) or (size and size < img.shape[-1]))
        )
        if need_antialias:
            kwargs["antialias"] = True

        return F.interpolate(img, size=size, scale_factor=scale_factor, **kwargs)

    def forward(self, img: torch.Tensor, eq_scale_factor: float, eq_angle_factor: int) -> torch.Tensor:
        
        if self.apply_equivariance:
            if eq_scale_factor != 1.0:
                img = self._interpolate(img, scale_factor=eq_scale_factor)
            if eq_angle_factor % 4 != 0:
                img = torch.rot90(img, k=eq_angle_factor, dims=[-1, -2])
        
        return img

    def multiscale_forward(self, img: torch.Tensor, targets: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._interpolate(img, size=t.shape[-1]) for t in targets]


class TotalLoss:
    def __init__(
        self,
        device: torch.device,
        G: nn.Module,
        D: nn.Module,
        vfm_name: str,                                      # Model name for VFM.
        resume_kimg: int,                                   # Resume training from this kimg.
        use_equivariance_regularization: bool,              # Use equivariance regularization. 
        blur_init_sigma: int = 2,                           # Blur sigma for generator.
        blur_fade_kimg: int = 0,
        l1_pixel_loss_weight: float = 1.0,                  # Pixel loss L1 for generator.
        l2_pixel_loss_weight: float = 0.0,                  # Pixel loss L2 for generator.
        perceptual_loss_weight: float = 10.0,               # Perceptual loss for generator.
        ssim_loss_weight: float = 0.0,                      # SSIM loss for generator.
        multiscale_pixel_loss_weights: list[float] = [],    # Multiscale pixel loss for generator.
        multiscale_block_indices: list[int] = [],                   
        multiscale_pixel_loss_start_kimg: int = 0,
        multiscale_pixel_loss_end_kimg: int = 2000,
        vf_loss_weight: float = 0.0,                        # Vision foundation loss for generator. 
        use_adaptive_vf_loss: bool = False,                 # Use adaptive VFM loss.
        clip_loss_weight: float = 0.0,                      # CLIP loss for generator.     
        clip_loss_start_kimg: int = 0,
        matching_aware_loss_weight: float = 0.0,            # Matching-aware loss for discriminator.
        matching_aware_loss_start_kimg: int = 0,            
        compression_mode: str = 'continuous',               # Compression mode: 'continuous' or 'discrete'.
        kl_loss_weight: float = 1e-6,                       # KL loss for continuous compression mode.
        entropy_loss_weight: float = 0.0,                   # Entropy loss for discrete compression mode.
        vq_loss_weight: float = 1.0,                        # VQ loss for discrete compression mode.
        stylegan_t_discriminator_loss_weight: float = 1.0,  # StyleGAN-T discriminator loss weight.
        patchgan_discriminator_loss_weight: float = 0.0,    # PatchGAN discriminator loss weight.
        patchgan_discriminator_loss_type: str = 'mse',      # PatchGAN discriminator loss type, 'mse', 'bce' or 'hinge'.
        feature_matching_loss_weight: float = 1.0,          # Feature matching loss weight for PatchGAN.
        use_stylegan_t_disc_warmup: bool = False,           # Use StyleGAN discriminator warm-up.
        use_patchgan_disc_warmup: bool = False,             # Use PatchGAN discriminator warm-up.
        total_kimg: int = 0,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D

        # Align Interpolation with vision foundation model.
        self.vfm_name = vfm_name.lower()
        for name in VFM2INTERPOLATION.keys():
            if name in self.vfm_name:
                self.interpolation = VFM2INTERPOLATION.get(name, 'bilinear')
                break

        # Safe loss checking.
        self.resume_kimg = resume_kimg
        self.prev_loss_dict = None
        self.safe_loss_checking_start_nimg = 50_000

        # Image transform.
        self.img_transform = ImageTransform(
            apply_equivariance=use_equivariance_regularization,
            interpolation=self.interpolation,
        )
        self.interpolation = self.img_transform.interpolation

        # Blur sigma.
        self.blur_init_sigma = blur_init_sigma
        self.blur_curr_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        
        # Pixel loss.
        self.l1_pixel_loss_weight = l1_pixel_loss_weight
        self.l2_pixel_loss_weight = l2_pixel_loss_weight

        # Perceptual loss.
        self.perceptual_loss_weight = perceptual_loss_weight
        self.perceptual_module = LPIPS().eval().to(self.device).requires_grad_(False) if perceptual_loss_weight > 0 else None

        # SSIM loss.
        self.ssim_loss_weight = ssim_loss_weight
        self.ssim_module = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device) if ssim_loss_weight > 0 else None

        # Multiscale pixel loss.
        assert len(multiscale_pixel_loss_weights) == len(multiscale_block_indices), "Multiscale pixel loss weights must match the number of multiscale resolutions."
        self.multiscale_pixel_loss_weights = multiscale_pixel_loss_weights
        self.multiscale_block_indices = multiscale_block_indices
        self.multiscale_pixel_loss_start_kimg = multiscale_pixel_loss_start_kimg
        self.multiscale_pixel_loss_end_kimg = multiscale_pixel_loss_end_kimg

        # Vision foundation model (VFM) loss.
        self.vf_loss_weight = vf_loss_weight
        self.use_adaptive_vf_loss = use_adaptive_vf_loss

        # ClIP loss.
        self.clip_loss_weight = clip_loss_weight
        self.clip_loss_start_kimg = clip_loss_start_kimg
        if self.clip_loss_weight > 0:
            self.clip = CLIP().eval().to(self.device).requires_grad_(False)

        # Matching-aware loss.
        self.matching_aware_loss_weight = matching_aware_loss_weight
        self.matching_aware_loss_start_kimg = matching_aware_loss_start_kimg

        # LDM adaption.
        self.compression_mode = compression_mode
        self.kl_loss_weight = kl_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.vq_loss_weight = vq_loss_weight

        # Discriminator settings.
        self.patchgan_discriminator_loss_type = patchgan_discriminator_loss_type
        self.stylegan_t_discriminator_loss_weight = stylegan_t_discriminator_loss_weight
        self.patchgan_discriminator_loss_weight = patchgan_discriminator_loss_weight
        self.use_stylegan_t_disc_warmup = use_stylegan_t_disc_warmup
        self.use_patchgan_disc_warmup = use_patchgan_disc_warmup
        dist.print0(f"[Manual Training] StyleGAN-T Discriminator Warm-up: {self.use_stylegan_t_disc_warmup}, PatchGAN Discriminator Warm-up: {self.use_patchgan_disc_warmup}")

        # PatchGAN Discriminator's feature matching loss.
        self.feature_matching_loss_weight = feature_matching_loss_weight

        # Discriminator warmup settings.
        self._stylegan_t_on = True if (self.stylegan_t_discriminator_loss_weight > 0 and not self.use_stylegan_t_disc_warmup) else False
        self._patchgan_on = True if (self.patchgan_discriminator_loss_weight > 0 and not self.use_patchgan_disc_warmup) else False

        self._perceptual_loss_on         = True if self.perceptual_loss_weight > 0 else False
        self._ssim_loss_on               = True if self.ssim_loss_weight > 0 else False
        self._multiscale_pixel_loss_on   = True if sum(self.multiscale_pixel_loss_weights) > 0 else False
        self._pixel_loss_on              = (self.l1_pixel_loss_weight > 0 or self.l2_pixel_loss_weight > 0)

        self._window_size                = 50 * 2                            # two windows of size 50
        
        # Pixel loss trigger is discarded because it is not used in the current implementation.
        # Keep it for the maunual StyleGAN-T Discriminator warm-up.
        self._pixel_loss_window_type     = 'l1' if self.l1_pixel_loss_weight > 0 else 'l2'
        self._pixel_window               = deque(maxlen=self._window_size)   # window for StyleGAN-T pixel loss
        self._pixel_thresh               = 0.1                               # threshold for StyleGAN-T pixel loss
        self._pixel_diff_thresh          = 0.01                              # threshold for difference between two windows of StyleGAN-T pixel loss
        self._pixel_patience             = 10                                # patience for StyleGAN-T pixel loss stability
        self._pixel_cn                   = 0                                 # count for reaching the threshold

        self._d_window                   = deque(maxlen=self._window_size)   # window for StyleGAN-T discriminator loss
        self._d_thresh                   = 0.1                               # threshold for StyleGAN-T discriminator loss
        self._d_diff_thresh              = 0.05                              # threshold for difference between two windows of StyleGAN-T discriminator loss
        self._d_patience                 = 10                                # patience for StyleGAN-T discriminator loss stability
        self._d_cn                       = 0                                 # count for reaching the threshold

        self._freeze_done                = False                             # flag to indicate if freeze32 is done
        self._off_done                   = False                             # flag to indicate if reconstruction losses are turned off

        # Total training steps.
        self.total_kimg = total_kimg

    @staticmethod
    def blur(img: torch.Tensor, blur_sigma: float) -> torch.Tensor:
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        return img

    def set_blur_sigma(self, cur_nimg: int) -> None:
        if self.blur_fade_kimg > 1:
            self.blur_curr_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
        else:
            self.blur_curr_sigma = 0

    def run_G(self, z: torch.Tensor, c: Union[list[str], torch.Tensor]) -> Tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, float, int, Optional[torch.Tensor]]:
        generator_output: GeneratorForwardOutput = self.G(z, c)
        return generator_output

    def run_D(self, img: torch.Tensor, c_enc: Union[list[str], torch.Tensor]) -> torch.Tensor:
        img = self.blur(img, self.blur_curr_sigma)
        discriminator_output: DiscriminatorForwardOutput = self.D(img, c_enc)
        return discriminator_output
    
    def calculate_pixel_loss(self, real, gen, type='l1'):
        if type == 'l1':
            return F.l1_loss(real, gen).mean()
        elif type == 'l2':
            return F.mse_loss(real, gen).mean()

    def calculate_perceptual_loss(self, real_img: torch.Tensor, gen_img: torch.Tensor) -> torch.Tensor:
        return self.perceptual_module(real_img, gen_img).mean()

    def calculate_ssim_loss(self, real_img, gen_img):
        gen_img_clipped = gen_img.clamp(-1, 1)
        real_img_clipped = real_img.clamp(-1, 1)
        return 1.0 - self.ssim_module(gen_img_clipped, real_img_clipped)

    def calculate_cur_vf_loss_weight(self, rec_loss: torch.Tensor, vf_loss: torch.Tensor, last_layer: torch.Tensor) -> torch.Tensor:
        if self.use_adaptive_vf_loss:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]
            cur_vf_loss_weight = torch.norm(rec_grads) / (torch.norm(vf_grads) + 1e-4)
            cur_vf_loss_weight = torch.clamp(cur_vf_loss_weight, 0.0, 1e8).detach()
            cur_vf_loss_weight = cur_vf_loss_weight * self.vf_loss_weight
            return cur_vf_loss_weight
        else:
            return self.vf_loss_weight

    @staticmethod
    def calculate_matching_aware_loss(real_logits: torch.Tensor, gen_logits: torch.Tensor) -> torch.Tensor:
        return (F.softplus(real_logits) + F.softplus(gen_logits)).mean()

    @staticmethod
    def calculate_spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x * y).sum(-1).arccos().pow(2)

    def calculate_stylegan_t_disc_loss(self, logits: torch.Tensor, type: str) -> torch.Tensor:
        if type == 'real':
            return F.relu(1.0 - logits).mean()
        elif type == 'fake':
            return F.relu(1.0 + logits).mean()

    def calculate_patchgan_disc_loss(self, logits: list[list[torch.Tensor]], type: str) -> torch.Tensor:
        assert type in ['real', 'fake'], f"Invalid type: {type}"
        if len(logits) == 0:  # avoid empty logits
            return torch.tensor(0.0, device=self.device)

        is_real = (type == 'real')
        loss = 0.
        for scale_output in logits:
            pred = scale_output[-1]

            if self.patchgan_discriminator_loss_type == 'bce':
                target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
                loss += F.binary_cross_entropy_with_logits(pred, target)

            elif self.patchgan_discriminator_loss_type == 'mse':
                target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
                loss += F.mse_loss(pred, target)

            elif self.patchgan_discriminator_loss_type == 'hinge':
                # Hinge loss does not use target tensors
                if is_real:
                    loss += F.relu(1.0 - pred).mean()
                else:
                    loss += F.relu(1.0 + pred).mean()

            else:
                raise ValueError(f"Unsupported PatchGAN loss type: {self.patchgan_discriminator_loss_type}")

        return loss / len(logits)  # Average loss across scales

    def calculate_patchgan_gen_loss(self, logits: list[list[torch.Tensor]]) -> torch.Tensor:
        """Generator-side PatchGAN loss.
        BCE/MSE: target=1; Hinge: -mean(D(fake))."""
        if len(logits) == 0:
            return torch.tensor(0.0, device=self.device)

        loss = 0.
        for scale_output in logits:
            pred = scale_output[-1]  # [B,1,H,W]

            if self.patchgan_discriminator_loss_type == 'bce':
                target = torch.ones_like(pred)
                loss += F.binary_cross_entropy_with_logits(pred, target)

            elif self.patchgan_discriminator_loss_type == 'mse':
                target = torch.ones_like(pred)
                loss += F.mse_loss(pred, target)

            elif self.patchgan_discriminator_loss_type == 'hinge':
                loss += (-pred).mean()

            else:
                raise ValueError(f"Unsupported PatchGAN loss type: {self.patchgan_discriminator_loss_type}")

        return loss / len(logits)

    def calculate_feature_matching_loss(self, real_features: list[list[torch.Tensor]], fake_features: list[list[torch.Tensor]]) -> torch.Tensor:
        loss = 0.
        D_weights = 1.0 / len(real_features)
        for rf, ff in zip(real_features, fake_features):
            feat_w = 4.0 / max(len(rf) - 1, 1) # 4.0 is an experimental constant in Pix2PixHD
            for r, f in zip(rf[:-1], ff[:-1]):
                loss += D_weights * feat_w * F.l1_loss(f, r.detach())
        return loss
    
    def _safe_resize(self, img: torch.Tensor, size: int) -> torch.Tensor:
        kwargs = dict(mode=self.interpolation)
        if self.interpolation in ["bilinear", "bicubic"]:
            kwargs["align_corners"] = False
            if img.size(-1) > size:
                kwargs["antialias"] = True
        return F.interpolate(img, size, **kwargs)

    def _off_reconstruction_and_quantization_losses(self) -> None:
        """Disable reconstruction losses."""
        self._perceptual_loss_on            = False
        self._ssim_loss_on                  = False
        self._multiscale_pixel_loss_on      = False
        self._pixel_loss_on                 = False
        
        self.perceptual_loss_weight         = 0.0
        self.ssim_loss_weight               = 0.0
        self.multiscale_pixel_loss_weights  = [0.0] * len(self.multiscale_pixel_loss_weights)
        self.l1_pixel_loss_weight           = 0.0
        self.l2_pixel_loss_weight           = 0.0

        self.kl_loss_weight                 = 0.0
        self.vq_loss_weight                 = 0.0
        self.vf_loss_weight                 = 0.0

        dist.print0("[Reconstruction & Quantization Losses] Off perceptual, SSIM, multiscale pixel, pixel, KL, VQ, and VF losses.")

    def _update_phase(
        self,
        cur_nimg: int,
        pixel_loss_now: float,
        d_now: float
    ) -> None:
        """
        Update training phase based on stability of losses.

        Logic:
        - Maintain a sliding window of recent pixel and discriminator losses.
        - Split the window into two halves.
        - Only trigger phase judgment if the window is full and low enough.
        - Compare mean of each half; if difference is small for N steps, trigger phase transition.
        """
        import numpy as np
        from collections import deque

        cur_kimg = cur_nimg // 1000
        is_primary = (not dist.is_initialized()) or (dist.get_rank() == 0)
        need_freeze32 = False

        if is_primary:
            # --- UPDATE LOSS WINDOWS ---
            self._d_window.append(d_now)        # append current discriminator loss to the window
            d_mean = np.mean(self._d_window)    # compute mean of the current window

            if not self._stylegan_t_on and self.use_stylegan_t_disc_warmup:
                self._pixel_window.append(pixel_loss_now) # append current pixel loss to the window
                pixel_mean = np.mean(self._pixel_window)  # compute mean of the current pixel loss window

            # --- MANUAL WARM-UP ---
            if not self._stylegan_t_on and self.use_stylegan_t_disc_warmup:
                if len(self._pixel_window) == self._pixel_window.maxlen and pixel_mean < self._pixel_thresh:
                    values = list(self._pixel_window)
                    half = len(values) // 2
                    early_half = values[:half]
                    late_half = values[half:]

                    mean_early = np.mean(early_half)
                    mean_late = np.mean(late_half)
                    diff = abs(mean_late - mean_early)

                    if diff < self._pixel_diff_thresh:
                        self._pixel_cn += 1
                        dist.print0(f"[WARMUP-StyleGAN-T] Δmean={diff:.2e} cnt={self._pixel_cn}/{self._pixel_patience}")
                    elif self._pixel_cn > 0:
                        dist.print0(f"[WARMUP-StyleGAN-T-reset] Δmean={diff:.2e} prev_cnt={self._pixel_cn}")
                        self._pixel_cn = 0

                    self._pixel_window = deque(late_half, maxlen=self._pixel_window.maxlen)

                    if self._pixel_cn >= self._pixel_patience:
                        self._stylegan_t_on = True
                        dist.print0(f"[WARM-UP-StyleGAN-T] enabled @ {cur_kimg} kimg")

            if not self._patchgan_on and self.use_patchgan_disc_warmup:
                if len(self._d_window) == self._d_window.maxlen and d_mean < self._d_thresh:
                    values = list(self._d_window)
                    half = len(values) // 2
                    early_half = values[:half]
                    late_half = values[half:]

                    mean_early = np.mean(early_half)
                    mean_late = np.mean(late_half)
                    diff = abs(mean_late - mean_early)

                    if diff < self._d_diff_thresh:
                        self._d_cn += 1
                        dist.print0(f"[WARMUP-PatchGAN] Δmean={diff:.2e} cnt={self._d_cn}/{self._d_patience}")
                    elif self._d_cn > 0:
                        dist.print0(f"[WARMUP-PatchGAN-reset] Δmean={diff:.2e} prev_cnt={self._d_cn}")
                        self._d_cn = 0

                    self._d_window = deque(late_half, maxlen=self._d_window.maxlen)

                    if self._d_cn >= self._d_patience:
                        need_freeze32 = True
                        self._patchgan_on = True
                        dist.print0(f"[WARM-UP-PatchGAN] enabled @ {cur_kimg} kimg")

        # Broadcast updated state
        flags = torch.tensor([
            int(self._stylegan_t_on),
            int(self._patchgan_on),
            int(self._perceptual_loss_on),
            int(self._pixel_loss_on),
            int(self._ssim_loss_on),
            int(self._multiscale_pixel_loss_on),
            int(need_freeze32)
        ], dtype=torch.int, device=self.device)

        if dist.is_initialized():
            torch.distributed.broadcast(flags, src=0)

        (
        self._stylegan_t_on,
        self._patchgan_on,
        self._perceptual_loss_on,
        self._pixel_loss_on,
        self._ssim_loss_on,
        self._multiscale_pixel_loss_on,
        need_freeze32
        ) = [bool(x.item()) if i else int(x.item()) for i, x in enumerate(flags)]

        if need_freeze32 and not getattr(self, "_freeze_done", False):
            self.G.set_train_mode('freeze32')
            self._freeze_done = True

        if self._patchgan_on and not getattr(self, "_off_done", False):
            self._off_reconstruction_and_quantization_losses()
            self._off_done = True

    @staticmethod
    def _agg_patchgan_per_scale(
        logits: list[Union[torch.Tensor, list[torch.Tensor]]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        For each scale in PatchGAN, compute:
            - fake_score: scalar mean over batch
            - fake_sign:  scalar sign mean over batch

        Args:
            logits: List of scale outputs, each a Tensor or list of Tensors

        Returns:
            List of (fake_score, fake_sign) tuples, one per scale
        """
        if not logits:
            return []

        results = []
        for scale_output in logits:
            pred = scale_output[-1] if isinstance(scale_output, list) else scale_output  # [B, 1, H, W]
            pred_flat = pred.view(pred.size(0), -1)  # [B, H*W]
            scores = pred_flat.mean(dim=1)           # [B]
            score_mean = scores.mean()               # scalar
            sign_mean = scores.sign().mean()         # scalar
            results.append((score_mean, sign_mean))

        return results

    def accumulate_gradients(
        self,
        phase: str,                                     # 'G' for generator, 'D' for discriminator
        real_img: torch.Tensor,                         # shape: (batch_size, c_dim, h, w)
        real_c: Union[list[str], torch.Tensor],         # shape: (batch_size * str) or (batch_size, c_dim)
        cur_nimg: int                                   # current training step
    ) -> None:
        """
        Gradient accumulation logic for generator and discriminator updates.

        --- Value Range Conventions ---
        • real_img (input GT): assumed to be in [0, 1] range by default.
        • gen_img (generator output): always in [-1, 1] range.

        --- Value Range Expectations ---
        • SytleGAN-T discriminator: expects images in [-1, 1] range.
        • PatchGAN discriminator: expects images in [-1, 1] range.
        • LPIPS/perceptual loss: expects images in [-1, 1].
        • Pixel loss (L1): computed in [-1, 1].
        • CLIP or other visual foundation models (VFM): expect input images in [0, 1], resized to target resolution.

        --- Interpolation Convention ---
        • All spatial interpolations (resizing, cropping, multiscale) should be performed in [0, 1] range
        to ensure consistency, especially when using non-linear kernels like 'area'.

        This function ensures all value range conversions and interpolation orders
        are consistent with these expectations.
        """

        # Set blur sigma.
        self.set_blur_sigma(cur_nimg)

        # Check if real_c is a list of strings or a tensor.
        is_text_cond = isinstance(real_c, list) and isinstance(real_c[0], str)
        
        if phase == 'D':
            # Total discriminator loss for backpropagation.
            d_loss = torch.zeros([], device=self.device, requires_grad=True)

            # Minimize logits for generated images.
            with torch.no_grad():
                generator_output: GeneratorForwardOutput = self.run_G(real_img, real_c)
            gen_img = generator_output.gen_img.detach() # gen_img is already in [-1, 1] range
            gen_multiscale_imgs = [t.detach() for t in generator_output.gen_multiscale_imgs]
            eq_scale_factor = generator_output.eq_scale_factor
            eq_angle_factor = generator_output.eq_angle_factor
            real_c_enc = generator_output.global_text_tokens
            del generator_output

            gen_discrminator_output: DiscriminatorForwardOutput = self.run_D(gen_img, real_c_enc if is_text_cond else real_c)

            # Maximize logits for real images.
            real_img = self.img_transform(real_img, eq_scale_factor, eq_angle_factor)
            real_img = real_img * 2 - 1. # convert to [-1, 1] range
            real_discrminator_output: DiscriminatorForwardOutput = self.run_D(real_img, real_c_enc if is_text_cond else real_c)

            # StyleGAN-T discriminator loss.
            stylegan_t_gen_loss = torch.tensor(0.0, device=self.device)
            stylegan_t_real_loss = torch.tensor(0.0, device=self.device)
            stylegan_t_disc_loss = torch.tensor(0.0, device=self.device)
            if self._stylegan_t_on and self.stylegan_t_discriminator_loss_weight > 0:
                stylegan_t_gen_logits = gen_discrminator_output.stylegan_t_logits
                stylegan_t_real_logits = real_discrminator_output.stylegan_t_logits
                stylegan_t_gen_loss = self.calculate_stylegan_t_disc_loss(stylegan_t_gen_logits, type='fake')
                stylegan_t_real_loss = self.calculate_stylegan_t_disc_loss(stylegan_t_real_logits, type='real')
                stylegan_t_disc_loss = stylegan_t_gen_loss + stylegan_t_real_loss
            d_loss = d_loss + self.stylegan_t_discriminator_loss_weight * stylegan_t_disc_loss

            # PatchGAN discriminator loss.
            patchgan_gen_loss = torch.tensor(0.0, device=self.device)
            patchgan_real_loss = torch.tensor(0.0, device=self.device)
            patchgan_disc_loss = torch.tensor(0.0, device=self.device)
            if self._patchgan_on and self.patchgan_discriminator_loss_weight > 0:
                patchgan_gen_logits = gen_discrminator_output.patchgan_logits
                patchgan_real_logits = real_discrminator_output.patchgan_logits
                patchgan_gen_loss = self.calculate_patchgan_disc_loss(patchgan_gen_logits, type='fake')
                patchgan_real_loss = self.calculate_patchgan_disc_loss(patchgan_real_logits, type='real')
                patchgan_disc_loss = patchgan_gen_loss + patchgan_real_loss
            d_loss = d_loss + self.patchgan_discriminator_loss_weight * patchgan_disc_loss

            # Matching-aware loss (only for StyleGAN-T).
            matching_aware_loss = torch.tensor(0.0, device=self.device)
            if cur_nimg >= (self.matching_aware_loss_start_kimg * 1e3) and self.matching_aware_loss_weight > 0 and self._stylegan_t_on:
                # Text prompt can be shuffled easily.
                if is_text_cond:
                    shuffle_idx = torch.randperm(len(real_c))
                    real_c_shuffle = [real_c[i] for i in shuffle_idx]
                    gen_shuffled_discriminator_output: DiscriminatorForwardOutput = self.run_D(gen_img, real_c_shuffle)
                    real_shuffled_discriminator_output: DiscriminatorForwardOutput = self.run_D(real_img, real_c_shuffle)

                else:
                    shuffle_idx = torch.randperm(len(real_c_enc), device=real_c_enc.device)
                    real_c_enc_shuffle = real_c_enc[shuffle_idx]
                    gen_shuffled_discriminator_output: DiscriminatorForwardOutput = self.run_D(gen_img, real_c_enc_shuffle)
                    real_shuffled_discriminator_output: DiscriminatorForwardOutput = self.run_D(real_img, real_c_enc_shuffle)
                
                gen_shuffled_stylegan_t_logits = gen_shuffled_discriminator_output.stylegan_t_logits
                real_shuffled_stylegan_t_logits = real_shuffled_discriminator_output.stylegan_t_logits
                matching_aware_loss = self.calculate_matching_aware_loss(real_shuffled_stylegan_t_logits, gen_shuffled_stylegan_t_logits)
            d_loss = d_loss + self.matching_aware_loss_weight * matching_aware_loss

            # Safe loss checking.
            skip_d_loss_local = torch.tensor(0.0, device=self.device)

            d_loss_safe_mark_dict_local = {
                'stylegan_t_gen_loss'   : SAFE_MARK,
                'stylegan_t_real_loss'  : SAFE_MARK,
                'stylegan_t_disc_loss'  : SAFE_MARK,
                'patchgan_gen_loss'     : SAFE_MARK,
                'patchgan_real_loss'    : SAFE_MARK,
                'patchgan_disc_loss'    : SAFE_MARK,
                'matching_aware_loss'   : SAFE_MARK,
            }
            d_loss_names = list(d_loss_safe_mark_dict_local.keys())

            if cur_nimg > (self.resume_kimg * 1e3 + self.safe_loss_checking_start_nimg):
                if self._stylegan_t_on and self.stylegan_t_discriminator_loss_weight > 0:
                    if not torch.isfinite(stylegan_t_gen_loss) or stylegan_t_gen_loss.abs() > 1e4:
                        skip_d_loss_local.fill_(1.0)
                        d_loss_safe_mark_dict_local['stylegan_t_gen_loss'] = UNSAFE_MARK
                    if not torch.isfinite(stylegan_t_real_loss) or stylegan_t_real_loss.abs() > 1e4:
                        skip_d_loss_local.fill_(1.0)
                        d_loss_safe_mark_dict_local['stylegan_t_real_loss'] = UNSAFE_MARK
                    if not torch.isfinite(stylegan_t_disc_loss) or stylegan_t_disc_loss.abs() > 1e4:
                        skip_d_loss_local.fill_(1.0)
                        d_loss_safe_mark_dict_local['stylegan_t_disc_loss'] = UNSAFE_MARK

                if self._patchgan_on and self.patchgan_discriminator_loss_weight > 0:
                    if not torch.isfinite(patchgan_gen_loss) or patchgan_gen_loss.abs() > 1e4:
                        skip_d_loss_local.fill_(1.0)
                        d_loss_safe_mark_dict_local['patchgan_gen_loss'] = UNSAFE_MARK
                    if not torch.isfinite(patchgan_real_loss) or patchgan_real_loss.abs() > 1e4:
                        skip_d_loss_local.fill_(1.0)
                        d_loss_safe_mark_dict_local['patchgan_real_loss'] = UNSAFE_MARK
                    if not torch.isfinite(patchgan_disc_loss) or patchgan_disc_loss.abs() > 1e4:
                        skip_d_loss_local.fill_(1.0)
                        d_loss_safe_mark_dict_local['patchgan_disc_loss'] = UNSAFE_MARK

                if cur_nimg >= (self.matching_aware_loss_start_kimg * 1e3) and self.matching_aware_loss_weight > 0 and self._stylegan_t_on:
                    if not torch.isfinite(matching_aware_loss) or matching_aware_loss.abs() > 1e4:
                        skip_d_loss_local.fill_(1.0)
                        d_loss_safe_mark_dict_local['matching_aware_loss'] = UNSAFE_MARK

            # Gather across all ranks.
            d_loss_safe_mark_tensor_local = torch.tensor(list(d_loss_safe_mark_dict_local.values()), device=self.device, dtype=torch.int)

            if dist.is_initialized():
                torch.distributed.all_reduce(skip_d_loss_local, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(d_loss_safe_mark_tensor_local, op=torch.distributed.ReduceOp.MIN)

            skip_d_loss_synced = bool(skip_d_loss_local.item()) 
            d_loss_safe_mark_dict_synced = {
                k: bool(d_loss_safe_mark_tensor_local[i].item()) for i, k in enumerate(d_loss_names)
            }
            
            # Skip D backward if unsafe loss.
            if skip_d_loss_synced:
                d_loss = torch.nan_to_num(d_loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0 # to release the computation graph
                training_stats.report('Loss/D/skipped', 1.0)
            else:
                training_stats.report('Loss/D/skipped', 0.0)
            
            # Log safe marks.
            for k in d_loss_names:
                training_stats.report(f'Loss/D/is_safe/{k}', int(d_loss_safe_mark_dict_synced[k]))
                if not d_loss_safe_mark_dict_synced[k]:
                    dist.print0(f"[SafeLoss][D] Unsafe {k} at {cur_nimg//1000} kimg - skipping.")

            # Backward pass for discriminator.
            d_loss.backward()

            if skip_d_loss_synced:
                return

            # Collect stats.
            if self._stylegan_t_on and self.stylegan_t_discriminator_loss_weight > 0:
                training_stats.report('Loss/D/stylegan_t/fake_scores', stylegan_t_gen_logits)
                training_stats.report('Loss/D/stylegan_t/fake_signs', stylegan_t_gen_logits.sign())
                training_stats.report('Loss/D/stylegan_t/real_scores', stylegan_t_real_logits)
                training_stats.report('Loss/D/stylegan_t/real_signs', stylegan_t_real_logits.sign())
                training_stats.report('Loss/D/stylegan_t/gen_loss', stylegan_t_gen_loss)
                training_stats.report('Loss/D/stylegan_t/real_loss', stylegan_t_real_loss)
                training_stats.report('Loss/D/stylegan_t/loss', stylegan_t_disc_loss)

            if self._patchgan_on and self.patchgan_discriminator_loss_weight > 0:
                training_stats.report('Loss/D/patchgan/gen_loss', patchgan_gen_loss)
                training_stats.report('Loss/D/patchgan/real_loss', patchgan_real_loss)  
                training_stats.report('Loss/D/patchgan/loss', patchgan_disc_loss)
                for i, (s, sign) in enumerate(self._agg_patchgan_per_scale(patchgan_gen_logits)):
                    training_stats.report(f'Loss/D/patchgan/fake/scale{i}/fake_scores', s)
                    training_stats.report(f'Loss/D/patchgan/fake/scale{i}/fake_signs', sign)
                for i, (s, sign) in enumerate(self._agg_patchgan_per_scale(patchgan_real_logits)):
                    training_stats.report(f'Loss/D/patchgan/real/scale{i}/real_scores', s)
                    training_stats.report(f'Loss/D/patchgan/real/scale{i}/real_signs', sign)

            if cur_nimg >= (self.matching_aware_loss_start_kimg * 1e3) and self.matching_aware_loss_weight > 0 and self._stylegan_t_on:
                training_stats.report('Loss/D/matching_aware_loss', matching_aware_loss)

        elif phase == 'G':
            # Maximize logits for generated images.
            generator_output: GeneratorForwardOutput = self.run_G(real_img, real_c)
            gen_img = generator_output.gen_img # gen_img is already in [-1, 1] range
            gen_multiscale_imgs = generator_output.gen_multiscale_imgs
            eq_scale_factor = generator_output.eq_scale_factor
            eq_angle_factor = generator_output.eq_angle_factor
            real_c_enc = generator_output.global_text_tokens

            gen_discrminator_output: DiscriminatorForwardOutput = self.run_D(gen_img, real_c_enc if is_text_cond else real_c)
            
            # Discriminator logits.
            stylegan_t_gen_logits = None
            stylegan_t_gen_loss = torch.tensor(0.0, device=self.device)
            if self._stylegan_t_on and self.stylegan_t_discriminator_loss_weight > 0:
                stylegan_t_gen_logits = gen_discrminator_output.stylegan_t_logits
                stylegan_t_gen_loss = (-stylegan_t_gen_logits).mean()

            patchgan_gen_logits = None
            patchgan_gen_loss = torch.tensor(0.0, device=self.device)
            if self._patchgan_on and self.patchgan_discriminator_loss_weight > 0:
                patchgan_gen_logits = gen_discrminator_output.patchgan_logits
                patchgan_gen_loss = self.calculate_patchgan_gen_loss(patchgan_gen_logits)

            # Transform GT images with the same parameters as the generated images.
            real_img = self.img_transform(real_img, eq_scale_factor, eq_angle_factor)
            real_img_for_loss = real_img * 2 - 1. # convert to [-1, 1] range for main reconstruction loss

            # Feature matching loss for PatchGAN.
            feature_matching_loss = torch.tensor(0.0, device=self.device)
            if self._patchgan_on and self.feature_matching_loss_weight > 0 and self.patchgan_discriminator_loss_weight > 0:
                real_discrminator_output: DiscriminatorForwardOutput = self.run_D(real_img_for_loss, real_c_enc if is_text_cond else real_c)
                real_features = real_discrminator_output.patchgan_logits
                fake_features = gen_discrminator_output.patchgan_logits
                feature_matching_loss = self.calculate_feature_matching_loss(real_features, fake_features)

            # Pixel loss.
            l1_pixel_loss = torch.tensor(0.0, device=self.device)
            if self._pixel_loss_on and self.l1_pixel_loss_weight > 0:
                l1_pixel_loss = self.calculate_pixel_loss(real_img_for_loss, gen_img, type='l1')
            
            l2_pixel_loss = torch.tensor(0.0, device=self.device)
            if self._pixel_loss_on and self.l2_pixel_loss_weight > 0:
                l2_pixel_loss = self.calculate_pixel_loss(real_img_for_loss, gen_img, type='l2')
                
            # Perceptual loss.
            perceptual_loss = torch.tensor(0.0, device=self.device)
            if self._perceptual_loss_on and self.perceptual_loss_weight > 0:
                perceptual_loss = self.calculate_perceptual_loss(real_img_for_loss, gen_img)

            # SSIM loss.
            ssim_loss = torch.tensor(0.0, device=self.device)
            if self._ssim_loss_on and self.ssim_loss_weight > 0:
                ssim_loss = self.calculate_ssim_loss(real_img_for_loss, gen_img)

            # Multiscale pixel loss: the value range of images is [-1, 1].
            multiscale_pixel_loss = torch.tensor(0.0, device=self.device)
            multiscale_pixel_losses = [] # for logging
            if self._multiscale_pixel_loss_on and len(self.multiscale_pixel_loss_weights) > 0:
                real_multiscale_imgs = self.img_transform.multiscale_forward(real_img, gen_multiscale_imgs) # still in [0, 1] range
                real_multiscale_imgs = [(x * 2 - 1) for x in real_multiscale_imgs] # convert to [-1, 1] range 
                for i in range(len(gen_multiscale_imgs)):
                    if i in self.multiscale_block_indices:
                        weight = self.multiscale_pixel_loss_weights[self.multiscale_block_indices.index(i)]
                    else:
                        weight = 0.0
                    pixel_loss_cur_resolution = self.calculate_pixel_loss(real_multiscale_imgs[i], gen_multiscale_imgs[i], type='l1')
                    if cur_nimg >= (self.multiscale_pixel_loss_start_kimg * 1e3) and cur_nimg < (self.multiscale_pixel_loss_end_kimg * 1e3):
                        multiscale_pixel_loss += weight * pixel_loss_cur_resolution
                    else:
                        multiscale_pixel_loss += weight * pixel_loss_cur_resolution * 0.0 # dummy loss
                    multiscale_pixel_losses.append(pixel_loss_cur_resolution)

            # Main reconstruction loss.
            main_rec_loss = torch.tensor(0.0, device=self.device)
            
            if self._pixel_loss_on and self.l1_pixel_loss_weight > 0:
                main_rec_loss += self.l1_pixel_loss_weight * l1_pixel_loss
            
            if self._pixel_loss_on and self.l2_pixel_loss_weight > 0:
                main_rec_loss += self.l2_pixel_loss_weight * l2_pixel_loss
            
            if self._perceptual_loss_on and self.perceptual_loss_weight > 0:
                main_rec_loss += self.perceptual_loss_weight * perceptual_loss
            
            if self._ssim_loss_on and self.ssim_loss_weight > 0:
                main_rec_loss += self.ssim_loss_weight * ssim_loss
            
            if self._multiscale_pixel_loss_on and sum(self.multiscale_pixel_loss_weights) > 0:
                main_rec_loss += multiscale_pixel_loss

            # Vision foundation loss.
            vf_loss = torch.tensor(0.0, device=self.device)
            cur_vf_loss_weight = self.vf_loss_weight
            if self.vf_loss_weight > 0:
                vf_loss = generator_output.vf_loss
                vf_last_layer = generator_output.vf_last_layer
                cur_vf_loss_weight = self.calculate_cur_vf_loss_weight(main_rec_loss, vf_loss, vf_last_layer)

            # CLIP loss.
            clip_loss = torch.tensor(0.0, device=self.device)
            if self.clip_loss_weight > 0 and cur_nimg >= (self.clip_loss_start_kimg * 1e3):
                gen_img = gen_img.add(1).div(2)  # convert to [0, 1] range
                
                if gen_img.size(-1) > 64:
                    gen_img = RandomCrop(64)(gen_img)
                
                gen_img = self._safe_resize(gen_img, 224)
                gen_img_features = self.clip.encode_image(gen_img)
                real_text_features = self.clip.encode_text(real_c)
                clip_loss = self.calculate_spherical_distance(gen_img_features, real_text_features).mean()

            # Compression losses.
            if self.compression_mode == 'continuous':
                kl_loss = generator_output.kl_loss
            
            elif self.compression_mode == 'discrete':
                entropy_loss = generator_output.entropy_loss
                vq_loss = generator_output.vq_loss
                codebook_usages = generator_output.codebook_usages
            
            # Safe loss checking.
            skip_g_loss_local = torch.tensor(0.0, device=self.device)

            g_loss_safe_mark_dict_local = {
                'l1_pixel_loss'         : SAFE_MARK,
                'l2_pixel_loss'         : SAFE_MARK,
                'perceptual_loss'       : SAFE_MARK,
                'ssim_loss'             : SAFE_MARK,
                'multiscale_pixel_loss' : SAFE_MARK,
                'stylegan_t_gen_loss'   : SAFE_MARK,
                'patchgan_gen_loss'     : SAFE_MARK,
                'feature_matching_loss' : SAFE_MARK,
                'clip_loss'             : SAFE_MARK,
            }
            g_loss_names = list(g_loss_safe_mark_dict_local.keys())

            base_losses = {
                'l1_pixel_loss'         : l1_pixel_loss,
                'l2_pixel_loss'         : l2_pixel_loss,
                'perceptual_loss'       : perceptual_loss,
                'ssim_loss'             : ssim_loss,
                'multiscale_pixel_loss' : multiscale_pixel_loss,
                'stylegan_t_gen_loss'   : stylegan_t_gen_loss,
                'patchgan_gen_loss'     : patchgan_gen_loss,
                'feature_matching_loss' : feature_matching_loss,
                'clip_loss'             : clip_loss,
            }
            
            loss_dict = {
                k: (v.detach().item() if isinstance(v, torch.Tensor) else v)
                for k, v in base_losses.items()
            }

            if cur_nimg > (self.resume_kimg * 1e3 + self.safe_loss_checking_start_nimg):
                if self.prev_loss_dict is not None:
                    for name, curr_val in loss_dict.items():
                        prev_val = self.prev_loss_dict[name]
                        
                        is_finite = math.isfinite(curr_val)
                        too_large = (prev_val > 1e-6) and (curr_val > prev_val * 10)

                        # Check if the current loss is finite and not too large according to the previous loss.
                        if name in ['l1_pixel_loss', 'l2_pixel_loss', 'perceptual_loss', 'ssim_loss', 'multiscale_pixel_loss']:
                            if not is_finite or too_large:
                                skip_g_loss_local.fill_(1.0)
                                g_loss_safe_mark_dict_local[name] = UNSAFE_MARK          
                        else:
                            if not is_finite:
                                skip_g_loss_local.fill_(1.0)
                                g_loss_safe_mark_dict_local[name] = UNSAFE_MARK

            # Gather across all ranks.
            g_loss_safe_mark_tensor_local = torch.tensor(list(g_loss_safe_mark_dict_local.values()), device=self.device, dtype=torch.int)

            if dist.is_initialized():
                torch.distributed.all_reduce(skip_g_loss_local, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(g_loss_safe_mark_tensor_local, op=torch.distributed.ReduceOp.MIN)

            skip_g_loss_synced = bool(skip_g_loss_local.item()) 
            g_loss_safe_mark_dict_synced = {
                k: bool(g_loss_safe_mark_tensor_local[i].item()) for i, k in enumerate(g_loss_names)
            }
            
            if self.compression_mode == 'continuous':
                g_loss = \
                (main_rec_loss +
                 self.stylegan_t_discriminator_loss_weight * stylegan_t_gen_loss +
                 self.patchgan_discriminator_loss_weight * patchgan_gen_loss +
                 self.feature_matching_loss_weight * feature_matching_loss +
                 cur_vf_loss_weight * vf_loss +
                 self.clip_loss_weight * clip_loss +             
                 self.kl_loss_weight * kl_loss)

            elif self.compression_mode == 'discrete':
                g_loss = \
                (main_rec_loss +
                 self.stylegan_t_discriminator_loss_weight * stylegan_t_gen_loss +
                 self.patchgan_discriminator_loss_weight * patchgan_gen_loss +
                 self.feature_matching_loss_weight * feature_matching_loss +
                 cur_vf_loss_weight * vf_loss +
                 self.clip_loss_weight * clip_loss +
                 self.entropy_loss_weight * entropy_loss +          
                 self.vq_loss_weight * vq_loss)

            # Skip G backward if unsafe loss.
            if skip_g_loss_synced:
                g_loss = torch.nan_to_num(g_loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0 # to release the computation graph
                training_stats.report('Loss/G/skipped', 1.0)
            else:
                training_stats.report('Loss/G/skipped', 0.0)        

            # Log safe marks.
            for k in g_loss_names:
                training_stats.report(f'Loss/G/is_safe/{k}', int(g_loss_safe_mark_dict_synced[k]))
                if not g_loss_safe_mark_dict_synced[k]:
                    dist.print0(f"[SafeLoss][G] Unsafe {k} at {cur_nimg//1000} kimg - skipping.")
            
            # Backward pass for generator.
            g_loss.backward()

            # Record the current loss dict for next iteration if safe.
            if skip_g_loss_synced:
                return
            else:
                self.prev_loss_dict = loss_dict

            # Collect stats.
            if self.l1_pixel_loss_weight > 0:
                training_stats.report('Loss/G/l1_pixel_loss', l1_pixel_loss)
            
            if self.l2_pixel_loss_weight > 0:
                training_stats.report('Loss/G/l2_pixel_loss', l2_pixel_loss)
            
            if self.perceptual_loss_weight > 0:
                training_stats.report('Loss/G/perceptual_loss', perceptual_loss)

            if self.ssim_loss_weight > 0:
                training_stats.report('Loss/G/ssim_loss', ssim_loss)

            if cur_nimg >= (self.multiscale_pixel_loss_start_kimg * 1e3) and cur_nimg < (self.multiscale_pixel_loss_end_kimg * 1e3) and \
                sum(self.multiscale_pixel_loss_weights) > 0:
                training_stats.report('Loss/G/multiscale_pixel_loss', multiscale_pixel_loss)
                for i, weight in enumerate(self.multiscale_pixel_loss_weights):
                    training_stats.report(f'Loss/G/multiscale_pixel_loss_block{self.multiscale_block_indices[i]:01d}', multiscale_pixel_losses[i])

            if self.clip_loss_weight > 0 and cur_nimg >= (self.clip_loss_start_kimg * 1e3):
                training_stats.report('Loss/G/clip_loss', clip_loss)

            if self._stylegan_t_on and self.stylegan_t_discriminator_loss_weight > 0:
                training_stats.report('Loss/G/stylegan_t/loss', stylegan_t_gen_loss)
                training_stats.report('Loss/G/stylegan_t/fake_scores', stylegan_t_gen_logits)
                training_stats.report('Loss/G/stylegan_t/fake_signs', stylegan_t_gen_logits.sign())
            
            if self._patchgan_on and self.patchgan_discriminator_loss_weight > 0:
                training_stats.report('Loss/G/patchgan/loss', patchgan_gen_loss)
                for i, (s, sign) in enumerate(self._agg_patchgan_per_scale(patchgan_gen_logits)):
                    training_stats.report(f'Loss/G/patchgan/fake/scale{i}/fake_scores', s)
                    training_stats.report(f'Loss/G/patchgan/fake/scale{i}/fake_signs', sign)         
            
            if self._patchgan_on and self.feature_matching_loss_weight > 0:
                training_stats.report('Loss/G/patchgan/feature_matching_loss', feature_matching_loss)

            if self.vf_loss_weight > 0:
                training_stats.report('Loss/G/vf_loss', vf_loss)

            if self.compression_mode == 'continuous' and self.kl_loss_weight > 0:
                training_stats.report('Loss/G/kl_loss', kl_loss)
            
            elif self.compression_mode == 'discrete':
                if self.entropy_loss_weight > 0:
                    training_stats.report('Loss/G/entropy_loss', entropy_loss)
                
                if self.vq_loss_weight > 0:
                    training_stats.report('Loss/G/vq_loss', vq_loss)
                    training_stats.report('Loss/G/codebook_usages', codebook_usages)

            # Update phase based on current losses: after safe loss checking and backward pass.
            pixel_now = l1_pixel_loss.detach().item() if self._pixel_loss_window_type == 'l1' else l2_pixel_loss.detach().item()
            d_now = stylegan_t_gen_loss.detach().item() if self.stylegan_t_discriminator_loss_weight > 0 else 0.
            self._update_phase(cur_nimg, pixel_now, d_now)
