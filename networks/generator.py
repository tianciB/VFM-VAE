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


"""
VFM-VAE architecture from
"Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch_utils import misc
from torch_utils import distributed as dist
from torch_utils.ops import upfirdn2d, conv2d_resample, bias_act, fma
from typing import Union, Any, Optional, List

from networks.utils.shared import FullyConnectedLayer, MLP, GroupNorm32, StyleSplit, ScaleAdaptiveAvgPool2d
from networks.utils.ldm_utils import LDMAdapter, EquivarianceTransform
from networks.utils.gigagan_utils import SelfAttentionBlock, CrossAttentionBlock
from networks.utils.convnext_utils import ConvNeXtSynthesisLayer, ConvNeXtToRGBLayer, SeparableUpsampleWithFixedBlur
from networks.utils.dataclasses import EncodeOutput, GeneratorForwardOutput
from networks.utils.vfm_utils import VFMEncoder


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt() # compute 2nd moment along dim and normalize the input tensor


def modulated_conv2d(
    x: torch.Tensor,                                 # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight: torch.Tensor,                            # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles: torch.Tensor,                            # Modulation coefficients of shape [batch_size, in_channels].
    noise: Optional[torch.Tensor] = None,            # Optional noise tensor to add to the output activations.
    up: int = 1,                                     # Integer upsampling factor.
    down: int = 1,                                   # Integer downsampling factor.
    padding: int = 0,                                # Padding with respect to the upsampled image.
    resample_filter: Optional[list[int]] = None,     # Low-pass filter to apply when resampling activations.
    demodulate: bool = True,                         # Apply weight demodulation?
    flip_weight: bool = True,                        # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv: bool = True,                      # Perform modulation, convolution, and demodulation as a single fused operation?
) -> torch.Tensor:
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


class SynthesisInput(torch.nn.Module):
    def __init__(
        self,
        w_dim: int,          # Intermediate latent (W) dimensionality.
        channels: int,       # Number of output channels.
        size: int,           # Output spatial size.
        sampling_rate: int,  # Output sampling rate.
        bandwidth: int,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])

        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = F.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x.contiguous()

    def extra_repr(self) -> str:
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,                        # Number of input channels.
        out_channels: int,                       # Number of output channels.
        w_dim: int,                              # Intermediate latent (W) dimensionality.
        resolution: int,                         # Resolution of this layer.
        kernel_size: int = 3,                    # Convolution kernel size.
        up: int = 1,                             # Integer upsampling factor.
        use_noise: bool = False,                 # Enable noise input?
        activation: str = 'lrelu',               # Activation function: 'relu', 'lrelu', etc.
        resample_filter: list[int] = [1,3,3,1],  # Low-pass filter to apply when resampling activations.
        conv_clamp: Optional[int] = None,        # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last: bool = False,             # Use channels_last format for the weights?
        layer_scale_init: float = 1e-5,          # Initial value of layer scale.
        residual: bool = False,                  # Residual convolution?
        gn_groups: int = 32,                     # Number of groups for GroupNorm
    ):
        super().__init__()
        if residual: assert in_channels == out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.residual = residual

        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = Parameter(torch.zeros([]))

        self.affine = StyleSplit(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = Parameter(torch.zeros([out_channels]))

        if self.residual:
            assert up == 1
            self.norm = GroupNorm32(gn_groups, out_channels)
            self.gamma = Parameter(layer_scale_init * torch.ones([1, out_channels, 1, 1])).to(memory_format=memory_format)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise_mode: str = 'const',
        fused_modconv: bool = True,
        gain: int = 1,
    ) -> torch.Tensor:
        dtype = x.dtype
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        styles = self.affine(w)

        if self.residual:
            x = self.norm(x)

        y = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, fused_modconv=fused_modconv,
                             padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight)
        y = y.to(dtype)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        y = bias_act.bias_act(y, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)

        if self.residual:
            y = self.gamma * y
            y = y.to(dtype).add_(x).mul(np.sqrt(2))

        return y

    def extra_repr(self) -> str:
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])


class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        kernel_size: int = 1,
        conv_clamp: Optional[int] = None,
        channels_last: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = StyleSplit(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = Parameter(0.1*torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'


# --------------------------------------------------------------------------------------
# The code above preserves compatibility with the legacy StyleGAN-T implementation.
# The subsequent part introduces the modified architecture for VFM-VAE.
# --------------------------------------------------------------------------------------


class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        block_index: int,                               # Block index in the synthesis network.
        in_channels: int,                               # Number of input channels, 0 = first block.
        out_channels: int,                              # Number of output channels.
        last_out_channels: Union[None, int],            # Number of output channels of the last block.
        c_dim: int,                                     # Text embedding dimension
        w_dim: int,                                     # Style latent (W) dimension.
        resolution: int,                                # Resolution of this block.
        img_channels: int,                              # Number of output color channels.
        is_first: bool,                                 # Is this the first block?
        is_last: bool,                                  # Is this the last block?
        num_res_blocks: int = 1,                        # Number of conv layers per block.
        use_multiscale_output: bool = False,            # Use multi-stage pixel output?
        architecture: str = 'skip',                     # Architecture: 'orig', 'skip'.
        resample_filter: list[int] = [1,3,3,1],         # Low-pass filter to apply when resampling activations.
        conv_clamp: Optional[int] = None,               # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16: bool = False,                         # Use FP16 for this block?
        fp16_channels_last: bool = False,               # Use channels-last memory format with FP16?
        fused_modconv_default: Any = 'inference_only',  # Default value of fused_modconv.
        attn_block_indices: list[int] = [],             # Block indices for attention.
        attn_depths: list[int] = [],                    # Depths for attention.
        use_self_attn: bool = False,                    # Use self-attention?
        use_cross_attn: bool = False,                   # Use cross-attention?
        attn_heads: int = 8,                            # Number of attention heads.
        attn_ff_mult: int = 4,                          # Multiplier for the hidden dimension of the FFN.
        use_convnext: bool = False,                     # Use ConvNext block?
        use_gaussian_blur: bool = True,                 # Use Gaussian blur for upsampling?
        add_additional_convnext: bool = False,          # Add additional ConvNeXt block for low block indices?
        legacy: bool = False,                           # Use legacy ConvNeXt implementation?
        **layer_kwargs,                                 # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip']
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_out_channels = last_out_channels # for upsampling
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.num_conv = 0
        self.num_torgb = 0
        
        # Multiscale output.
        assert architecture == 'skip' if use_multiscale_output else True, "Only skip architecture is supported for multiscale pixel loss."
        self.use_multiscale_output = use_multiscale_output

        # Whether to use ConvNeXt block, instead of the original SynthesisLayer.
        self.use_convnext = use_convnext
        self.add_additional_convnext = add_additional_convnext
        if not use_convnext:
            self.fused_modconv_default = fused_modconv_default
            self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        
        # Setting the kernel size for the ConvNeXt block.
        kernel_size = 5 if block_index <= 1 else 7 # smaller kernel size for lower resolution
        self.kernel_size = kernel_size

        # Setting the blur kernel size for upsampling.
        blur_kernel = "3x3" if block_index <= 2 else "5x5" # smaller kernel size for lower resolution

        # If this is the first block and do not use concatanated latent, we need to add an input layer.
        if in_channels == 0:
            self.input = SynthesisInput(w_dim=self.w_dim, channels=out_channels, size=resolution, sampling_rate=resolution, bandwidth=2)
            self.num_conv += 1

        # If using ConvNeXt, we need to add a separate upsampling layer, followed by a ConvNeXt block.
        if in_channels != 0:
            if use_convnext:
                dist.print0(f"At resolution {resolution}, using sampling with {not is_first} pre-normalization and ConvNeXt block with kernel size {kernel_size}, use gaussian blur = {use_gaussian_blur}, and blur kernel = {blur_kernel}.")
                self.seperate_upsample_conv = SeparableUpsampleWithFixedBlur(in_channels, out_channels, upscale_factor=2, pre_normalize=not is_first, use_gaussian_blur=use_gaussian_blur, blur_kernel=blur_kernel)
                # Default residual connection and 5x5 / 7x7 kernel.
                self.conv0 = ConvNeXtSynthesisLayer(out_channels, w_dim=w_dim, kernel_size=kernel_size,
                                                    channels_last=self.channels_last, block_index=block_index, legacy=legacy)
            else:
                dist.print0(f"At resolution {resolution}, using default SynthesisLayer with kernel size 3.")
                # Default no residual connection and 3x3 kernel.
                self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                            resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        # The main convolutional layers.
        convs = []
        for _ in range(num_res_blocks):
            if use_convnext:
                num_layers = 3 if block_index <= 3 and add_additional_convnext else 2
                for _ in range(num_layers):
                    convs.append(ConvNeXtSynthesisLayer(out_channels, w_dim=w_dim, kernel_size=kernel_size,
                                                        channels_last=self.channels_last, block_index=block_index, legacy=legacy))
            else:
                convs.append(SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs))
                convs.append(SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                            conv_clamp=conv_clamp, channels_last=self.channels_last,
                                            residual=True, **layer_kwargs))

        self.convs1 = torch.nn.ModuleList(convs)
        self.num_conv += len(convs)

        # ToRGB layer.
        if is_last or architecture == 'skip':
            if use_convnext:
                self.torgb = ConvNeXtToRGBLayer(out_channels, img_channels, w_dim=w_dim, channels_last=self.channels_last)
            else:
                self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        # For upsampling the last output features.
        if use_multiscale_output and last_out_channels is not None:
            dist.print0(f"At resolution {resolution}, using last upsample conv with gaussian blur = {use_gaussian_blur} and blur kernel = {blur_kernel}.")
            self.last_upsample_conv = SeparableUpsampleWithFixedBlur(last_out_channels, out_channels, upscale_factor=2, use_gaussian_blur=use_gaussian_blur, blur_kernel=blur_kernel)

        # Attention blocks.
        self.attn_block_indices = attn_block_indices        # block indices for attention, recommended at fp32 precision
        self.attn_depths = attn_depths                      # depths for attention, recommended at fp32 precision
        self.use_self_attn = use_self_attn                  # use self-attention
        self.use_cross_attn = use_cross_attn                # use cross-attention
        self.attn_heads = attn_heads                        # number of attention heads, default=8            
        self.attn_ff_mult = attn_ff_mult                    # multiplier for the hidden dimension of the FFN, default=4

        if block_index in attn_block_indices:
            depth = attn_depths[attn_block_indices.index(block_index)]
        else:
            depth = 0

        self.has_self_attn = use_self_attn and depth > 0
        self.has_cross_attn = use_cross_attn and depth > 0

        if self.has_self_attn:
            self.self_attns = torch.nn.ModuleList([
                SelfAttentionBlock(
                    out_channels,
                    dim_head=out_channels // attn_heads,
                    heads=attn_heads,
                    ff_mult=attn_ff_mult
                )
                for _ in range(depth)
            ])
        else:
            self.self_attns = None

        if self.has_cross_attn:
            self.cross_attns = torch.nn.ModuleList([
                CrossAttentionBlock(
                    out_channels,
                    dim_context=c_dim,
                    dim_head=out_channels // attn_heads,
                    heads=attn_heads,
                    ff_mult=attn_ff_mult
                )
                for _ in range(depth)
            ])
        else:
            self.cross_attns = None

    def forward(
        self,
        x: torch.Tensor,
        x_sum: Optional[torch.Tensor],
        img: Optional[torch.Tensor],
        ws: torch.Tensor,
        text: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        force_fp32: bool = False,
        fused_modconv: bool = True,
        **layer_kwargs,
    ) -> Union[torch.Tensor, Union[torch.Tensor, None]]:
        # Unbind the weights.
        w_iter = iter(ws.unbind(dim=1))
        
        # Mixed precision.
        if ws.device.type != 'cuda':
            force_fp32 = True
            dtype = torch.float32
            memory_format = torch.contiguous_format
            amp_enabled = False
            amp_dtype = torch.float32
        
        else:
            dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
            memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
            amp_enabled = self.use_fp16 and not force_fp32
            amp_dtype = torch.float16
        
        # Only the handcrafted implementation is supported for fused_modconv.
        if self.use_convnext:
            if fused_modconv is None:
                fused_modconv = self.fused_modconv_default
            
            if fused_modconv == 'inference_only':
                fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.input(next(w_iter))
        
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.use_convnext:
            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                x = self.seperate_upsample_conv(x)
                x = self.conv0(x, next(w_iter))
                for conv in self.convs1:
                    x = conv(x, next(w_iter))
        
        else:
            if self.in_channels == 0:
                for conv in self.convs1:
                    x = conv(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            else:
                x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
                for conv in self.convs1:
                    x = conv(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        
        # Attention layers.
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            if self.has_self_attn:
                for attn in self.self_attns:
                    x = attn(x)

            if self.has_cross_attn:
                assert text is not None, "Text input must be provided for cross-attention."
                for attn in self.cross_attns:
                    x = attn(x, text, mask=text_mask)

        # For manual FP16 control (StyleGAN-T style). Not related to torch.autocast.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # ToRGB.
        if self.use_multiscale_output:
            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                x_sum = self.last_upsample_conv(x_sum) + x if self.last_out_channels is not None else x
                img = self.torgb(x_sum, next(w_iter))
            img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        
        else:
            if img is not None:
                misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
                img = upfirdn2d.upsample2d(img, self.resample_filter)

            if self.is_last or self.architecture == 'skip':
                y = self.torgb(x, next(w_iter))
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y
    
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, x_sum, img

    def extra_repr(self) -> str:
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

 
class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim_input: int,               # Input latent (Z) dimensionality for MLP, 0 = no latent.
        z_dim_output: int,              # Output latent (Z) dimensionality for MLP, 0 = no latent.
        c_dim: int,                     # Text embedding dimensionality, 0 = no text.
        w_dim: int,                     # Intermediate latent (W) dimensionality.
        label_type: str,                # Type of label conditioning: 'text', 'cls2text', 'cls2id'.
        num_layers: int = 2,            # Number of mapping layers.
        activation: str = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 0.01,    # Learning rate multiplier for the mapping layers.
        x_avg_beta: float = 0.995,      # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim_input = z_dim_input
        self.z_dim_output = z_dim_output
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.x_avg_beta = x_avg_beta
        self.num_ws = None
        
        self.label_type = label_type
        if label_type in ['text', 'cls2text']:
            self.mlp = MLP([z_dim_input] * num_layers + [z_dim_output], activation=activation, lr_multiplier=lr_multiplier, linear_out=True)
            self.register_buffer('x_avg', torch.zeros([z_dim_output], dtype=torch.float32))

        elif label_type == 'cls2id':
            assert z_dim_input < w_dim, 'z_dim must be less than w_dim for cls2id.'
            c_embed_dim = 1024 # align with the text embedding dimension
            self.embed = FullyConnectedLayer(c_dim, c_embed_dim) if c_dim > 0 else None
            self.mlp = MLP([z_dim_input + c_embed_dim] * num_layers + [w_dim] if self.c_dim > 0 else [z_dim_input] * num_layers + [w_dim], 
                           activation=activation, lr_multiplier=lr_multiplier, linear_out=True)
            self.register_buffer('x_avg', torch.zeros([self.w_dim], dtype=torch.float32))

    def forward(
        self,
        z: torch.Tensor,            # pooled latent feature
        c: Optional[torch.Tensor],  # None, global text embedding, or one-hot class label
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        # Differentiate between label types.
        if self.label_type in ['text', 'cls2text']:
            x = self.mlp(normalize_2nd_moment(z))

            if self.x_avg_beta is not None and self.training:
                self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

            if truncation_psi != 1:
                assert self.x_avg_beta is not None
                x = self.x_avg.lerp(x, truncation_psi)

            w = torch.cat([x, F.normalize(c, dim=1)], dim=1) if self.c_dim > 0 else x

        elif self.label_type == 'cls2id':
            x = self.mlp(torch.cat([normalize_2nd_moment(z), normalize_2nd_moment(self.embed(c))], dim=1)) \
                if self.c_dim > 0 else self.mlp(normalize_2nd_moment(z))

            if self.x_avg_beta is not None and self.training:
                self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))
            
            if truncation_psi != 1:
                assert self.x_avg_beta is not None
                x = self.x_avg.lerp(x, truncation_psi)

            w = x

        # Broadcast latent codes.
        if self.num_ws is not None:
            w = w.unsqueeze(1).repeat([1, self.num_ws, 1])
        
        return w


class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        c_dim: int,                                     # Text embedding dimension.
        w_dim: int,                                     # Style latent (W) dimension.
        img_resolution: int,                            # Output image resolution.
        img_channels: int = 3,                          # Number of color channels.
        channel_base: int = 32768,                      # Overall multiplier for the number of channels.
        channel_max: int = 512,                         # Maximum number of channels in any layer.
        num_fp16_res: int = 3,                          # Use FP16 for the N highest block indices.
        conv_clamp: Optional[int] = None,               # Clamp the output of convolution layers to +-X, None = disable clamping.
        num_blocks: int = 6,                            # Number of synthesis blocks.
        num_res_blocks: int = 3,                        # Number of residual blocks.
        z_resolution: int = 16,                         # Resolution of the latent (Z).
        z_dim: int = 8,                                 # Dimensionality of the latent (Z).
        concat_z_block_indices: list[int] = [],         # Block indices for concatenated latent (Z) processing.
        concat_z_mapped_dims: list[int] = [],           # Mapped concatenated latent (Z) dimensionalities.
        how_to_process_concat_z: str = 'unshuffle',     # How to process the concatenated latent (Z): 'unshuffle' or 'pooling'.
        activation_for_concat_z: str = 'gelu',          # Activation function for concatenated latent (Z) processing.
        use_multiscale_output: bool = False,            # Use multi-stage output?
        attn_block_indices: list[int] = [],             # Block indices for attention layers.
        attn_depths: list[int] = [],                    # Depths for attention layers.
        use_self_attn: bool = False,                    # Use self-attention layer?
        use_cross_attn: bool = False,                   # Use cross-attention layer?
        use_convnext: bool = False,                     # Use ConvNext block?
        use_gaussian_blur: bool = True,                 # Use Gaussian blur for upsampling?
        add_additional_convnext: bool = False,          # Add additional ConvNeXt block for low block indices.
        legacy: bool = False,                           # Use legacy ConvNeXt implementation?
        **block_kwargs,                                 # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4
        super().__init__()
        
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels

        # Set the number of channels for each resolution.
        self.num_blocks = num_blocks
        res_start = img_resolution // (2 ** (num_blocks - 1))
        self.block_resolutions = [res_start * (2 ** i) for i in range(num_blocks)]
        channel_scale_factor = img_resolution / 256
        channels_dict = {idx: min(channel_base // int(res / channel_scale_factor), 
                                  channel_max) for idx, res in enumerate(self.block_resolutions)}
        
        # Set the mixed precision.
        self.amp_dtype = torch.float16
        self.num_fp16_res = num_fp16_res
        fp16_idx = num_blocks - num_fp16_res

        # Latent (Z) dimensionality.
        self.z_resolution = z_resolution
        self.z_dim = z_dim

        # Concatenated latent (Z) processing.
        self.concat_z_block_indices = concat_z_block_indices
        self.concat_z_mapped_dims = concat_z_mapped_dims
        self.how_to_process_concat_z = how_to_process_concat_z
        self.activation_for_concat_z = activation_for_concat_z

        # Multiscale output.
        self.use_multiscale_output = use_multiscale_output
        
        # Atention layers.
        self.attn_block_indices = attn_block_indices
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn

        # Process the latent (Z) before concatenation: for each resolution block, the resolution of input is halved.
        self.z_convs = nn.ModuleDict()     # for concatenated latent (Z) processing
        self.adjust_concat_z_dims = dict() # for storing the adjusted concatenated latent (Z) dimensionalities

        for idx in self.concat_z_block_indices:
            layers = []
            res = self.block_resolutions[idx]

            if res < self.z_resolution * 2:
                if how_to_process_concat_z == 'unshuffle':
                    downscale_factor = int(self.z_resolution / res * 2)
                    input_channels = int(z_dim * (downscale_factor) ** 2)
                    adjust_concat_z_dim = concat_z_mapped_dims[idx] if len(concat_z_mapped_dims) > 0 else input_channels
                    layers += [
                        nn.PixelUnshuffle(downscale_factor),
                        self._make_3x3_conv(input_channels, adjust_concat_z_dim, activation=activation_for_concat_z, use_activation=True),
                        self._make_1x1_conv(adjust_concat_z_dim, adjust_concat_z_dim, use_activation=False),
                    ]
                
                elif how_to_process_concat_z == 'pooling':
                    downscale_factor = int(self.z_resolution / res * 2)
                    adjust_concat_z_dim = concat_z_mapped_dims[idx] if len(concat_z_mapped_dims) > 0 else z_dim
                    layers += [
                        ScaleAdaptiveAvgPool2d(downscale_factor),
                        self._make_3x3_conv(z_dim, adjust_concat_z_dim, activation=activation_for_concat_z, use_activation=True),
                        self._make_1x1_conv(adjust_concat_z_dim, adjust_concat_z_dim, use_activation=False),
                    ]
            
            elif res == self.z_resolution * 2:
                adjust_concat_z_dim = concat_z_mapped_dims[idx] if len(concat_z_mapped_dims) > 0 else z_dim
                layers += [
                    self._make_3x3_conv(z_dim, adjust_concat_z_dim, activation=activation_for_concat_z, use_activation=True),
                    self._make_1x1_conv(adjust_concat_z_dim, adjust_concat_z_dim, use_activation=False),
                ]
            
            else:
                if how_to_process_concat_z == 'unshuffle':
                    upscale_factor = int(res / self.z_resolution / 2)
                    adjust_concat_z_dim = concat_z_mapped_dims[idx] if len(concat_z_mapped_dims) > 0 else z_dim
                    output_channels = int(adjust_concat_z_dim * (upscale_factor ** 2))
                    layers += [
                        self._make_3x3_conv(z_dim, output_channels, activation=activation_for_concat_z, use_activation=True),
                        nn.PixelShuffle(upscale_factor),
                        self._make_1x1_conv(adjust_concat_z_dim, adjust_concat_z_dim, use_activation=False),
                    ]

                elif how_to_process_concat_z == 'pooling':
                    dist.print0(f'Warning: For resolution {res}, using pooling to process concatenated latent (Z) is not recommended, cued by lacking of positional information. We use shuffle instead.')
                    upscale_factor = int(res / self.z_resolution / 2)
                    adjust_concat_z_dim = concat_z_mapped_dims[idx] if len(concat_z_mapped_dims) > 0 else z_dim
                    output_channels = int(adjust_concat_z_dim * (upscale_factor ** 2))
                    layers += [
                        self._make_3x3_conv(z_dim, output_channels, activation=activation_for_concat_z, use_activation=True),
                        nn.PixelShuffle(upscale_factor),
                        self._make_1x1_conv(adjust_concat_z_dim, adjust_concat_z_dim, use_activation=False),
                    ]

            self.z_convs[f"{idx:01d}"] = nn.Sequential(*layers)
            self.adjust_concat_z_dims[idx] = adjust_concat_z_dim

        dist.print0(f'Constructing SynthesisNetwork with concatenated latent (Z) processing at block indices: {self.concat_z_block_indices}.')
        self.blocks = nn.ModuleDict()
        self.num_ws = 0

        for idx in range(num_blocks):
            in_channels = channels_dict[idx - 1] if idx > 0 else 0
            last_out_channels = channels_dict[idx - 1] if idx > 0 else None
            res = self.block_resolutions[idx]
            
            if idx in concat_z_block_indices:
                dist.print0(f'At resolution {res}, original in_channels = {in_channels}, concat_z_dim = {self.concat_z_mapped_dims[idx]}, activation = {self.activation_for_concat_z}.')
                in_channels += self.adjust_concat_z_dims[idx]
            else:
                dist.print0(f'At resolution {res}, original in_channels = {in_channels}, concat_z_dim = 0.')
            
            out_channels = channels_dict[idx]
            use_fp16 = (idx >= fp16_idx)

            is_first = (idx == 0)
            is_last = (idx == num_blocks - 1)

            attn_kwargs = {
                'attn_block_indices': attn_block_indices,
                'attn_depths': attn_depths,
                'use_self_attn': use_self_attn,
                'use_cross_attn': use_cross_attn,
            }

            block = SynthesisBlock(
                block_index=idx,
                in_channels=in_channels, 
                out_channels=out_channels,
                last_out_channels=last_out_channels,
                c_dim=c_dim,
                w_dim=w_dim,
                resolution=res, 
                img_channels=img_channels, 
                is_first=is_first,
                is_last=is_last, 
                use_fp16=use_fp16,
                conv_clamp=conv_clamp,
                num_res_blocks=num_res_blocks, 
                use_multiscale_output=use_multiscale_output,
                use_convnext=use_convnext,
                use_gaussian_blur=use_gaussian_blur,
                add_additional_convnext=add_additional_convnext,
                legacy=legacy,
                **attn_kwargs,
                **block_kwargs
            )
            
            self.num_ws += (block.num_conv + block.num_torgb)
            self.blocks[f"{idx:01d}"] = block

    def _make_3x3_conv(self, cin, cout, activation='gelu', use_activation=True):
        layers = [
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            GroupNorm32(min(32, cout), cout),
        ]
        if use_activation:
            if activation == 'lrelu':
                layers += [nn.LeakyReLU(negative_slope=0.2)]
            elif activation == 'silu':
                layers += [nn.SiLU()]
            elif activation == 'gelu':
                layers += [nn.GELU()]

        return nn.Sequential(*layers)

    def _make_1x1_conv(self, cin, cout, activation='gelu', use_activation=True):
        layers = [
            nn.Conv2d(cin, cout, 1, bias=False),
            GroupNorm32(min(32, cout), cout),
        ]
        if use_activation:
            if activation == 'lrelu':
                layers += [nn.LeakyReLU(negative_slope=0.2)]
            elif activation == 'silu':
                layers += [nn.SiLU()]
            elif activation == 'gelu':
                layers += [nn.GELU()]
        
        return nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        ws: torch.Tensor,
        text: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        **block_kwargs
    ) -> torch.Tensor:
        # Split the style latent (W) into blocks.
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0
            for idx in range(self.num_blocks):
                block = self.blocks[f"{idx:01d}"]
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv + block.num_torgb

        # Initialize variables.
        x = x_sum = img = None
        multiscale_imgs = []

        # Main loop through each block.
        for idx, cur_ws in zip(range(self.num_blocks), block_ws):
            block = self.blocks[f"{idx:01d}"]

            if idx in self.concat_z_block_indices:
                with torch.amp.autocast('cuda', enabled=True if block.use_fp16 else False, dtype=self.amp_dtype):
                    z_concat = self.z_convs[f"{idx:01d}"](z)
                    x = torch.cat([x, z_concat], dim=1) if x is not None else z_concat
            
            x, x_sum, img = block(x, x_sum, img, cur_ws, text, text_mask, **block_kwargs)
            
            if not block.is_last:
                multiscale_imgs.append(img)

        return img, multiscale_imgs[::-1] # reverse the order of the images from the smallest to the largest
    
    def extra_repr(self) -> str:
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


class Generator(torch.nn.Module):
    def __init__(
        self,
        # Conditioning settings.
        conditional: bool,                                                  # Label conditional?
        label_type: str,                                                    # Type of label: 'text', 'cls2text', 'cls2id'.
        label_dim: Optional[int],                                           # Label dimensionality.
        # Vision foundation model settings.
        vfm_name: str,                                                      # Name of the vision foundation model.
        scale_factor: float,                                                # Scale factor for the vision foundation model.
        patch_from_layers: List[int],                                       # Patch from layers for the vision foundation model.
        patch_in_dimensions: List[int],                                     # Patch input dimensions for projection.
        patch_out_dimensions: List[int],                                    # Patch output dimensions for projection.
        # Compression & decompression settings.
        compression_mode: str,                                              # Compression mode: 'continuous' or 'discrete'.
        how_to_compress: str,                                               # How to compress embedding: 'conv' or 'attnproj'.
        how_to_decompress: str,                                             # How to decompress embedding: 'conv' or 'attnproj'.
        decompress_factor: int,                                             # Decompression factor.
        attnproj_quant_layers: int = 1,                                     # Number of quantization attention projection layers.
        attnproj_post_quant_layers: int = 1,                                # Number of post-quantization attention projection layers.
        # Latent (Z) settings.
        resolution_compression_factor: int = 16,                            # Compression factor for the image resolution to get latent (Z) resolution.
        z_dimension: int = 32,                                              # Dimensionality of the latent (Z) at continuous mode.
        vocab_width: int = 64,                                              # Dimensionality of the latent (Z) at discrete mode.
        z_pooled_resolution: int = 1,                                       # Resolution of the pooled latent (Z) for mapping.
        z_dim_for_mapping_mlp_output: int = 128,                            # Dimensionality of the latent (Z) for mapping MLP output.
        # Discrete VQ settings.
        vocab_size: int = 32768,                                            # Vocabulary size for discrete tokenizer.
        vocab_beta: float = 0.25,                                           # Beta parameter for vector quantization.
        use_entropy_loss: bool = False,                                     # Whether to use entropy loss for discrete tokenizers.
        entropy_temp: float = 0.01,                                         # Temperature for entropy loss.
        num_codebooks: int = 8,                                             # Number of codebooks for vector quantization.
        # Quantization losses settings.
        use_kl_loss: bool = False,                                          # Whether to use KL loss for continuous tokenizers.
        # VF loss settings.
        use_vf_loss: bool = False,                                          # Whether to use vision foundation loss.
        use_adaptive_vf_loss: bool = False,                                 # Whether to use adaptive vision foundation loss.
        distmat_margin: float = 0.0,                                        # Margin for distance matrix in VF loss.
        cos_margin: float = 0.0,                                            # Margin for cosine similarity in VF loss.
        distmat_weight: float = 1.0,                                        # Weight for distance matrix loss in VF loss.
        cos_weight: float = 1.0,                                            # Weight for cosine similarity loss in VF loss.
        # Concatenated latent (Z) settings.
        concat_z_block_indices: list[int] = [],                             # Block indices for concatenated latent (Z) processing.
        concat_z_mapped_dims: list[int] = [],                               # Mapped concatenated latent (Z) dimensionalities.
        how_to_process_concat_z: str = 'unshuffle',                         # How to process the latent (Z) before concat: 'unshuffle' or 'pooling'.
        activation_for_concat_z: str = 'gelu',                              # Activation function for concatenated latent (Z).
        # Architecture settings.
        use_multiscale_output: bool = True,                                 # Use multi-stage pixel output.
        attn_block_indices: list[int] = [],                                 # Block indices for attention layers.
        attn_depths: list[int] = [],                                        # Depths for attention layers.
        use_self_attn: bool = True,                                         # Use self-attention layer.
        use_cross_attn: bool = False,                                       # Use cross-attention layer.
        use_convnext: bool = True,                                          # Use ConvNext block.
        use_gaussian_blur: bool = True,                                     # Use Gaussian blur for upsampling.
        add_additional_convnext: bool = True,                               # Add additional ConvNeXt block for low block indices.
        # Equivariance regularization settings.
        use_equivariance_regularization: bool = False,                      # Use equivariance regularization.
        equivariance_regularization_p_prior: float = 0.5,                   # Probability of applying scale and rotation.
        equivariance_regularization_p_prior_scale: float = 0.25,            # Probability of applying scale.
        # Output image settings.
        img_resolution: int = 256,                                          # Output image resolution.
        img_channels: int = 3,                                              # Number of output color channels.
        # Training settings.
        train_mode: str = 'train_all',                                      # Control which layers are trainable.
        num_blocks: int = 6,                                                # Number of synthesis blocks.
        num_fp16_res: int = 3,                                              # Use FP16 for the N highest block indices.
        conv_clamp: Optional[int] = 256,                                    # Clamp the convolution output.
        legacy: bool = False,                                               # Use legacy ConvNeXt implementation?
        # Other settings.
        synthesis_kwargs: dict = {},
    ):
        super().__init__()

        # Conditioning settings.
        self.conditional = conditional
        self.label_type = label_type

        # Number of synthesis blocks.
        self.num_blocks = num_blocks

        # Image settings.
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Vision foundation model settings.
        self.vfm_encoder = VFMEncoder(
            model_name=vfm_name,
            conditional=conditional,
            label_type=label_type,
            scale_factor=scale_factor,
            patch_from_layers=patch_from_layers
        )
        self.patch_resolutions = [int(img_resolution * scale_factor // self.vfm_encoder.patch_size) for _ in patch_from_layers]

        assert img_resolution % self.vfm_encoder.patch_size == 0, \
            f'Image resolution {img_resolution} must be divisible by the vision foundation model patch size {self.vfm_encoder.patch_size}.'

        # Latent settings.
        self.z_resolution = int(img_resolution // resolution_compression_factor)
        self.z_dim = z_dimension if compression_mode == 'continuous' else vocab_width
        
        self.z_pooled_resolution = z_pooled_resolution
        self.z_dim_for_mapping = self.z_dim * decompress_factor * self.z_pooled_resolution ** 2
        self.z_dim_for_concatenated = self.z_dim * decompress_factor
        self.z_dim_for_mapping_mlp_output = z_dim_for_mapping_mlp_output
        
        if conditional:
            if label_type in ['text', 'cls2text']:
                self.c_dim = self.vfm_encoder.text_model.config.hidden_size
                self.z_dim_for_mapping_mlp_input = self.z_dim_for_mapping
                self.w_dim = self.z_dim_for_mapping_mlp_output + self.c_dim # concatenate the text embedding dimension

            elif label_type == 'cls2id':
                self.label_dim = label_dim
                self.c_dim = self.label_dim
                self.c_embed_dim = 1024 # align with the text embedding dimension
                self.z_dim_for_mapping_mlp_input = self.z_dim_for_mapping + self.c_embed_dim
                self.w_dim = self.z_dim_for_mapping_mlp_output
        
        else:
            self.c_dim = 0
            self.z_dim_for_mapping_mlp_input = self.z_dim_for_mapping
            self.w_dim = self.z_dim_for_mapping_mlp_output
        
        # Latent adaption settings.
        if compression_mode in ['continuous', 'discrete']:
            self.ldm_adapter = LDMAdapter(patch_from_layers=patch_from_layers, patch_resolutions=self.patch_resolutions,
                                          patch_in_dimensions=patch_in_dimensions, patch_out_dimensions=patch_out_dimensions,
                                          compression_mode=compression_mode, how_to_compress=how_to_compress, 
                                          how_to_decompress=how_to_decompress, decompress_factor=decompress_factor, 
                                          attnproj_quant_layers=attnproj_quant_layers, attnproj_post_quant_layers=attnproj_post_quant_layers,
                                          z_resolution=self.z_resolution, z_dimension=z_dimension,
                                          vocab_width=vocab_width, vocab_size=vocab_size, vocab_beta=vocab_beta,
                                          use_entropy_loss=use_entropy_loss, entropy_temp=entropy_temp, num_codebooks=num_codebooks,
                                          use_kl_loss=use_kl_loss, use_vf_loss=use_vf_loss, use_adaptive_vf_loss=use_adaptive_vf_loss,
                                          distmat_margin=distmat_margin, cos_margin=cos_margin, distmat_weight=distmat_weight, cos_weight=cos_weight,
                                        )

        # Concat latent z settings.
        self.concat_z_block_indices = concat_z_block_indices
        self.concat_z_mapped_dims = concat_z_mapped_dims

        # Multiscale output settings.
        self.use_multiscale_output = use_multiscale_output

        # Equivariant regularization settings.
        self.equivariance_transform = EquivarianceTransform(
            apply=use_equivariance_regularization,
            p_eq_prior=equivariance_regularization_p_prior,
            p_eq_prior_scale=equivariance_regularization_p_prior_scale,
        )

        dist.print0(
            f"\n=== Generator Latent Configuration ====\n"
            f"conditional: {conditional}, label type: {label_type}\n"
            f"z_resolution: {self.z_resolution}, c_dim: {self.c_dim}, w_dim: {self.w_dim}\n"
            f"z_pooled_resolution: {self.z_pooled_resolution}, z_dim (mapping): {self.z_dim_for_mapping}\n"
            f"z_dim (mapping mlp input): {self.z_dim_for_mapping_mlp_input}, z_dim (mapping mlp output): {self.z_dim_for_mapping_mlp_output}\n"
            f"w_dim: {self.w_dim}, z_dim (concatenated): {self.z_dim_for_concatenated}\n"
            f"=======================================\n"
        )

        # Networks.
        self.mapping = MappingNetwork(z_dim_input=self.z_dim_for_mapping_mlp_input, z_dim_output=self.z_dim_for_mapping_mlp_output, 
                                      c_dim=self.c_dim, w_dim=self.w_dim, label_type=self.label_type)
        self.synthesis = SynthesisNetwork(z_resolution=self.z_resolution, z_dim=self.z_dim_for_concatenated, c_dim=self.c_dim, w_dim=self.w_dim,
                                          img_resolution=img_resolution, img_channels=img_channels, 
                                          concat_z_block_indices=concat_z_block_indices, concat_z_mapped_dims=concat_z_mapped_dims, 
                                          how_to_process_concat_z=how_to_process_concat_z,
                                          activation_for_concat_z=activation_for_concat_z,
                                          attn_block_indices=attn_block_indices, attn_depths=attn_depths,
                                          use_self_attn=use_self_attn, use_cross_attn=use_cross_attn,
                                          use_convnext=use_convnext, use_gaussian_blur=use_gaussian_blur,
                                          add_additional_convnext=add_additional_convnext,
                                          use_multiscale_output=use_multiscale_output,
                                          num_blocks=num_blocks, num_fp16_res=num_fp16_res, 
                                          conv_clamp=conv_clamp, legacy=legacy,
                                          **synthesis_kwargs)

        self.num_ws = self.synthesis.num_ws
        self.mapping.num_ws = self.num_ws # the ws of mapping network depends on the number of ws needed in the synthesis network

        # Set trainable layers.
        self.set_train_mode(train_mode)

    def set_train_mode(self, mode: str):
        if mode == 'train_all':
            trainable_layers = ['synthesis', 'mapping.mlp', 'ldm_adapter']
            if self.conditional and self.label_type == 'cls2id':
                trainable_layers.append('mapping.embed')

        elif mode == 'train_text_encoder':
            trainable_layers = ['clip']

        elif mode == 'train_the_second_half_decoder':
            trainable_layers = []
            for res in self.synthesis.block_resolutions:
                if res > 32:
                    trainable_layers.append(f'synthesis.b{res}')
            for res in self.concat_z_block_indices:
                if res > 32:
                    trainable_layers.append(f'z_convs.{res}')

        elif mode == 'train_decoder':
            trainable_layers = ['synthesis', 'mapping.mlp', 'ldm_adapter.post_quant']
            if self.conditional and self.label_type == 'cls2id':
                trainable_layers.append('mapping.embed')
            
        else:
            raise ValueError(f"Unknown train_mode {mode}")

        self.train_mode = mode
        self.trainable_layers = trainable_layers
        dist.print0(f"[Generator] train_mode set to {mode}.")

    @torch.no_grad()
    def encode(self, img: torch.Tensor, return_z_before_quantize=False, eq_scale_factor: float = 1.0, is_eq_prior: bool = False) -> torch.Tensor:
        patch_features, *_ = self.vfm_encoder.encode_image(img, eq_scale_factor=eq_scale_factor, is_eq_prior=is_eq_prior)
        ldm_out: EncodeOutput = self.ldm_adapter.encode(patch_features, return_z_before_quantize)
        return ldm_out.z  # z shape: [B, z_dim, H, W] or z_before_quantize, e.g. (mean || logvar) at continuous mode
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor, c: Optional[Union[list[str], torch.Tensor]] = None, truncation_psi: float = 1.0, **synthesis_kwargs) -> torch.Tensor:
        z = self.ldm_adapter.decode(z)
        z_pooled = F.adaptive_avg_pool2d(z, (self.z_pooled_resolution, self.z_pooled_resolution)).flatten(1)

        if self.label_type in ['text', 'cls2text']:
            fine_text_tokens, global_text_tokens, text_mask = self.vfm_encoder.encode_text(c)
            ws = self.mapping(z_pooled, global_text_tokens, truncation_psi=truncation_psi)
            img, *_ = self.synthesis(z, ws, fine_text_tokens, text_mask, **synthesis_kwargs)

        elif self.label_type == 'cls2id':
            ws = self.mapping(z_pooled, c, truncation_psi=truncation_psi)
            img, *_ = self.synthesis(z, ws, None, None, **synthesis_kwargs)
        
        return img

    def forward(
        self,
        img: torch.Tensor,                  # input image, shape: [B, 3, H, W]
        c: Union[list[str], torch.Tensor],  # text embedding or class label, shape: [B] or [B, c_dim]
        truncation_psi: float = 1.0,        # truncation psi for mapping network
        validation: bool = False,           # whether to use the equivariance transform
        **synthesis_kwargs
    ) -> torch.Tensor:
        # Compression and decompression.
        eq_scale_factor, eq_angle_factor, is_eq_prior = self.equivariance_transform(validation=validation) # get the scale and angle for EQ regularization
        patch_features, *_ = self.vfm_encoder.encode_image(img, eq_scale_factor=eq_scale_factor if is_eq_prior else 1.0,
                                                           is_eq_prior=is_eq_prior) # patch_features: list of tensors, each shape: [B, patch_num, patch_dim]
        ldm_out: EncodeOutput = self.ldm_adapter.encode(patch_features)
        z = ldm_out.z # z shape: [B, z_dim, H, W]

        # EQ regularization is only applied during training.
        if not validation and not is_eq_prior:
            z = F.interpolate(z, scale_factor=eq_scale_factor, mode='bilinear', align_corners=False) if eq_scale_factor != 1.0 else z
            z = torch.rot90(z, k=eq_angle_factor, dims=[-1, -2]) if eq_angle_factor != 0 else z

        z = self.ldm_adapter.decode(z) # z shape: [B, decompress_dim, H, W]
        z_pooled = F.adaptive_avg_pool2d(z, (self.z_pooled_resolution, self.z_pooled_resolution)).flatten(1)

        if self.label_type in ['text', 'cls2text']:
            fine_text_tokens, global_text_tokens, text_mask = self.vfm_encoder.encode_text(c)
            ws = self.mapping(z_pooled, global_text_tokens, truncation_psi=truncation_psi)
            gen_img, gen_multiscale_imgs = self.synthesis(z, ws, fine_text_tokens, text_mask, **synthesis_kwargs)

        elif self.label_type == 'cls2id':
            global_text_tokens = None # dummy
            ws = self.mapping(z_pooled, c, truncation_psi=truncation_psi)
            gen_img, gen_multiscale_imgs = self.synthesis(z, ws, None, None, **synthesis_kwargs)
        
        if self.ldm_adapter is not None:
            return GeneratorForwardOutput(
                gen_img=gen_img,
                gen_multiscale_imgs=gen_multiscale_imgs,
                vf_loss=ldm_out.vf_loss,
                vf_last_layer=ldm_out.vf_last_layer,
                kl_loss=ldm_out.kl_loss,
                vq_loss=ldm_out.vq_loss,
                entropy_loss=ldm_out.entropy_loss,
                codebook_usages=ldm_out.codebook_usages,
                eq_scale_factor=eq_scale_factor,
                eq_angle_factor=eq_angle_factor,
                global_text_tokens=global_text_tokens,
            )
        else:
            return GeneratorForwardOutput(
                gen_img=gen_img,
                gen_multiscale_imgs=gen_multiscale_imgs,
                eq_scale_factor=eq_scale_factor,
                eq_angle_factor=eq_angle_factor,
                global_text_tokens=global_text_tokens,
            )
