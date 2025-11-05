# ------------------------------------------------------------------------------
# ConvNeXt Utilities
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Historical Note:
#   The `noise_const` buffer in ConvNeXtSynthesisLayer is a legacy artifact 
#   inherited from early StyleGAN-T initialization, where static per-resolution
#   noise maps were injected into feature maps to improve fine-grained texture.
#
#   In VFM-VAE, this buffer is **retained only for compatibility** with pretrained
#   checkpoints (e.g., 256×256 models) and will be auto-resized via bilinear
#   interpolation when loading weights at different resolutions.
#
#   For modern dynamic-resolution pipelines, **retraining from scratch** should
#   disable this mechanism entirely:
#
#       use_convnext_noise: False
#
#   This improves resolution generalization and avoids unintended local biases.
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
from networks.utils.shared import GroupNorm32, StyleSplit


def modulated_pointwise_conv2d(x, weight, style, bias=None, demodulate=True):
    B, I, H, W = x.shape
    O = weight.shape[0]

    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / I) ** 0.5 / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True)
        style = style / style.norm(float('inf'), dim=1, keepdim=True)

    w = weight.unsqueeze(0) * style.view(B, 1, I, 1, 1)

    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w = w * dcoefs.view(B, O, 1, 1, 1)

    w = w.view(B * O, I, 1, 1)
    x = x.contiguous().view(1, B * I, H, W)
    x = F.conv2d(x, w, padding=0, groups=B)
    x = x.view(B, O, H, W)

    if bias is not None:
        x = x + bias
    return x


class ModulatedPointwiseConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.empty([out_channels, in_channels, 1, 1]))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        # Initialize pointwise convolution weights and biases.
        trunc_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x, style):
        return modulated_pointwise_conv2d(x, self.weight, style, self.bias, self.demodulate)


class ConvNeXtSynthesisLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        w_dim: int,
        kernel_size: int,
        channels_last: bool = False,
        layer_scale_init: float = 1e-5,
        demodulate: bool = True,
        block_index: int = 0,
        legacy: bool = False,
    ):
        super().__init__()
        self.legacy = legacy

        self.channels = channels
        self.kernel_size = kernel_size

        memory_format = torch.channels_last if channels_last else torch.contiguous_format

        self.affine_pw1 = StyleSplit(w_dim, channels, bias_init=1)

        # Depthwise conv with bias.
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels)
        trunc_normal_(self.dwconv.weight, std=0.02)
        nn.init.constant_(self.dwconv.bias, 0)

        # Legacy noise const buffer for compatibility with pretrained checkpoints.
        if self.legacy:
            resolution = 8 * 2 ** block_index # 256px starts from resolution 8
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.zeros([]))

        self.pwconv1 = ModulatedPointwiseConv2DLayer(channels, 4 * channels, demodulate)
        self.pwconv2 = nn.Conv2d(4 * channels, channels, kernel_size=1)
        trunc_normal_(self.pwconv2.weight, std=0.02)
        nn.init.zeros_(self.pwconv2.bias)

        self.norm = GroupNorm32(min(32, channels // 4), channels)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones([1, channels, 1, 1]).to(memory_format=memory_format)
        ) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_in = x
        style_pw1 = self.affine_pw1(w)

        x = self.dwconv(x)
        
        if self.legacy:
            _, _, H, W = x.shape
            noise = self.noise_const[None, None] * self.noise_strength
            noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
            x = x + noise

        x = self.norm(x)
        x = self.pwconv1(x, style_pw1)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        return (x + x_in).to(dtype)


class ConvNeXtToRGBLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        kernel_size: int = 1,
        channels_last: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.channels_last = channels_last

        memory_format = torch.channels_last if channels_last else torch.contiguous_format

        # Initialize convolution weights.
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1).to(memory_format=memory_format)

        # Optimized 4D bias for direct broadcasting.
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1)).to(memory_format=memory_format)

        assert w_dim > 0, "w_dim must be set when use_style=True"
        self.affine = StyleSplit(w_dim, in_channels, bias_init=1)
        self.weight_gain = 1 / np.sqrt(in_channels * kernel_size ** 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        B, C, H, W = x.shape
        weight = self.weight
        bias = self.bias

        # Style modulation.
        style = self.affine(w) * self.weight_gain           # [B, C]
        w_mod = weight[None] * style.view(B, 1, -1, 1, 1)   # [B, Cout, Cin, k, k]

        # Grouped conv.
        x = x.contiguous().view(1, B * C, H, W)
        w_mod = w_mod.reshape(B * self.out_channels, C, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, w_mod, bias=None, stride=1, padding=0, groups=B) # not using bias
        x = x.reshape(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = x + bias
        return x


GAUSSIAN_KERNELS = {
    "3x3": [1, 2, 1],       # for low-resolution images, e.g., 32x32
    "4x4": [1, 3, 3, 1],    # the same as StyleGAN-T, but causes shift 0.5 pixels when padded
    "5x5": [1, 4, 6, 4, 1], # for high-resolution images, e.g., 256x256
}


class SeparableUpsampleWithFixedBlur(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 2,
        blur_kernel: str | list[int] = "3x3",
        blur_normalize: bool = True,
        pad_mode: str = "replicate",
        pre_normalize: bool = True,
        use_gaussian_blur: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre_normalize = pre_normalize
        self.use_gaussian_blur = use_gaussian_blur
        self.upscale_factor = upscale_factor
        self.pad_mode = pad_mode

        # -------- norm / depthwise / pointwise / shuffle -------- #
        self.norm = nn.GroupNorm(min(32, in_channels // 4), in_channels) if pre_normalize else \
                    nn.GroupNorm(min(32, out_channels // 4), out_channels)

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels * upscale_factor ** 2, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(upscale_factor)

        # -------- blur kernel -------- #
        if self.use_gaussian_blur:
            if isinstance(blur_kernel, str):
                blur_kernel = GAUSSIAN_KERNELS[blur_kernel]
            kernel = torch.tensor(blur_kernel, dtype=torch.float32)
            kernel2d = kernel[:, None] * kernel[None, :]
            if blur_normalize:
                kernel2d /= kernel2d.sum()

            kH, kW = kernel2d.shape
            pad_h, pad_w = (kH - 1) // 2, (kW - 1) // 2
            extra_h = int(kH % 2 == 0)
            extra_w = int(kW % 2 == 0)
            self.pad = (pad_w, pad_w + extra_w, pad_h, pad_h + extra_h)
            blur_weight = kernel2d[None, None].repeat(out_channels, 1, 1, 1)
            self.register_buffer("blur_weight", blur_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_normalize:
            x = self.norm(x)
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.shuffle(x)
        else:
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.shuffle(x)
            x = self.norm(x)

        if self.use_gaussian_blur:
            x = F.pad(x, self.pad, mode=self.pad_mode)
            x = F.conv2d(x, self.blur_weight, groups=self.out_channels)

        return x