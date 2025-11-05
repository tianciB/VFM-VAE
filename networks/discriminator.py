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
# This version includes minor modifications based on NVIDIA's StyleGAN-T.
# ------------------------------------------------------------------------------

"""
Projected discriminator architecture from
"Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models".
"""

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.spectral_norm import SpectralNorm
from torch_utils import distributed as dist
from torchvision.transforms import RandomCrop, Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from networks.utils.dataclasses import DiscriminatorForwardOutput
from networks.utils.shared import ResidualBlock, FullyConnectedLayer
from networks.utils.vit_utils import make_vit_backbone, forward_vit
from networks.utils.vfm_utils import VFM2INTERPOLATION
from training.diffaug import DiffAugment


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


# This is for PatchGAN discriminator.
class BatchNormLocal2d(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        B, C, H, W = shape
        G = int(np.ceil(B / self.virtual_bs))
        x = x.view(G, -1, C, H, W)

        mean = x.mean([1, 3, 4], keepdim=True)
        var = x.var([1, 3, 4], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x = x * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class DINO(torch.nn.Module):
    def __init__(
            self,
            hooks: list[int] = [2,5,8,11], # ascent order of hook depths
            hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)
        self.patch_size = 16

        self.model = make_vit_backbone(
            timm.create_model('vit_small_patch16_224_dino', pretrained=True, 
                              # pretrained_cfg_overlay=dict(file="huggingface/vit_small_patch16_224_dino/pytorch_model.bin")
                            ),
            patch_size=[self.patch_size, self.patch_size], hooks=hooks, hook_patch=hook_patch,
        )
        self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input: x in [0, 1]; output: dict of activations '''
        features = forward_vit(self.model, x)
        return features


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=BatchNormLocal2d, use_sigmoid=False, get_interm_feat=False):
        super(NLayerDiscriminator, self).__init__()
        self.get_interm_feat = get_interm_feat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf), nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if get_interm_feat:
            for n in range(len(sequence)):
                setattr(self, f'model{n}', nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.get_interm_feat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, f'model{n}')
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=BatchNormLocal2d, use_sigmoid=False, num_D=3, get_interm_feat=True):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.get_interm_feat = get_interm_feat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, get_interm_feat)
            if get_interm_feat:
                for j in range(n_layers + 2):
                    setattr(self, f'scale{i}_layer{j}', getattr(netD, f'model{j}'))
            else:
                setattr(self, f'layer{i}', netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.get_interm_feat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            if self.get_interm_feat:
                model = [getattr(self, f'scale{self.num_D - 1 - i}_layer{j}') for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, f'layer{self.num_D - 1 - i}')
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class ProjectedDiscriminator(nn.Module):
    def __init__(
            self,
            c_dim: int,
            vfm_name: str,
            # DINO ViT preprocessing settings.
            use_stylegan_t_discriminator: bool = True,
            diffaug: bool = True, 
            p_crop: float = 0.5, 
            # PatchGAN discriminator settings.
            use_patchgan_discriminator: bool = False,
            get_interm_feat: bool = False,
        ):
        super().__init__()

        # Determine which discriminator to use.
        self.use_stylegan_t_discriminator = use_stylegan_t_discriminator
        self.use_patchgan_discriminator = use_patchgan_discriminator
        dist.print0(f"[Manual Training] Using StyleGAN-T Discriminator: {self.use_stylegan_t_discriminator}, Using PatchGAN Discriminator: {self.use_patchgan_discriminator}")

        # StyleGAN-T Discriminator's settings.
        if self.use_stylegan_t_discriminator:
            # Preprocessing settings.
            self.diffaug = diffaug
            self.p_crop = p_crop
            self.vfm_name = vfm_name.lower() # align the preprocessing of images with the VFM name
            for name in VFM2INTERPOLATION.keys():
                if name in self.vfm_name:
                    self.interpolation = VFM2INTERPOLATION.get(name, 'bilinear')
                    break
            self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD) # take it out from DINO for readability
            
            # DINO ViT.
            self.dino = DINO()

            # DINO discriminator heads.
            self.c_dim = c_dim # the class conditional is only used for StyleGAN-T discriminator
            heads = [(str(i), DiscHead(self.dino.embed_dim, self.c_dim)) for i in range(self.dino.n_hooks)]
            self.heads = nn.ModuleDict(heads)

        # PatchGAN discriminator's settings.
        if self.use_patchgan_discriminator:
            self.get_interm_feat = get_interm_feat
            self.patchgan_discriminator = MultiscaleDiscriminator(input_nc=3, num_D=3, get_interm_feat=get_interm_feat)
            self.patchgan_discriminator.apply(weights_init)

    def train(self, mode: bool = True):
        if self.use_stylegan_t_discriminator:
            self.dino = self.dino.train(False)
            self.heads = self.heads.train(mode)
        if self.use_patchgan_discriminator:
            self.patchgan_discriminator = self.patchgan_discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def stylegan_t_forward(self, x: torch.Tensor, c_enc: torch.Tensor) -> torch.Tensor:
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Transform to [0, 1].
        x = (x + 1.0) / 2.0

        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
                x = RandomCrop(self.dino.img_resolution)(x)
        
        # Resize to DINO input resolution.
        if x.size(-1) < self.dino.img_resolution:
            x = F.interpolate(x, self.dino.img_resolution, mode=self.interpolation, align_corners=False)
        
        elif x.size(-1) > self.dino.img_resolution:
            x = F.interpolate(x, self.dino.img_resolution, mode=self.interpolation, align_corners=False, antialias=True)
    
        # ImageNet normalization.
        x = self.norm(x)
        
        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Compute logits from DINO discriminator heads.
        logits = [head(features[str(i)], c_enc).view(x.size(0), -1) for i, head in self.heads.items()]

        return torch.cat(logits, dim=1)

    def patchgan_forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input x is in [-1, 1].
        return self.patchgan_discriminator(x)

    def forward(self, x: torch.Tensor, c_enc: torch.Tensor) -> DiscriminatorForwardOutput:
        return DiscriminatorForwardOutput(
            stylegan_t_logits = self.stylegan_t_forward(x, c_enc) if self.use_stylegan_t_discriminator else None,
            patchgan_logits = self.patchgan_forward(x) if self.use_patchgan_discriminator else None,
        )
