# ------------------------------------------------------------------------------
# The following components are adapted from:
#   - PlainAttention, GeGluMlp, AttnProjection
#     https://github.com/FoundationVision/UniTok/blob/main/models/vqvae.py#L53
#
# All other components are original implementations.
# ------------------------------------------------------------------------------


import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from einops import rearrange
from functools import partial
try:
    from timm.layers import get_norm_layer
except ImportError:
    from timm.models.layers import get_norm_layer
from torch_utils import distributed as dist
from networks.utils.kl_utils import DiagonalGaussianDistribution
from networks.utils.quant_utils import VectorQuantizerM
from networks.utils.dataclasses import EncodeOutput


def init_weights(model, conv_std_or_gain):
    dist.print0(f'[init_weights] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if conv_std_or_gain > 0:
                nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            else:
                nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.)


class PlainAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x


class AttnProjectionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_layers, is_quant, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        if is_quant:
            self.blocks = nn.ModuleList([
                AttnProjectionBlock(in_dim, in_dim, num_heads, norm_layer, mlp_ratio)
                if i < num_layers - 1 else
                AttnProjectionBlock(in_dim, out_dim, num_heads, norm_layer, mlp_ratio)
                for i in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                AttnProjectionBlock(in_dim, out_dim, num_heads, norm_layer, mlp_ratio)
                if i == 0 else
                AttnProjectionBlock(out_dim, out_dim, num_heads, norm_layer, mlp_ratio)
                for i in range(num_layers)
            ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class GeneralPixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor: int, flatten_output: bool = False):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.flatten_output = flatten_output
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)

    def forward(self, x):
        if x.dim() == 4:
            x = self.pixel_unshuffle(x)
            if self.flatten_output:
                B, C, H, W = x.shape
                return x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            return x

        elif x.dim() == 3:
            B, HW, D = x.shape
            side = int(HW**0.5)
            assert side * side == HW
            x = x.permute(0, 2, 1).reshape(B, D, side, side)
            x = self.pixel_unshuffle(x)
            if self.flatten_output:
                B, C, H, W = x.shape
                return x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            return x

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


class LDMAdapter(nn.Module):
    def __init__(
        self,
        patch_from_layers: List[int],           # Indices of layers to extract patch features.
        patch_resolutions: List[int],           # Patch resolutions for each layer.
        patch_in_dimensions: List[int],         # Patch input dimensions for each layer.
        patch_out_dimensions: List[int],        # Patch output dimensions for each layer.
        compression_mode: str,                  # Compression mode: 'continuous' or 'discrete'.
        how_to_compress: str,                   # How to compress the input, 'conv' or 'attnproj'.
        how_to_decompress: str,                 # How to decompress the latent, 'conv' or 'attnproj'.
        decompress_factor: int,                 # Decompression factor for scaling up the latent dimension.
        attnproj_quant_layers: int,             # Number of attention projection layers for quantization.
        attnproj_post_quant_layers: int,        # Number of attention projection layers for post-quantization.
        z_resolution:int,                       # Latent resolution for quantization or compression.
        # Continuous arguments.
        z_dimension: int,                       # Latent dimension for continuous tokenzier.
        # Discrete arguments.
        vocab_width: int = 64,                  # Width of each token in discrete tokenizer.
        vocab_size: int = 32768,                # Vocabulary size for discrete tokenizer.
        vocab_beta: float = 0.25,               # Beta parameter for vector quantization.
        use_entropy_loss: bool = False,         # Whether to use entropy loss for discrete tokenizers.
        entropy_temp: float = 0.01,             # Temperature for entropy loss.
        num_codebooks: int = 8,                 # Number of codebooks for vector quantization.
        # Losses.
        use_kl_loss: bool = False,              # Whether to use KL loss for continuous tokenizers.
        # VF loss arguments.
        use_vf_loss: bool = False,              # Whether to use VF loss.
        use_adaptive_vf_loss: bool = False,     # Whether to use adaptive VF loss.
        distmat_margin: float = 0.0,            # Margin for distance matrix in VF loss.
        cos_margin: float = 0.0,                # Margin for cosine similarity in VF loss.
        distmat_weight: float = 1.0,            # Weight for distance matrix loss in VF loss.
        cos_weight: float = 1.0,                # Weight for cosine similarity loss in VF loss.
        ) -> None:
        super().__init__()

        assert len(patch_from_layers) == len(patch_resolutions) == len(patch_in_dimensions) == len(patch_out_dimensions), \
            f'Attention layers, patch resolutions, input dimensions, and output dimensions must have the same length.'
        assert all([res >= z_resolution for res in patch_resolutions]), \
            f'All patch resolutions must be greater than or equal to the latent resolution {z_resolution}.'
        assert all([res % z_resolution == 0 for res in patch_resolutions]), \
            f'All patch resolutions must be divisible by the latent resolution {z_resolution}.'
        assert all([in_dim >= out_dim for in_dim, out_dim in zip(patch_in_dimensions, patch_out_dimensions)]), \
            f'All patch input dimensions must be greater than or equal to the output dimensions.'
        self.patch_from_layers = patch_from_layers
        self.patch_resolutions = patch_resolutions
        self.patch_in_dimensions = patch_in_dimensions
        self.patch_out_dimensions = patch_out_dimensions

        self.compression_mode = compression_mode
        self.z_resolution = z_resolution
        self.z_dimension = z_dimension
        self.how_to_compress = how_to_compress
        self.how_to_decompress = how_to_decompress
        self.decompress_factor = decompress_factor
        
        assert -1 in patch_from_layers if use_vf_loss else True, \
            "VF loss requires the last layer (-1) to be included in patch_from_layers."
        self.use_kl_loss = use_kl_loss
        self.use_vf_loss = use_vf_loss
        self.use_adaptive_vf_loss = use_adaptive_vf_loss

        # Per-patch compression conv or attention projection.
        if how_to_compress == 'conv':
            self.patch_quants = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=patch_in_dimensions[i],
                        out_channels=patch_out_dimensions[i],
                        kernel_size=1,
                        bias=True
                    ),
                    GeneralPixelUnshuffle(
                        downscale_factor=patch_resolutions[i] // z_resolution,
                        flatten_output=False
                    ) if patch_resolutions[i] > z_resolution else nn.Identity()
                ) for i in range(len(patch_from_layers))
            ])
        
        elif how_to_compress == 'attnproj':
            self.patch_quants = nn.ModuleList([
                nn.Sequential(
                    AttnProjection(
                        in_dim=patch_in_dimensions[i],
                        out_dim=patch_out_dimensions[i],
                        num_heads=max(1, patch_in_dimensions[i] // patch_out_dimensions[i]),
                        num_layers=attnproj_quant_layers,
                        is_quant=True
                    ),
                    GeneralPixelUnshuffle(
                        downscale_factor=patch_resolutions[i] // z_resolution,
                        flatten_output=True
                    ) if patch_resolutions[i] > z_resolution else nn.Identity()
                ) for i in range(len(patch_from_layers))
            ])

        for m in self.patch_quants:
            if isinstance(m, nn.Sequential):
                init_weights(m[0], conv_std_or_gain=-0.5)
            else:
                init_weights(m, conv_std_or_gain=-0.5)
        dist.print0(f'[LDMAdapter] Patch quantization: {len(self.patch_quants)} layers with {how_to_compress} mode.')
        
        # Global compression conv or attention projection after concatenation.
        final_in_dimension = 0
        for i in range(len(patch_resolutions)):
            if patch_resolutions[i] > z_resolution:
                final_in_dimension += patch_out_dimensions[i] * (patch_resolutions[i] // z_resolution) ** 2
            else:
                final_in_dimension += patch_out_dimensions[i]
        final_out_dimension = z_dimension * 2 if compression_mode == 'continuous' else vocab_width
        
        if how_to_compress == 'conv':
            self.final_quant = nn.Conv2d(
                in_channels=final_in_dimension,
                out_channels=final_out_dimension,
                kernel_size=1,
                bias=True
            )

        elif how_to_compress == 'attnproj':
            self.final_quant = AttnProjection(
                in_dim=final_in_dimension,
                out_dim=final_out_dimension,
                num_heads=max(1, final_in_dimension // final_out_dimension),
                num_layers=attnproj_quant_layers,
                is_quant=True
            )
        
        init_weights(self.final_quant, conv_std_or_gain=-0.5)
        dist.print0(f'[LDMAdapter] Final projection: {final_in_dimension} -> {final_out_dimension}.')

        # Post-decompression conv or attention projection.
        in_ch = z_dimension if compression_mode == 'continuous' else vocab_width
        out_ch = in_ch * decompress_factor
        
        if how_to_decompress == 'conv':
            self.post_quant = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                bias=True
            )
        
        elif how_to_decompress == 'attnproj':
            self.post_quant = AttnProjection(
                in_dim=in_ch,
                out_dim=out_ch,
                num_heads=max(1, out_ch // in_ch),
                num_layers=attnproj_post_quant_layers,
                is_quant=False
            )
        
        init_weights(self.post_quant, conv_std_or_gain=-0.5)
        dist.print0(f'[LDMAdapter] Post quantization: {in_ch} -> {out_ch} with {how_to_decompress} mode.')
        
        # Vector Quantizer for discrete mode.
        if self.compression_mode == 'discrete':
            self.quantizer = VectorQuantizerM(
                vocab_size=vocab_size,
                vocab_width=vocab_width,
                beta=vocab_beta,
                use_entropy_loss=use_entropy_loss,
                entropy_temp=entropy_temp,
                num_codebooks=num_codebooks,
            )
            self.quantizer.init_vocab(eini=-1)
        else:
            self.quantizer = None

        # VF loss.
        if self.use_vf_loss:
            assert -1 in patch_from_layers, "VF loss requires the last layer to be included in patch_from_layers."
            vf_dim = patch_in_dimensions[patch_from_layers.index(-1)] # dimension of the last layer for VF loss
            in_dim = z_dimension if compression_mode == 'continuous' else vocab_width
            self.linear_proj = nn.Conv2d(in_dim, vf_dim, 1, bias=False)
            init_weights(self.linear_proj, conv_std_or_gain=-0.5)
            self.distmat_margin = distmat_margin
            self.cos_margin = cos_margin
            self.distmat_weight = distmat_weight
            self.cos_weight = cos_weight
        else:
            self.linear_proj = None

    # ------------------------------------------------------------------
    # Internal helper – VF loss
    # ------------------------------------------------------------------
    def _compute_vf_loss(self, z: torch.Tensor, aux_feature: torch.Tensor) -> torch.Tensor:
        z_flat = rearrange(z, 'b c h w -> b c (h w)')
        aux_feature_flat = rearrange(aux_feature, 'b c h w -> b c (h w)')
        z_norm = F.normalize(z_flat, dim=1)
        aux_norm = F.normalize(aux_feature_flat, dim=1)
        z_cos = torch.einsum('bci,bcj->bij', z_norm, z_norm)
        aux_cos = torch.einsum('bci,bcj->bij', aux_norm, aux_norm)
        diff = torch.abs(z_cos - aux_cos)
        vf_loss_1 = F.relu(diff - self.distmat_margin).mean()
        vf_loss_2 = F.relu(1 - self.cos_margin - F.cosine_similarity(aux_feature, z)).mean()
        return vf_loss_1 * self.distmat_weight + vf_loss_2 * self.cos_weight

    # ------------------------------------------------------------------
    # encode()
    # ------------------------------------------------------------------
    def encode(self, patch_features: List[torch.Tensor], return_z_before_quantize: bool=False) -> EncodeOutput:
        assert len(patch_features) == len(self.patch_quants), \
            f'Number of patch features ({len(patch_features)}) must match the number of patch quantizers ({len(self.patch_quants)}).'
        
        # Per-patch compression.
        middle_features = []
        for x, proj in zip(patch_features, self.patch_quants):
            if self.how_to_compress == 'conv':
                Hi = int(x.shape[1] ** 0.5)
                x = rearrange(x, 'b (h w) d -> b d h w', h=Hi, w=Hi)  # [B, L, Hi, Wi]
                x = proj(x)                                           # [B, D, Ht, Wt] after conv and unshuffle
                x = rearrange(x, 'b d h w -> b (h w) d')              # [B, Ht*Wt, D]                  
            
            elif self.how_to_compress == 'attnproj':
                x = proj(x)                                           # [B, Ht*Wt, D] after attention projection and unshuffle
            
            middle_features.append(x)
        
        # Concatenate all patch features.
        x = torch.cat(middle_features, dim=-1)
        
        # Final compression. 
        if self.how_to_compress == 'conv':
            Ht = int(x.shape[1] ** 0.5)
            x = rearrange(x, 'b (h w) d -> b d h w', h=Ht, w=Ht)    # [B, D, Ht, Wt]
            x = self.final_quant(x)                                 # [B, D', Ht, Wt]

        elif self.how_to_compress == 'attnproj':
            x = self.final_quant(x)                                 # [B, Ht*Wt, D']
            Ht = int(x.shape[1] ** 0.5)
            x = rearrange(x, 'b (h w) d -> b d h w', h=Ht, w=Ht)    # [B, D', Ht, Wt]

        # Compute quantization losses.
        vq_loss = entropy_loss = usages = kl_loss = 0.0
        z_before_quantize = x

        if self.compression_mode == 'continuous':
            distribution = DiagonalGaussianDistribution(x)
            z = distribution.sample()
            if self.use_kl_loss:
                kl_loss = distribution.kl().mean()

        elif self.compression_mode == 'discrete':
            tokens = rearrange(x, 'b d h w -> b (h w) d')                       # [B, Ht*Wt, D']
            z_tokens, vq_loss, entropy_loss, usages = self.quantizer(tokens)
            Ht = int(z_tokens.shape[1] ** 0.5)
            z = rearrange(z_tokens, 'b (h w) d -> b d h w', h=Ht, w=Ht)         # [B, D', Ht, Wt] after codebook quantization
    
        # Optional VF loss.
        vf_loss = 0.0
        vf_last_layer = None
        if self.use_vf_loss:
            aux = patch_features[self.patch_from_layers.index(-1)].detach().clone()  # last layer features
            Ha = int(aux.shape[1] ** 0.5)
            aux = rearrange(aux, 'b (h w) d -> b d h w', h=Ha, w=Ha)
            Ht = int(z.shape[2])  # Ht is the height of the quantized latent
            if Ha != Ht:
                aux = F.adaptive_avg_pool2d(aux, (Ht, Ht))
            z_vf = self.linear_proj(z)
            vf_loss = self._compute_vf_loss(z_vf, aux)

            if self.use_adaptive_vf_loss:
                if self.how_to_compress == 'conv':
                    vf_last_layer = self.final_quant.weight
                elif self.how_to_compress == 'attnproj':
                    vf_last_layer = self.final_quant.blocks[-1].mlp.w2.weight

        return EncodeOutput(
            z=z if not return_z_before_quantize else z_before_quantize,
            vf_loss=vf_loss,
            vf_last_layer=vf_last_layer,
            kl_loss=kl_loss,
            vq_loss=vq_loss,
            entropy_loss=entropy_loss,
            codebook_usages=usages
        )

    # ------------------------------------------------------------------
    # decode()
    # -----------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        H, W = z.shape[-2:]
        if self.how_to_decompress == 'conv':
            z = self.post_quant(z)
        elif self.how_to_decompress == 'attnproj':
            z = rearrange(z, 'b d h w -> b (h w) d', h=H, w=W)
            z = self.post_quant(z)
            z = rearrange(z, 'b (h w) d -> b d h w', h=H, w=W)
        return z


class EquivarianceTransform(nn.Module):
    def __init__(
        self,
        apply: bool = False,
        p_eq_prior: float = 0.5,
        p_eq_prior_scale: float = 0.25
    ):
        super().__init__()
        self.apply = apply
        self.p_eq_prior = p_eq_prior
        self.p_eq_prior_scale = p_eq_prior_scale

    def forward(self, validation: bool):
        if not self.apply or validation:
            return 1.0, 0, False # no equivariance regularization during validation

        if random.random() < self.p_eq_prior:
            eq_scale_factor = random.choice([0.25, 0.5, 0.75, 1.0])
            eq_angle_factor = random.choice([0, 1, 2, 3])  # 0: 0 degrees, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
            is_eq_prior = False
        
        else:
            eq_scale_factor = random.choice([0.25, 0.5, 0.75]) if random.random() < self.p_eq_prior_scale else 1.0
            eq_angle_factor = 0
            is_eq_prior = True

        return eq_scale_factor, eq_angle_factor, is_eq_prior