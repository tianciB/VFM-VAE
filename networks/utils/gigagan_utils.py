# ------------------------------------------------------------------------------------------------------------------------------
# This file is adapted and modified from:
#   - https://github.com/lucidrains/gigagan-pytorch/blob/main/gigagan_pytorch/gigagan_pytorch.py
#
# Modifications:
# - Replaced the original custom attention with PyTorch's native attention for faster dot-product attention.
# - Added null key/value support in the cross-attention layer to allow the model to "attend to nothing."
# - See https://github.com/lucidrains/gigagan-pytorch/issues/14 for known issues regarding L2 attention in the generator.
# ------------------------------------------------------------------------------------------------------------------------------


import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat
from functools import partial


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim=1)
        return normed * self.scale * self.gamma


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = ChannelRMSNorm(dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim_inner, 1, bias=False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head) * 0.02)

        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)
        nn.init.zeros_(self.to_out.weight)   # zero init residual path

    def forward(self, fmap):
        B, _, H, W = fmap.shape
        h = self.heads

        fmap = self.norm(fmap)
        q, k, v = self.to_q(fmap), self.to_k(fmap), self.to_v(fmap)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=h), (q, k, v))

        nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b=B), self.null_kv)
        k = torch.cat((nk, k), dim=-2) # (b, h, 1 + H*W, d)
        v = torch.cat((nv, v), dim=-2)

        # dot product
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = H, y = W, h = h)
        
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head=64,
        heads=8
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads
        kv_input_dim = default(dim_context, dim)

        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(kv_input_dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_kv = nn.Linear(kv_input_dim, dim_inner * 2, bias=False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)
        nn.init.zeros_(self.to_out.weight)   # zero init residual path

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head) * 0.02)

    def forward(self, fmap, context, mask=None):
        fmap = self.norm(fmap)
        context = self.norm_context(context)

        x, y = fmap.shape[-2:]
        h = self.heads
        b = fmap.shape[0]

        q = self.to_q(fmap)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=h)

        k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (k, v))

        nk, nv = self.null_kv.unbind(dim=0)
        nk = repeat(nk, 'h d -> b h 1 d', b=b)
        nv = repeat(nv, 'h d -> b h 1 d', b=b)
        k = torch.cat([nk, k], dim=2)
        v = torch.cat([nv, v], dim=2)

        if exists(mask):
            batch = mask.shape[0]
            pad = torch.zeros(batch, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([pad, mask], dim=1)        # （b, 1 + l)
            mask = mask.unsqueeze(1).unsqueeze(2)       # (b, 1, 1, 1 + l)
            mask = mask.expand(-1, h, q.shape[2], -1)   # (b, h, q_len, 1 + l)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=x, y=y, h=h)
        return self.to_out(out)


def FeedForward(dim, mult=4, channel_first=False):
    dim_hidden = int(dim * mult)
    norm_klass = ChannelRMSNorm if channel_first else RMSNorm
    proj = partial(nn.Conv2d, kernel_size=1) if channel_first else nn.Linear

    # first and second projections
    proj1 = proj(dim, dim_hidden)
    proj2 = proj(dim_hidden, dim)
    # zero-init output projection for residual
    nn.init.zeros_(proj2.weight)

    layer = nn.Sequential(
        norm_klass(dim),
        proj1,
        nn.GELU(),
        proj2
    )

    return layer


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.attn = SelfAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult, channel_first=True)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head=64,
        heads=8,
        ff_mult=4
    ):
        super().__init__()
        self.attn = CrossAttention(dim=dim, dim_context=dim_context, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult, channel_first=True)

    def forward(self, x, context, mask=None):
        x = self.attn(x, context=context, mask=mask) + x
        x = self.ff(x) + x
        return x
