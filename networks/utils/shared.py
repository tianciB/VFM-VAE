# ------------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# ------------------------------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, fn: nn.Module):
        """
        A residual block that scales the skip connection by 1/sqrt(2).

        Args:
            fn: A neural network module to apply to the input.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / math.sqrt(2)


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = 'linear',
        lr_multiplier: float = 1.0,
        weight_init: float = 1.0,
        bias_init: float = 0.0,
    ):
        """
        A fully connected layer with optional bias, custom activation, and learning-rate multiplier.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include an additive bias.
            activation: Activation to apply: 'linear', 'relu', 'lrelu', 'gelu'.
            lr_multiplier: Factor to adjust weight and bias scales.
            weight_init: Std for weight initialization before scaling.
            bias_init: Initial bias value before scaling.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Initialize weight with gain and learning-rate multiplier.
        weight = torch.randn(out_features, in_features) * (weight_init / lr_multiplier)
        self.weight = nn.Parameter(weight)

        # Initialize bias if needed.
        if bias:
            bias_tensor = torch.full((out_features,), bias_init / lr_multiplier)
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.bias = None

        # Scaling factors.
        self.weight_gain = lr_multiplier / math.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Scale weight and bias.
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype) * self.bias_gain

        # Linear or activation path.
        if self.activation == 'linear':
            if b is not None:
                return torch.addmm(b.unsqueeze(0), x, w.t()) # x: [batch, in], w: [out, in]
            else:
                return x.matmul(w.t())
        else:
            # Compute linear output then bias and activation.
            x = x.matmul(w.t())
            if b is not None:
                x = x + b
            if self.activation == 'relu':
                return F.relu(x)
            elif self.activation == 'lrelu':
                return F.leaky_relu(x, negative_slope=0.2)
            elif self.activation == 'gelu':
                return F.gelu(x)
            else:
                raise NotImplementedError(f"Activation '{self.activation}' not implemented.")

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, activation={self.activation}'


class MLP(nn.Module):
    def __init__(
        self,
        features_list: list[int],
        activation: str = 'linear',
        lr_multiplier: float = 1.0,
        linear_out: bool = False
    ):
        """
        A simple MLP built from FullyConnectedLayer modules.

        Args:
            features_list: List of layer feature sizes, e.g. [in, hidden1, ..., out].
            activation: Activation for hidden layers.
            lr_multiplier: LR multiplier for each layer.
            linear_out: If True, last layer uses 'linear' activation.
        """
        super().__init__()
        num_layers = len(features_list) - 1
        self.num_layers = num_layers
        self.out_dim = features_list[-1]

        for idx in range(num_layers):
            in_f = features_list[idx]
            out_f = features_list[idx + 1]
            act = activation
            if linear_out and idx == num_layers - 1:
                act = 'linear'
            layer = FullyConnectedLayer(
                in_features=in_f,
                out_features=out_f,
                bias=True,
                activation=act,
                lr_multiplier=lr_multiplier
            )
            self.add_module(f'fc{idx}', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If input is BxKxC, merge batch and sequence dims.
        shift2batch = (x.ndim == 3)
        if shift2batch:
            B, K, C = x.shape
            x = x.flatten(0, 1)

        # Apply each FC layer.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Restore sequence shape.
        if shift2batch:
            x = x.reshape(B, K, -1)
        
        return x


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class StyleSplit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.proj = FullyConnectedLayer(in_channels, 3*out_channels, **kwargs)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        m1, m2, m3 = x.chunk(3, 1)
        return m1 * m2 + m3


class ScaleAdaptiveAvgPool2d(nn.Module):
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        target_h = max(1, int(H * self.scale_factor))
        target_w = max(1, int(W * self.scale_factor))
        return F.adaptive_avg_pool2d(x, (target_h, target_w))