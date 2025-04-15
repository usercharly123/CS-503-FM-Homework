# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Some functions are based on the timm and 4M code bases
# https://github.com/huggingface/pytorch-image-models
# https://github.com/apple/ml-4m
# --------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    """Custom implementation of LayerNorm with the option to disable the bias term."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer("bias", torch.zeros(normalized_shape))

        # Normalized shape must be a tuple for F.layer_norm
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)


class Mlp(nn.Module):
    """
    MLP module with GELU activation.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (optional)
        out_features: Number of output features (optional)
        bias: Whether to include bias in the linear layers
    """
    def __init__(self, 
            in_features: int, 
            hidden_features: Optional[int] = None, 
            out_features: Optional[int] = None, 
            bias: bool = False,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(self, dim: int, head_dim: int = 64, qkv_bias: bool = False, proj_bias: bool = False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # TODO: Define here the linear layer(s) producing K, Q, V from the input x
        # Hint: Do you need to define three different projections, or can you use a single one for all three?
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.head_dim = head_dim

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape # Batch size, sequence length, and dimension

        # TODO: Compute the keys K, queries Q, and values V from x. Each should be of shape [B num_heads L head_dim].
        q, k, v = self.qkv(x).chunk(3, dim=-1) # Split the output into Q, K, V
        q = rearrange(q, "B L (num_heads head_dim) -> B num_heads L head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        k = rearrange(k, "B L (num_heads head_dim) -> B num_heads L head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        v = rearrange(v, "B L (num_heads head_dim) -> B num_heads L head_dim", num_heads=self.num_heads, head_dim=self.head_dim)

        # TODO: Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads L L].
        # Hint: Use the already defined self.scale
        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale # [B num_heads L L]

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # TODO: Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # TODO: Compute the softmax over the last dimension
        attn = attn.softmax(dim=-1)

        # TODO: Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B L D]
        x = torch.einsum("b h i j, b h j d -> b h i d", attn, v) # [B num_heads L head_dim]
        x = rearrange(x, "b h i d -> b i (h d)", h=self.num_heads) # [B L D]

        # Output projection
        x = self.attn_out_proj(x)
        return x


class Block(nn.Module):
    """
    Basic transformer block with a multi-head self-attention mechanism and a feed-forward MLP.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: float = 4., use_bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim) # TODO (use the LayerNorm defined above)
        self.attn = Attention(dim, head_dim, qkv_bias=use_bias, proj_bias=use_bias)
        self.norm2 = LayerNorm(dim) # TODO (use the LayerNorm defined above)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = self.norm1(x)
        x = self.attn(x, mask=mask) + x # Residual connection
        x = self.norm2(x)
        x = self.mlp(x) + x # Residual connection
        return x


class TransformerTrunk(nn.Module):
    """Basic Transformer trunk definition that can be used for encoder-only,
    decoder-only and prefixLM models, depending on the attention mask applied.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
        ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_bias=use_bias) for _ in range(depth)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(x, mask=mask)
        return x
