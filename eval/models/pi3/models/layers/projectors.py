from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union
import math
from cv2 import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend

from .attention import FlashAttentionRope, MemEffAttentionRope
from ..dinov2.layers import Mlp
from ..dinov2.layers.layer_scale import LayerScale

class MLPProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # Init: normal first layer, zero second layer
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # initially outputs all zeros
        return x
    
class ZeroConvNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=1024, kernel_size=14):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=kernel_size)

        # Zero init for all weights & biases
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    "Redidual block for Dense Representation Encoder"

    def __init__(self, in_channels: int, out_channels: int, act_layer: Type[nn.Module] = nn.GELU):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out += identity

        return self.act(out)

class DenseCondProjector(nn.Module):
    # adopted from https://github.com/castacks/UniCeption/blob/main/uniception/models/encoders/dense_rep_encoder.py
    def __init__(
        self,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        patch_size: int = 14,
        intermediate_dims: List[int] = [588, 768, 1024],
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Callable] = partial(nn.LayerNorm, eps=1e-6),
        pretrained_checkpoint_path: str = None,
    ):
        # Init the base class
        super().__init__()

        # Init the specific attributes
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.enc_embed_dim = enc_embed_dim
        self.intermediate_dims = intermediate_dims

        # Initialize the encoder with a pixel unshuffle and conv projection to patchify the input
        self.unshuffle = nn.PixelUnshuffle(self.patch_size)
        self.conv_in = nn.Conv2d(self.in_chans * (self.patch_size**2), self.intermediate_dims[0], 3, 1, 1)

        # Add residual blocks
        layers = []
        for intermediate_idx in range(len(self.intermediate_dims) - 1):
            layers.append(
                ResidualBlock(
                    in_channels=self.intermediate_dims[intermediate_idx],
                    out_channels=self.intermediate_dims[intermediate_idx + 1],
                    act_layer=act_layer,
                )
            )

        # Final projection to match encoder embeddings dim
        layers.append(
            nn.Conv2d(
                in_channels=self.intermediate_dims[-1],
                out_channels=self.enc_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.encoder = nn.Sequential(*layers)

        # Init norm layer after encoder if required
        self.norm_layer = norm_layer(enc_embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

        # Load the pretrained checkpoint if provided
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path:
            print(
                f"Loading custom pretrained Dense Representation Encoder checkpoint from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # Check the dtype and shape of the input
        assert isinstance(input_data, torch.Tensor), "Input must be a torch.Tensor"
        assert input_data.ndim == 4, "Input must be of shape (B, C, H, W)"
        assert input_data.shape[1] == self.in_chans, f"Input channels must be {self.in_chans}"
        batch_size, channels, height, width = input_data.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.patch_size}"

        # Encode the dense representation
        features = self.unshuffle(input_data)
        features = self.conv_in(features)
        features = self.encoder(features)
        features = features.flatten(2).transpose(
            1, 2
        )  # (B, E, H / Patch_Size, W / Patch_Size) -> (B, H / Patch_Size * W / Patch_Size, E)
        features = self.norm_layer(features)  # Normalize the features after patch encoding

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return features

class DescProjector(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 rope):
        super().__init__()
        num_heads = 16
        mlp_ratio = 4

        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        self.norm1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.ls1 = LayerScale(input_dim, init_values=0.01)
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-6)
        self.ls2 = LayerScale(input_dim, init_values=0.01)

        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(input_dim, input_dim, bias=True)
        self.kv = nn.Linear(input_dim, input_dim * 2, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.attn_drop_rate = 0.0
        self.proj = nn.Linear(input_dim, input_dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)

        self.rope = rope

        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            out_features=output_dim,
            act_layer=nn.GELU,
            drop=0.0,
            bias=True,
        )

    def forward(self, x, xpos=None):
        x = self.norm1(x)

        B, N, C = x.shape
        # import ipdb; ipdb.set_trace() # check the shape of x, make sure that the shape corresponds to the shape of self attention, where N is the patch num per image
        repeated_q = self.norm1(self.query.expand(B, -1, -1))
        q = self.q(repeated_q).reshape(B, 1, 1, self.num_heads, C // self.num_heads).transpose(1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = q[:,:,0], kv[:,:,0], kv[:,:,1]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            k = self.rope(k, xpos)

        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)

        x = x.transpose(1, 2).reshape([B, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.ls2(self.mlp(self.norm2(self.ls1(x))))
        
        x = F.normalize(x, p=2, dim=-1)

        return x