"""Multi-Scale Patch Embedding (MSPE) based on: "Liu et al., https://arxiv.org/abs/2405.18240.
   Following MONAI embedding API.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import trunc_normal_, Conv
from monai.utils import ensure_tuple_rep
from typing import List, Tuple, Sequence

DEFAULT_RESOLUTIONS_ViT = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 448] # table 1 of MSPE paper
DEFAULT_RESOLUTIONS_SwinUNETRv2 = [64, 96, 128, 160, 192, 224, 256, 320] # multiples of 32, len() = 8

def pi_resize(w, size_new):
    """Pseudo inverse (PI)-resize for 2D/3D from FlexiViT paper <https://arxiv.org/abs/2212.08013>.
    Adapted from FlexiViT source:
    <https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py> 

    Args:
        w: original kernel weights.
           w dims (2D): (out_channels, in_channels, old_kH, old_kW)
           w dims (3D): (out_channels, in_channels, old_kD, old_kH, old_kW)
        size_new: target spatial size (new_kH, new_kW) or (new_kD, new_kH, new_kW).

    Returns:
        resized weights.
    """
    assert w.ndim in (4, 5), f"four or five dimensions expected"
    assert len(size_new) in (2, 3), f"new shape should either be 2D or 3D"

    old_size = w.shape[2:]

    # if dims already match do nothing
    if old_size == tuple(size_new):
        return w

    mode = "bilinear" if len(size_new) == 2 else "trilinear"

    # create resize matrix B 
    n_old = math.prod(old_size)
    eye = torch.eye(n_old).reshape(n_old, 1, *old_size) 
 
    B = F.interpolate(eye, size=size_new, mode=mode,  
                      align_corners=False).reshape(n_old, -1).T # (n_old, n_new)

    P = torch.linalg.pinv(B.T) # (n_new, n_old)
    w_flat = w.reshape(*w.shape[:2], -1).float() # (O, I, n_old)
    w_new = (w_flat @ P.to(w.device).T).reshape(*w.shape[:2], *size_new)

    return w_new.to(w.dtype)

# classic bilinear/trilinear img resize 
def img_resize(img, size):
    spatial_dims = img.ndim - 2
    mode = "bilinear" if spatial_dims == 2 else "trilinear"
    if isinstance(size, int):
        size = [size] * spatial_dims
    return F.interpolate(img, size=tuple(size), mode=mode, align_corners=False)

# get effective resolution
def get_effective_resolution(spatial_shape):
    return int(round(math.pow(math.prod(spatial_shape), 1.0 / len(spatial_shape))))

# get the closest resolution from DEFAULT_RESOLUTIONS_ to current resolution
def find_nearest_resolution(r_star, resolutions):
    best_idx = 0
    best_dist = float("inf")
    for i, r in enumerate(resolutions):
        d = abs(r - r_star)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return resolutions[best_idx], best_idx


# ViT embedding block
class MSPEPatchEmbedd(nn.Module):
    """Multi-Scale Patch Embedding (MSPE).

    Replaces the default single convolution in MONAI ViT PatchEmbeddingBlock 
    with K learnable convolution kernels of different resolutions. 

    Args:
        in_channels: dimension of input channels.
        img_size: dimension of input image.
        patch_size: dimension of patch size.
        hidden_size: dimension of hidden layer.
        num_heads: number of attention heads.
        pos_embed_type: position embedding layer type.
        dropout_rate: fraction of the input units to drop.
        spatial_dims: number of spatial dimensions. 
        K: number of multi-res patching kernels. 
        resolutions: full list of training resolutions.
    """
   
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int, 
        pos_embed_type: str = "learnable", 
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
        K: int = 4, #  paper recomednation
        resolutions: List[int] = DEFAULT_RESOLUTIONS_ViT,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        self.spatial_dims = spatial_dims
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)

        for m, p in zip(self.img_size, self.patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")

        self.in_channels = in_channels         
        self.hidden_size = hidden_size       
        self.K = K                       
        self.resolutions = sorted(resolutions)  
        # number of patches along ONE spatial dim (square/cube assumption)
        self.N: int = self.img_size[0] // self.patch_size[0] 

        # Add positional embedding and dropout
        self.pos_embed_type = pos_embed_type
        self.n_patches = self.N ** spatial_dims
        self.dropout = nn.Dropout(dropout_rate)

        if self.pos_embed_type == "none": 
            self.position_embeddings = None  # clear preallocated params 
        elif self.pos_embed_type == "learnable": # ViT case
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.pos_embed_type == "sincos": 
            pe = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
            with torch.no_grad():
                grid_size = [self.N] * spatial_dims
                pos_embeddings = build_sincos_position_embedding(grid_size, hidden_size, spatial_dims)
                pe.data.copy_(pos_embeddings.float())
            self.position_embeddings = pe
        else:
            raise ValueError(
                f"pos_embed_type '{pos_embed_type}' not supported. "
                "Choose from 'none', 'learnable', 'sincos'."
            )
        # add K patching kernels of different sizes
        self.patch_kernels = nn.ModuleList()
        for i in range(K):
            scale_i = (i + 1) / K # scale factors 1/K, 2/K, ... 1.0 K for conv kernels
            k_size = tuple(max(1, int(p * scale_i)) for p in self.patch_size)
            conv = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=k_size,
                stride=k_size,  # non-overlapping as in  ViT
                bias=True,
            )
            self.patch_kernels.append(conv)

        self.apply(self._init_weights)

    # standard ViT weight initialization
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adp_conv(
        self,
        img,
        func_idx,
        hw,
    ):
        """Adaptive convolution for inference.

        1. Selecet the kernel weights from list of all kernels generated by K patch_kernels.
        2. PI-resize the weights to (hw // N, hw // N) so that the encoder input has N x N tokens.
        3. Do convolution with adapted weights.

        Args:
            img:      Input image batch, already resized to hw x hw.
                      dim: (B, in_channels, hw, hw)
            func_idx: What kernel size generated by K patch_kernels to use.
            hw:       Current spatial resolution.

        Returns:
            Patch embeddings.
            dim: (B, hidden_size, N, N)
        """

        conv_layer = self.patch_kernels[func_idx]

        w = conv_layer.weight
        b = conv_layer.bias

        # Target kernel size
        k_size = tuple(max(1, hw // self.N) for _ in range(self.spatial_dims))

        w_star = pi_resize(w, k_size)
        if self.spatial_dims == 2:
            out = F.conv2d(img, w_star, bias=b, stride=k_size) 
        elif self.spatial_dims == 3:
            out = F.conv3d(img, w_star, bias=b, stride=k_size)
            
        return out

    def _resample_pos_embed(self, pos_embed, target_shape):
        """Resample the base position embedding to the target shape."""
        B, N_patches, C = pos_embed.shape
        N_old = int(round(math.pow(N_patches, 1.0 / self.spatial_dims)))

        if tuple(target_shape) == tuple([N_old] * self.spatial_dims):
            return pos_embed
            
        if self.spatial_dims == 2:
            pos_embed = pos_embed.reshape(B, N_old, N_old, C).permute(0, 3, 1, 2)
            resampled = F.interpolate(
                pos_embed, 
                size=target_shape, 
                mode="bilinear", 
                align_corners=False
            )
        else:
            pos_embed = pos_embed.reshape(B, N_old, N_old, N_old, C).permute(0, 4, 1, 2, 3)
            resampled = F.interpolate(
                pos_embed, 
                size=target_shape, 
                mode="trilinear", 
                align_corners=False
            )
        return resampled.flatten(2).transpose(1, 2)

    def forward(self, x):
        """Compute multi-scale patch embeddings.

        Given an image, forward picks the best kernel, PI-resizes its weights, 
        and produces the embedding.
    
        Args:
            x: image batch (B, in_channels, H, W)

        Returns:
            Patch embeddings (B, N_patches, hidden_size) 
        """
        spatial_shape = x.shape[2:]
        hw_eff = get_effective_resolution(spatial_shape)

        # Get the nearest resolution
        _, res_idx = find_nearest_resolution(hw_eff, self.resolutions)

        # map sorted resolution index to kernel index 
        func_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1) 
        conv_layer = self.patch_kernels[func_idx]

        w = conv_layer.weight
        b = conv_layer.bias

        # target kernel size
        k_size = tuple(max(dim // self.N, 1) for dim in spatial_shape)
        w_star = pi_resize(w, k_size)
        
        if self.spatial_dims == 2:
            out = F.conv2d(x, w_star, bias=b, stride=k_size)
        else:
            out = F.conv3d(x, w_star, bias=b, stride=k_size)
            
        out_spatial_shape = out.shape[2:]
        out = out.flatten(2).transpose(-1, -2) # (B, N_patches, hidden_size)
            
        if self.position_embeddings is not None:
            pos_embed = self._resample_pos_embed(self.position_embeddings, out_spatial_shape)
            out = out + pos_embed
            
        out = self.dropout(out)
            
        return out

    def forward_k(self, img, func_idx, hw):
        """Forward pass using a specific kernel k at resolution hw.

        Like forward(), but uses only the specified kernel instead of auto select.
        Used for multi-resolution training (Algorithm 1 in the MSPE paper).

        Args:
            img:      Input image batch, already resized to hw x hw.
                      dim: (B, in_channels, hw, hw)
            func_idx: Index of the kernel to use from patch_kernels.
            hw:       Current spatial resolution.

        Returns:
            Patch embeddings (B, N_h * N_w, hidden_size)
        """
        z = self.adp_conv(img, func_idx=func_idx, hw=hw)
        out_spatial_shape = z.shape[2:]
        z = z.flatten(2).transpose(-1, -2)

        if self.position_embeddings is not None:
            pos_embed = self._resample_pos_embed(self.position_embeddings, out_spatial_shape)
            z = z + pos_embed
        z = self.dropout(z)
        return z

    # divide resolutions into K subsets for even kernels train
    def sample_resolutions(self):
        subsets = [[] for _ in range(self.K)]
        for res_idx, r in enumerate(self.resolutions):
            func_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1)
            subsets[func_idx].append(r)
            
        hw_list = []
        for subset in subsets:
            if subset:
                hw_list.append(int(np.random.choice(subset)))
            else:
                hw_list.append(int(np.random.choice(self.resolutions)))
        return hw_list

    # for printing mspe embeddings
    def extra_repr(self) -> str:
        kernel_sizes = []
        for i in range(self.K):
            c = self.patch_kernels[i]
            kernel_sizes.append(f"{c.kernel_size}")
        return (
            f"spatial_dims={self.spatial_dims}, patch_size={self.patch_size}, in_channels={self.in_channels}, "
            f"hidden_size={self.hidden_size}, img_size={self.img_size}, "
            f"K={self.K}, N={self.N}, "
            f"kernel_sizes=[{', '.join(kernel_sizes)}], "
            f"resolutions={self.resolutions}"
        )


# Swin embedding block
class MSPEPatchEmbedSwin(nn.Module):
    """Multi-Scale Patch Embedding (MSPE) for SwinUNETR.
    Args:
        patch_size: dimension of patch size.
        in_chans: dimension of input channels.
        embed_dim: number of linear projection output channels.
        norm_layer: normalization layer.
        spatial_dims: number of spatial dimensions.
        K: number of multi-res patching kernels.
        resolutions: full list of training resolutions.
        img_size: base image resolution.
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int = 2, # 1 will cause padd issues
        in_chans: int = 1,
        embed_dim: int = 24,
        norm_layer: type | None = nn.LayerNorm,
        spatial_dims: int = 3,
        K: int = 3,
        resolutions: List[int] = DEFAULT_RESOLUTIONS_SwinUNETRv2,
        img_size: int = 96,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("MSPEPatchEmbedSwin supports spatial_dims = 2 or 3.")

        self.spatial_dims = spatial_dims
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.K = K
        self.resolutions = sorted(resolutions)
        # number of patches along ONE spatial dim at base resolution
        self.N: int = img_size // self.patch_size[0]

        # normalization
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        # NOTE: K patching kernels with multipled sizing
        self.patch_kernels = nn.ModuleList()
        for i in range(K):
            k_size = tuple(p * (i + 1) for p in self.patch_size)
            conv = Conv[Conv.CONV, spatial_dims](
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=k_size,
                stride=self.patch_size,
                bias=True,
            )
            self.patch_kernels.append(conv)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pad_input(self, x):
        """Pad all spatial dims so they are divisible by patch_size."""
        spatial = x.shape[2:] 
        pad = []
        for dim_size, p in zip(reversed(spatial), reversed(self.patch_size)):
            pad_amount = (p - dim_size % p) % p
            pad += [0, pad_amount]
        if any(p > 0 for p in pad):
            x = F.pad(x, pad)
        return x

    def _apply_norm(self, x):
        """Flatten spatial dims, apply LayerNorm, then reshape back."""
        if self.norm is not None:
            spatial_shape = x.shape[2:]
            B, C = x.shape[:2]
            x = x.flatten(2).transpose(1, 2)   # (B, N_tokens, C)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, C, *spatial_shape)
        return x

    def adp_conv(self, img, func_idx, hw):
        """Adaptive convolution: PI-resize kernel weights to match hw, then convolve.

        Args:
            img:      Input batch (B, in_chans, *spatial).
            func_idx: Index of the kernel to use from patch_kernels.
            hw:       Effective spatial resolution (isotropic).

        Returns:
            Feature map (B, embed_dim, *out_spatial).
        """
        conv_layer = self.patch_kernels[func_idx]
        w = conv_layer.weight
        b = conv_layer.bias

        k_size = tuple(max(1, hw // self.N) for _ in range(self.spatial_dims))
        w_star = pi_resize(w, k_size)

        pad_size = tuple((k - 1) // 2 for k in k_size)

        if self.spatial_dims == 2:
            return F.conv2d(img, w_star, bias=b, stride=self.patch_size, padding=pad_size)
        else:
            return F.conv3d(img, w_star, bias=b, stride=self.patch_size, padding=pad_size)

    def forward(self, x):
        """Compute multi-scale patch embedding for Swin.

        Args:
            x: image batch (B, in_chans, *spatial).

        Returns:
            Spatial feature map (B, embed_dim, *out_spatial).
        """
        x = self._pad_input(x)
        spatial_shape = x.shape[2:]
        hw_eff = get_effective_resolution(spatial_shape)

        _, res_idx = find_nearest_resolution(hw_eff, self.resolutions)
        func_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1)

        conv_layer = self.patch_kernels[func_idx]
        w = conv_layer.weight
        b = conv_layer.bias

        k_size = tuple(max(dim // self.N, 1) for dim in spatial_shape)
        w_star = pi_resize(w, k_size)

        pad_size = tuple((k - 1) // 2 for k in k_size)

        if self.spatial_dims == 2:
            out = F.conv2d(x, w_star, bias=b, stride=self.patch_size, padding=pad_size)
        else:
            out = F.conv3d(x, w_star, bias=b, stride=self.patch_size, padding=pad_size)

        out = self._apply_norm(out)
        return out

    def forward_k(self, img, func_idx, hw):
        """Forward using a specific kernel k at resolution hw.
        Used for optional uniform multi-resolution training.

        Args:
            img:      Input batch (B, in_chans, *spatial).
            func_idx: Index of the kernel to use.
            hw:       Effective spatial resolution.

        Returns:
            Spatial feature map (B, embed_dim, *out_spatial).
        """
        img = self._pad_input(img)
        spatial_shape = img.shape[2:]
        hw_eff = get_effective_resolution(spatial_shape)
        out = self.adp_conv(img, func_idx=func_idx, hw=hw_eff)
        out = self._apply_norm(out)
        return out

    def sample_resolutions(self):
        subsets = [[] for _ in range(self.K)]
        for res_idx, r in enumerate(self.resolutions):
            func_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1)
            subsets[func_idx].append(r)

        hw_list = []
        for subset in subsets:
            if subset:
                hw_list.append(int(np.random.choice(subset)))
            else:
                hw_list.append(int(np.random.choice(self.resolutions)))
        return hw_list

    def extra_repr(self) -> str:
        kernel_sizes = [f"{c.kernel_size}" for c in self.patch_kernels]
        return (
            f"spatial_dims={self.spatial_dims}, patch_size={self.patch_size}, in_chans={self.in_chans}, "
            f"embed_dim={self.embed_dim}, img_size={self.img_size}, "
            f"K={self.K}, N={self.N}, "
            f"kernel_sizes=[{', '.join(kernel_sizes)}], "
            f"resolutions={self.resolutions}"
        )
