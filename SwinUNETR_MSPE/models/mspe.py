"""Multi-Scale Patch Embedding (MSPE) based on: "Liu et al., https://arxiv.org/abs/2405.18240.
   Following MONAI embedding API.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import trunc_normal_
from typing import List, Tuple, Sequence

DEFAULT_RESOLUTIONS_ViT = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 448] # table 1 of MSPE paper
DEFAULT_RESOLUTIONS_SwinUNETRv2 = [64, 96, 128, 160, 192, 224, 256, 320] # multiples of 32

def pi_resize(w, hw_new):
    """Pseudo inverse (PI)-resize from FlexiViT paper <https://arxiv.org/abs/2212.08013>.
    Adapted from FlexiViT source:
    <https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py> 

    Args:
        w: original kernel weights.
        w dims: (out_channels (O), in_channels (I), old_kH, old_kW)
        hw_new: target spatial size (new_kH, new_kW).

    Returns:
        resized weights.
        w_new dims: (out_channels (O), in_channels (I), new_kH, new_kW)
    """
    assert w.ndim == 4, f"four dimensions expected"
    assert len(hw_new) == 2, f"new shape should only be hw_new"

    old_h, old_w = w.shape[2], w.shape[3]
    new_h, new_w = hw_new 

    # if dims already match do nothing
    if (old_h, old_w) == (new_h, new_w):
        return w

    # create resize matrix B 
    n_old = old_h * old_w
    eye = torch.eye(n_old).reshape(n_old, 1, old_h, old_w) 
 
    B = F.interpolate(eye, size=(new_h, new_w), mode="bilinear",  
                      align_corners=False).reshape(n_old, -1).T # ( n_old, new_h*new_w)

    P = torch.linalg.pinv(B.T) # (new_h*new_w, n_old)
    w_flat = w.reshape(*w.shape[:2], -1).float() # (O, I, n_old)
    w_new = (w_flat @ P.to(w.device).T).reshape(*w.shape[:2], new_h, new_w) #  (O, I, new_h, new_w)

    return w_new.to(w.dtype)

# classic bilinear img resize 
def img_resize(img, hw):
    return F.interpolate(img, size=(hw, hw), mode="bilinear", align_corners=False)

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
        img_size: int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int, 
        pos_embed_type: str = "learnable", 
        dropout_rate: float = 0.0,
        spatial_dims: int = 2, # 2D is hardcoded for now
        K: int = 4, #  paper recomednation
        resolutions: List[int] = DEFAULT_RESOLUTIONS_ViT,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        if isinstance(patch_size, (list, tuple)):
            self.patch_size: Tuple[int, int] = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))

        if self.patch_size[0] > img_size:
            raise ValueError("patch_size should be smaller than img_size.")

        self.in_channels = in_channels         
        self.hidden_size = hidden_size       
        self.img_size = img_size         
        self.K = K                       
        self.resolutions = sorted(resolutions)  
        # number of patches along ONE spatial dim (square assumption), rect. 
        self.N: int = img_size // self.patch_size[0] 

        # Add positional embedding and dropout
        self.pos_embed_type = pos_embed_type
        self.n_patches = self.N * self.N
        self.dropout = nn.Dropout(dropout_rate)

        if self.pos_embed_type == "none": 
            self.position_embeddings = None  # clear preallocated params 
        elif self.pos_embed_type == "learnable": # ViT case
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.pos_embed_type == "sincos": 
            pe = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
            with torch.no_grad():
                grid_size = [self.N, self.N]
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
            kH = max(1, int(self.patch_size[0] * scale_i))  # kH >= 1 
            kW = max(1, int(self.patch_size[1] * scale_i))  # kW >= 1 
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=(kH, kW),
                stride=(kH, kW),  # non-overlapping as in  ViT
                bias=True,
            )
            self.patch_kernels.append(conv)

        self.apply(self._init_weights)

    # standard ViT weight initialization
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
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
        k_h = max(1, hw // self.N)
        k_w = max(1, hw // self.N)

        w_star = pi_resize(w, (k_h, k_w))
        out = F.conv2d(img, w_star, bias=b, stride=(k_h, k_w)) 

        return out

    def _resample_pos_embed(self, pos_embed, target_shape):
        """Resample the base position embedding to the target shape."""
        B, N_patches, C = pos_embed.shape
        N_old = int(math.sqrt(N_patches))
        N_h, N_w = target_shape

        if (N_old, N_old) == (N_h, N_w):
            return pos_embed
            
        pos_embed = pos_embed.reshape(B, N_old, N_old, C).permute(0, 3, 1, 2)
        resampled = F.interpolate(
            pos_embed, 
            size=(N_h, N_w), 
            mode="bilinear", 
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
            Patch embeddings (B, N_h * N_w, hidden_size) 
        """
        _, _, H, W = x.shape # (B, C, H, W)
        hw_eff = int(math.sqrt(H * W)) # geometric mean for non-square imgs, 

        # Get the nearest resolution
        _, res_idx = find_nearest_resolution(hw_eff, self.resolutions)

        # map sorted resolution index to kernel index 
        func_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1) 
        conv_layer = self.patch_kernels[func_idx]

        w = conv_layer.weight # (hidden_size, in_channels, base_kH, base_kW)
        b = conv_layer.bias  # (hidden_size, )

        # target kernel size
        k_h = max(H // self.N, 1) # ensure at least kernel size 1
        k_w = max(W // self.N, 1)

        w_star = pi_resize(w, (k_h, k_w)) #(hidden_size, in_channels, k_h, k_w)
        
        out = F.conv2d(x, w_star, bias=b, stride=(k_h, k_w)) # (B, hidden_size, N_h, N_w)
        _, _, N_h, N_w = out.shape # for resampling PE
        out = out.flatten(2).transpose(-1, -2) # (B, N_h * N_w, hidden_size)
            
        if self.position_embeddings is not None:
            pos_embed = self._resample_pos_embed(self.position_embeddings, (N_h, N_w))
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
        _, _, N_h, N_w = z.shape
        z = z.flatten(2).transpose(-1, -2)

        if self.position_embeddings is not None:
            pos_embed = self._resample_pos_embed(self.position_embeddings, (N_h, N_w))
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
            f"patch_size={self.patch_size}, in_channels={self.in_channels}, "
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
        spatial_dims: number of spatial dimensions (2D hardcoded).
        K: number of multi-res patching kernels.
        resolutions: full list of training resolutions.
        img_size: base image resolution.
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type | None = nn.LayerNorm,
        spatial_dims: int = 2,
        K: int = 4,
        resolutions: List[int] = DEFAULT_RESOLUTIONS_SwinUNETRv2,
        img_size: int = 96, #
    ):
        super().__init__()

        if spatial_dims != 2:
            raise ValueError("MSPEPatchEmbedSwin currently works only with spatial_dims=2.")

        if isinstance(patch_size, (list, tuple)):
            self.patch_size: Tuple[int, int] = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))

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
            kH = self.patch_size[0] * (i + 1)
            kW = self.patch_size[1] * (i + 1)
            conv = nn.Conv2d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=(kH, kW),
                stride=self.patch_size,  # pu-pu-pu
                bias=True,
            )
            self.patch_kernels.append(conv)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
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
        """Pad spatial dims so they are divisible by patch_size (2D only)."""
        _, _, h, w = x.size()
        pad_h = (self.patch_size[0] - h % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - w % self.patch_size[1]) % self.patch_size[1]
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def _apply_norm(self, x):
        if self.norm is not None:
            # x: (B, C, H', W') -> (B, H'*W', C) -> norm -> (B, C, H', W')
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, C, H, W)
        return x

    def adp_conv(self, img, func_idx, hw):
        """Adaptive convolution: PI-resize kernel weights, then convolve.

        Args:
            img:      Input image batch (B, in_chans, hw, hw).
            func_idx: Index of the kernel to use from patch_kernels.
            hw:       Current spatial resolution.

        Returns:
            Patch embeddings (B, embed_dim, N_h, N_w).
        """
        conv_layer = self.patch_kernels[func_idx]
        w = conv_layer.weight
        b = conv_layer.bias

        k_h = max(1, hw // self.N)
        k_w = max(1, hw // self.N)

        w_star = pi_resize(w, (k_h, k_w))
        out = F.conv2d(img, w_star, bias=b, stride=self.patch_size)
        return out

    def forward(self, x):
        """Compute multi-scale patch embedding for Swin.

        Args:
            x: image batch (B, in_chans, H, W).

        Returns:
            Spatial feature map (B, embed_dim, H', W').
        """
        x = self._pad_input(x)
        _, _, H, W = x.shape

        hw_eff = int(math.sqrt(H * W))

        _, res_idx = find_nearest_resolution(hw_eff, self.resolutions)
        func_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1)

        conv_layer = self.patch_kernels[func_idx]
        w = conv_layer.weight
        b = conv_layer.bias

        k_h = max(H // self.N, 1)
        k_w = max(W // self.N, 1)

        w_star = pi_resize(w, (k_h, k_w))
        out = F.conv2d(x, w_star, bias=b, stride=self.patch_size)  # (B, embed_dim, H', W')
        out = self._apply_norm(out)
        return out

    def forward_k(self, img, func_idx, hw):
        """Forward using a specific kernel k at resolution hw.
        Used for multi-resolution training.

        Args:
            img:      Input image batch (B, in_chans, hw, hw).
            func_idx: Index of the kernel to use.
            hw:       Current spatial resolution.

        Returns:
            Spatial feature map (B, embed_dim, N_h, N_w).
        """
        img = self._pad_input(img)
        _, _, H_padded, W_padded = img.shape
        out = self.adp_conv(img, func_idx=func_idx, hw=max(H_padded, W_padded))
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

    # for printing
    def extra_repr(self) -> str:
        kernel_sizes = []
        for i in range(self.K):
            c = self.patch_kernels[i]
            kernel_sizes.append(f"{c.kernel_size}")
        return (
            f"patch_size={self.patch_size}, in_chans={self.in_chans}, "
            f"embed_dim={self.embed_dim}, img_size={self.img_size}, "
            f"K={self.K}, N={self.N}, "
            f"kernel_sizes=[{', '.join(kernel_sizes)}], "
            f"resolutions={self.resolutions}"
        )
