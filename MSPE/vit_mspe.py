"""Multi-Scale Patch Embedding (MSPE) based on: "Liu et al., https://arxiv.org/abs/2405.18240.
   Following MONAI embedding API.
"""

import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.blocks import TransformerBlock, PatchEmbeddingBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.layers import trunc_normal_, Conv, Norm
from monai.utils import ensure_tuple_rep
from typing import List, Tuple, Sequence

DEFAULT_RESOLUTIONS = [256, 384, 512, 768, 1024, 1280] # Vessels
DEFAULT_K = 3

@functools.lru_cache(maxsize=None)
def _pi_matrix(old_size, new_size, mode):
    """Pseudo-inverse of the resize matrix."""

    n_old = math.prod(old_size)
    eye = torch.eye(n_old).reshape(n_old, 1, *old_size)
    B = F.interpolate(eye, size=new_size, mode=mode,
                      align_corners=False).reshape(n_old, -1).T # (n_new, n_old)
    return torch.linalg.pinv(B.T) # (n_new, n_old)

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

    if old_size == tuple(size_new):
        return w

    mode = "bilinear" if len(size_new) == 2 else "trilinear"

    P = _pi_matrix(tuple(old_size), tuple(size_new), mode)
    w_flat = w.reshape(*w.shape[:2], -1).float()
    w_new = (w_flat @ P.to(w.device).T).reshape(*w.shape[:2], *size_new)

    return w_new.to(w.dtype)

# classic bilinear/trilinear img resize 
def img_resize(img, size):
    spatial_dims = img.ndim - 2
    mode = "bilinear" if spatial_dims == 2 else "trilinear"
    if isinstance(size, int):
        size = [size] * spatial_dims
    return F.interpolate(img, size=tuple(size), mode=mode, align_corners=False)

# nearest neighbor resize for labels and roi masks
def label_resize(label, size):
    spatial_dims = label.ndim - 2
    if isinstance(size, int):
        size = [size] * spatial_dims
    return F.interpolate(label.float(), size=tuple(size), mode="nearest").to(label.dtype)


# preserve aspect ratio while targeting an effective resolution
def get_aspect_preserving_target_size(inputs, effective_resolution, divisor):
    spatial_shape = inputs.shape[2:]
    spatial_dims = len(spatial_shape)
    current_eff = float(math.prod(spatial_shape)) ** (1.0 / spatial_dims)
    scale = effective_resolution / current_eff

    target_size = []
    for dim in spatial_shape:
        resized_dim = int(round(dim * scale))
        resized_dim = max(divisor, int(round(resized_dim / divisor)) * divisor)
        target_size.append(resized_dim)
    return target_size

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

# Defult ViT embedding
class VanillaPatchEmbed(PatchEmbeddingBlock):
    """Standard ViT Patch Embedding with support for dynamic positional embedding resampling 
    for mixed-resolution training.
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop("K", None)
        kwargs.pop("resolutions", None)
        super().__init__(*args, **kwargs)
        self.spatial_dims = kwargs.get("spatial_dims", 2)
        
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
        # Bypass super().forward(x) to prevent pos embedding addition
        x = self.patch_embeddings(x)
        
        out_spatial_shape = x.shape[2:]
        self._out_spatial_shape = out_spatial_shape
        
        x = x.flatten(2).transpose(-1, -2)
        
        if self.position_embeddings is not None:
            pos_embed = self._resample_pos_embed(self.position_embeddings, out_spatial_shape)
            x = x + pos_embed
            
        x = self.dropout(x)
        return x


# ViT embedding bleock
class MSPEPatchEmbedd(nn.Module):
    """Multi-Scale Patch Embedding (MSPE).

    Replaces the default single convolution in MONAI ViT PatchEmbeddingBlock 
    with K learnable convolution kernels of different resolutions. 
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
        K: int = 3, 
        resolutions: List[int] = DEFAULT_RESOLUTIONS,
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

        # Add K patching kernels of different sizes
        self.patch_kernels = nn.ModuleList()
        for i in range(K):
            scale_i = (i + 1) / K # scale factors 1/K, 2/K, ... 1.0 K for conv kernels
            k_size = tuple(max(1, int(p * scale_i)) for p in self.patch_size)
            conv = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=k_size,
                stride=k_size,  # non-overlapping 
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
    
    def adp_conv(self, x, func_idx):
        """Adaptive convolution for inference.

        1. Select the kernel weights from list of all kernels generated by K patch_kernels.
        2. PI-resize the weights to (hw // N, hw // N) so that the encoder input has N x N tokens.
        3. Do convolution with adapted weights.
        """

        conv_layer = self.patch_kernels[func_idx]

        w = conv_layer.weight
        b = conv_layer.bias

        # Target kernel size
        spatial_shape = x.shape[2:]
        k_size = tuple(max(round(dim / self.N), 1) for dim in spatial_shape)

        # Pad symmetrically 
        pad = []
        for dim, k in zip(reversed(spatial_shape), reversed(k_size)):
            extra = (-dim) % k
            pad += [extra // 2, extra - extra // 2]
        if any(pad):
            x = F.pad(x, pad, mode="replicate")

        w_star = pi_resize(w, k_size)
        if self.spatial_dims == 2:
            out = F.conv2d(x, w_star, bias=b, stride=k_size) 
        elif self.spatial_dims == 3:
            out = F.conv3d(x, w_star, bias=b, stride=k_size)
            
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
        """Compute multi-scale patch embeddings (inference and training).

        Given an image, forward (inference) picks the best kernel, PI-resizes its weights, 
        and produces the embedding.
        """
        spatial_shape = x.shape[2:]
        hw_eff = get_effective_resolution(spatial_shape)

        # Get the nearest resolution
        _, res_idx = find_nearest_resolution(hw_eff, self.resolutions)

        # Map sorted resolution index to kernel index 
        func_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1) 

        # Adaptive convolution
        out = self.adp_conv(x, func_idx)
            
        out_spatial_shape = out.shape[2:]
        self._out_spatial_shape = out_spatial_shape
        out = out.flatten(2).transpose(-1, -2) # (B, N_patches, hidden_size)
            
        if self.position_embeddings is not None:
            pos_embed = self._resample_pos_embed(self.position_embeddings, out_spatial_shape)
            out = out + pos_embed
            
        out = self.dropout(out)
            
        return out

    def forward_k(self, x, func_idx):
        """Forward with a specific kernel k (bypasses auto-routing).
        """
        out = self.adp_conv(x, func_idx)

        out_spatial_shape = out.shape[2:]
        self._out_spatial_shape = out_spatial_shape
        out = out.flatten(2).transpose(-1, -2)

        if self.position_embeddings is not None:
            pos_embed = self._resample_pos_embed(self.position_embeddings, out_spatial_shape)
            out = out + pos_embed

        out = self.dropout(out)
        return out

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


# UNETR skip-connected decoder
class UNETRDecoder(nn.Module):
    """UNETR decoder for a ViT encoder with token grid resampling for MSPE.

    Args:
        in_channels: number of input image channels (for the full-res encoder1 skip).
        hidden_size: transformer hidden dimension.
        out_channels: number of output segmentation classes.
        patch_size: ViT patch size (assumes 16 -> 4 upsampling stages).
        spatial_dims: 2 or 3.
        feature_size: base decoder channel width (stages are fs, 2fs, 4fs, 8fs).
        norm_name: normalization for the conv blocks (UNETR default "instance").
        res_block: use residual conv blocks inside the UNETR blocks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        patch_size: int | Sequence[int] = 16,
        spatial_dims: int = 2,
        feature_size: int = 32,
        norm_name: Tuple | str = "instance",
        res_block: bool = True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        self._mode = "bilinear" if spatial_dims == 2 else "trilinear"

        # Decoder depth follows the patch size (4 for P = 16 (vanilla), 3 for P = 8 (vessles))
        self.n_up = int(round(math.log2(self.patch_size)))

        # Full-resolution input skip
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=feature_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        # Transformer-skip encoders 
        self.encoder_skips = nn.ModuleList()
        for i in range(self.n_up - 1):
            self.encoder_skips.append(
                UnetrPrUpBlock(
                    spatial_dims=spatial_dims, in_channels=hidden_size,
                    out_channels=feature_size * (2 ** (i + 1)),
                    num_layer=self.n_up - 2 - i, kernel_size=3, stride=1, upsample_kernel_size=2,
                    norm_name=norm_name, conv_block=True, res_block=res_block,
                )
            )
        # Decoder upsampling path (n_up stages)
        self.decoders = nn.ModuleList()
        for j in range(1, self.n_up + 1):
            in_ch = hidden_size if j == 1 else feature_size * (2 ** (self.n_up - j + 1))
            self.decoders.append(
                UnetrUpBlock(
                    spatial_dims=spatial_dims, in_channels=in_ch,
                    out_channels=feature_size * (2 ** (self.n_up - j)),
                    kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
                )
            )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
    # Resample MSPE tokens
    def _to_map(self, tokens, grid_shape, canonical_grid):
        """Reshape (B, N, C) tokens to a spatial map, resized to the canonical grid."""
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, *grid_shape)
        if tuple(grid_shape) != tuple(canonical_grid):
            x = F.interpolate(x, size=canonical_grid, mode=self._mode, align_corners=False)
        return x

    def forward(self, x_final, hidden_skips, grid_shape, x_in):
        """Decode token embeddings (+ skips) to a full-resolution segmentation map. """
        spatial_shape = x_in.shape[2:]
        canonical_grid = tuple(d // self.patch_size for d in spatial_shape)

        enc1 = self.encoder1(x_in)

        encs = [enc(self._to_map(hidden_skips[i], grid_shape, canonical_grid))
                for i, enc in enumerate(self.encoder_skips)]

        dec = self._to_map(x_final, grid_shape, canonical_grid)
        for j in range(1, self.n_up + 1):
            skip = encs[self.n_up - 1 - j] if j < self.n_up else enc1
            dec = self.decoders[j - 1](dec, skip)
        return self.out(dec)



# MSPE ViT encoder + UNETR decoder
class MSPE_UNETR(nn.Module):
    """MSPE/Vanilla ViT encoder + UNETR decoder for semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        img_size: int | Sequence[int] = 1024,  # Change per dataset, e.g. 1152 for Vessels
        patch_size: int | Sequence[int] = 16,
        hidden_size: int = 384,  # DeiT-S param
        num_heads: int = 6,      # DeiT-S param
        num_layers: int = 12,    # DeiT-S param
        spatial_dims: int = 2,
        dropout_rate: float = 0.0,
        pos_embed_type: str = "learnable",
        K: int = DEFAULT_K,
        resolutions: List[int] = DEFAULT_RESOLUTIONS,
        feature_size: int = 32,
        norm_name: Tuple | str = "instance", # batch_size = 1, change to batch if higher
        patch_embed_class=None,
        use_flash_attention: bool = True,  # memory efficient  attention 
    ):
        super().__init__()

        if patch_embed_class is None:
            patch_embed_class = MSPEPatchEmbedd

        self.spatial_dims = spatial_dims

        # MSPE OR Vanilla patch embedding 
        self.patch_embed = patch_embed_class(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            K=K,
            resolutions=resolutions,
        )

        # ViT transformer encoder
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, hidden_size * 4, num_heads, dropout_rate, qkv_bias=True, # qkv_bias DeIT
                              use_flash_attention=use_flash_attention) # flash attention
             for _ in range(num_layers)]
        )

        # Del cross-attention modukles 
        for blk in self.blocks:
            del blk.norm_cross_attn, blk.cross_attn

        self.norm = nn.LayerNorm(hidden_size)

        # Block outputs
        ps = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        n_up = int(round(math.log2(ps)))
        self.skip_layers = [num_layers * i // n_up for i in range(1, n_up)]

        # UNETR decoder
        self.decoder = UNETRDecoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            out_channels=out_channels,
            patch_size=patch_size,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
        )

    def _encode_blocks(self, x):
        """Run transformer blocks; return (final_normed, [skip0, skip1, skip2])."""
        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)
        final = self.norm(x)
        skips = [hidden_states[i] for i in self.skip_layers]
        return final, skips

    def _forward_encoder(self, x):
        """Encode through patch embedding + transformer blocks."""
        x = self.patch_embed(x)
        return self._encode_blocks(x)

    def _forward_encoder_k(self, x, func_idx):
        """Encode using a specific MSPE kernel k."""
        x = self.patch_embed.forward_k(x, func_idx)
        return self._encode_blocks(x)

    def _decode(self, enc_out, img_shape, x_in):
        """Decode (final, skips) token sequences to a segmentation map."""
        final, skips = enc_out
        patch_grid = self.patch_embed._out_spatial_shape
        out = self.decoder(final, skips, patch_grid, x_in)
        if out.shape[2:] != img_shape:
            mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
            out = F.interpolate(out, size=img_shape, mode=mode, align_corners=False)
        return out

    def forward(self, x):
        """Standard forward pass (auto-selects best MSPE kernel)."""
        img_shape = x.shape[2:]
        enc_out = self._forward_encoder(x)
        return self._decode(enc_out, img_shape, x)


def mspe_vit_forward(model, x_in, func_idx):
    """Forward pass through an MSPE model  using a specific MSPE kernel."""
    img_shape = x_in.shape[2:]
    enc_out = model._forward_encoder_k(x_in, func_idx)
    return model._decode(enc_out, img_shape, x_in)


def mspe_vit_train_step(model, img, label, loss_fn, lam=1.0, mask=None, backward_fn=None):
    """MSPE training step: K resolution-specific forwards + 1 original resolution forward.

    Implements Algorithm 1 from the MSPE paper (adapted for ViT/SETR).

    For each batch:
      1. Sample K resolutions (one per each kernel).
      2. For each k: resize (img + label + mask) -- forward with kernel k -- loss.
      3. Forward at native resolution -- add lambda weight loss.
      4. Normalize loss by (K + 1) for vanilla training comparison.
    """
    patch_embed = model.patch_embed
    hw_list = patch_embed.sample_resolutions()
    K = len(hw_list)

    # Get the patch size 
    divisor = patch_embed.patch_size[0]

    # Normalize each pass by (K + 1) for vanilla training comparison
    def accumulate(loss):
        loss = loss / (K + 1)
        if backward_fn is not None: 
            with torch.autocast(device_type=img.device.type, enabled=False):
                backward_fn(loss)
            loss = loss.detach()
        return loss

    total_loss = 0.0

    # K forward passes at different resolutions
    for k, eff_target_k in enumerate(hw_list):
        target_size = get_aspect_preserving_target_size(img, eff_target_k, divisor)
        img_k = img_resize(img, target_size)
        label_k = label_resize(label, target_size)
        mask_k = label_resize(mask, target_size) if mask is not None else None

        logits_k = mspe_vit_forward(model, img_k, func_idx=k)

        if mask_k is not None:
            loss_k = loss_fn(logits_k, label_k, mask=mask_k)
        else:
            loss_k = loss_fn(logits_k, label_k)

        total_loss = total_loss + accumulate(loss_k)

    # Original resolution forward (lambda weighted)
    logits_orig = model(img)
    if mask is not None:
        loss_orig = loss_fn(logits_orig, label, mask=mask)
    else:
        loss_orig = loss_fn(logits_orig, label)

    total_loss = total_loss + accumulate(lam * loss_orig)

    return total_loss
