"""Multi-Scale Patch Embedding (MSPE) based on: "Liu et al., https://arxiv.org/abs/2405.18240.
   Following MONAI embedding API.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import trunc_normal_, Conv
from monai.utils import ensure_tuple_rep
from typing import List, Tuple, Sequence

DEFAULT_RESOLUTIONS = [256, 384, 512, 768, 1024, 1280] # VESSELS, 95% of train split in [1024–1151] eff  res
DEFAULT_K = 3

def pi_resize(w, size_new):
    """Pseudo inverse (PI)-resize for 2D/3D from FlexiViT paper <https://arxiv.org/abs/2212.08013>.
    Adapted from FlexiViT source:
    <https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py> 
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

# nearest neighbor resize for labels and roi masks
def label_resize(label, size):
    spatial_dims = label.ndim - 2
    if isinstance(size, int):
        size = [size] * spatial_dims
    return F.interpolate(label.float(), size=tuple(size), mode="nearest").to(label.dtype)

# preserve aspect ratio while targeting an effective resolution
def get_aspect_preserving_target_size(inputs, effective_resolution, divisor=32):
    spatial_shape = inputs.shape[2:]
    spatial_dims = len(spatial_shape)
    current_eff = float(np.prod(spatial_shape)) ** (1.0 / spatial_dims)
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

# get the closest resolution from DEFAULT_RESOLUTIONS to current resolution
def find_nearest_resolution(r_star, resolutions):
    best_idx = 0
    best_dist = float("inf")
    for i, r in enumerate(resolutions):
        d = abs(r - r_star)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return resolutions[best_idx], best_idx


class _MSPEPatchEmbedSwinBase(nn.Module):
    """Multi-Scale Patch Embedding (MSPE) for SwinUNETR."""

    def __init__(
        self,
        patch_size: Sequence[int] | int = 2, 
        in_chans: int = 1,
        embed_dim: int = 24,
        norm_layer: type | None = nn.LayerNorm,
        spatial_dims: int = 3,
        K: int = DEFAULT_K,
        resolutions: List[int] = DEFAULT_RESOLUTIONS,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("MSPEPatchEmbedSwin supports spatial_dims = 2 or 3.")

        self.spatial_dims = spatial_dims
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.K = K
        self.resolutions = sorted(resolutions)

        # normalization
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        # K patching kernels
        self.patch_kernels = nn.ModuleList()
        self._build_patch_kernels()

        self.apply(self._init_weights)

    def _build_patch_kernels(self):
        raise NotImplementedError

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
    # Layer Norm
    def _apply_norm(self, x):
        if self.norm is not None:
            spatial_shape = x.shape[2:]
            B, C = x.shape[:2]
            x = x.flatten(2).transpose(1, 2)   # (B, N_tokens, C)
            x = self.norm(x)
            x = x.transpose(1, 2).reshape(B, C, *spatial_shape)
        return x

    def forward(self, x):
        """Compute multi-scale patch embedding for Swin."""
        spatial_shape = x.shape[2:]
        hw_eff = get_effective_resolution(spatial_shape)

        _, res_idx = find_nearest_resolution(hw_eff, self.resolutions)
        kernel_idx = min((res_idx * self.K) // len(self.resolutions), self.K - 1)
        # NOTE: auto-select the best kernel for current resolution
        out = self.patch_kernels[kernel_idx](x) 
        out = self._apply_norm(out)
        return out 

    def forward_k(self, img, func_idx):
        """Forward using a specific kerel k.
        """
        out = self.patch_kernels[func_idx](img)
        out = self._apply_norm(out)
        return out

    # for each kernlel K sample one resoluition 
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
            f"embed_dim={self.embed_dim}, K={self.K}, "
            f"kernel_sizes=[{', '.join(kernel_sizes)}], "
            f"resolutions={self.resolutions}"
        )

# NAIVE ROUTING MSPE-SWIN (identical kernels, Q.: does naive routing emmbeding help?
class MSPEPatchEmbedSwinNaiveRouting(_MSPEPatchEmbedSwinBase):
    """MSPE Swin patch embedding with K identical patch kernels."""

    def _build_patch_kernels(self):
        # THE SAME K patching kernels, for routing testing
        for i in range(self.K):
            conv = Conv[Conv.CONV, self.spatial_dims](
                in_channels=self.in_chans,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size, # default  SWIN kernels
                stride=self.patch_size,
                bias=True,
            )
            self.patch_kernels.append(conv)

# DILATING K3 MSPE-SWIN (Q.: is dilation a good idea for receptive field expansion?)
class MSPEPatchEmbedSwinDilatingK3(_MSPEPatchEmbedSwinBase):

    def _build_patch_kernels(self):
        # k = patch_size + 1 
        k_size = tuple(p + 1 for p in self.patch_size)  # (3, 3) when patch_size (2,2)
        for i in range(self.K):
            dilation = i + 1
            pad = tuple(dilation * (k - 1) // 2 for k in k_size)
            conv = Conv[Conv.CONV, self.spatial_dims](
                in_channels=self.in_chans,
                out_channels=self.embed_dim,
                kernel_size=k_size,
                stride=self.patch_size,  
                padding=pad,
                dilation=dilation,
                bias=True,
            )
            self.patch_kernels.append(conv)



# OVERLAPPING MSPE-SWIN 
class MSPEPatchEmbedSwinOverlapping(_MSPEPatchEmbedSwinBase):
    """MSPE Swin patch embedding with overlapping kernels."""

    def _build_patch_kernels(self):
        for i in range(self.K):
            k_size = tuple(p * (i + 1) for p in self.patch_size) # k_size (2x2, 4x4, 6x6, ...)
            pad = tuple((k - s) // 2 for k, s in zip(k_size, self.patch_size))
            conv = Conv[Conv.CONV, self.spatial_dims](
                in_channels=self.in_chans,
                out_channels=self.embed_dim,
                kernel_size=k_size, # Overlapping kernels
                stride=self.patch_size,
                padding=pad,
                bias=True,
            )
            self.patch_kernels.append(conv)


def mspe_swin_forward(model, x_in, func_idx):
    """Forward pass using a specific MSPE kernel """
    swin = model.swinViT
    
    # Custom embedding 
    x0 = swin.patch_embed.forward_k(x_in, func_idx=func_idx)
    x0 = swin.pos_drop(x0)
   
    # The rest is the same as in monai_swin.py
    x0_out = swin.proj_out(x0, model.normalize)
    if swin.use_v2:
        x0 = swin.layers1c[0](x0.contiguous())
    x1 = swin.layers1[0](x0.contiguous())
    x1_out = swin.proj_out(x1, model.normalize)

    if swin.use_v2:
        x1 = swin.layers2c[0](x1.contiguous())
    x2 = swin.layers2[0](x1.contiguous())
    x2_out = swin.proj_out(x2, model.normalize)

    if swin.use_v2:
        x2 = swin.layers3c[0](x2.contiguous())
    x3 = swin.layers3[0](x2.contiguous())
    x3_out = swin.proj_out(x3, model.normalize)

    if swin.use_v2:
        x3 = swin.layers4c[0](x3.contiguous())
    x4 = swin.layers4[0](x3.contiguous())
    x4_out = swin.proj_out(x4, model.normalize)

    hidden_states_out = [x0_out, x1_out, x2_out, x3_out, x4_out]

    enc0 = model.encoder1(x_in)
    enc1 = model.encoder2(hidden_states_out[0])
    enc2 = model.encoder3(hidden_states_out[1])
    enc3 = model.encoder4(hidden_states_out[2])
    dec4 = model.encoder10(hidden_states_out[4])
    dec3 = model.decoder5(dec4, hidden_states_out[3])
    dec2 = model.decoder4(dec3, enc3)
    dec1 = model.decoder3(dec2, enc2)
    dec0 = model.decoder2(dec1, enc1)
    out = model.decoder1(dec0, enc0)
    logits = model.out(out)
    return logits


def mspe_swin_train_step(model, img, label, loss_fn, lam=1.0, mask=None):
    """MSPE training step: K resolution-specific forwards + 1 original resolution forward.

    Implements Algorithm 1 from the MSPE paper.

    For each batch:
      1. Sample K resolutions (one per each kernel).
      2. For each k: resize (img + label + mask) -- forward with kernel k -- loss.
      3. Forward at native resolution -- add lambda weight loss.
      4. Normalize loss by (K + 1) for vanilla training comparasing.
    """
    patch_embed = model.swinViT.patch_embed
    hw_list = patch_embed.sample_resolutions()  # get K resolutions
    K = len(hw_list)

    total_loss = 0.0

    # K forward passes + original res forward
    for k, eff_target_k in enumerate(hw_list):
        target_size = get_aspect_preserving_target_size(img, eff_target_k)  # preserve AR for histology
        # Resize (img + label + mask)
        img_k = img_resize(img, target_size)
        label_k = label_resize(label, target_size)
        mask_k = label_resize(mask, target_size) if mask is not None else None
        # Forward with kernel k
        logits_k = mspe_swin_forward(model, img_k, func_idx=k)
        # If roi mask is avaliable
        if mask_k is not None:
            loss_k = loss_fn(logits_k, label_k, mask=mask_k)
        else:
            loss_k = loss_fn(logits_k, label_k)

        total_loss = total_loss + loss_k

    # Original res forward (lambda weighted)
    logits_orig = model(img)
    if mask is not None:
        loss_orig = loss_fn(logits_orig, label, mask=mask)
    else:
        loss_orig = loss_fn(logits_orig, label)

    total_loss = total_loss + lam * loss_orig
    # Normalize loss for vanilla train comp
    return total_loss / (K + 1)
