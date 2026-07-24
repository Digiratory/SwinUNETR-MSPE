"""Multi-Scale Patch Embedding (MSPE) for hierarchical and plain vision transformers.

Reference implementation for *MSPE-Swin: Multi-Scale Patch Embedding For Robust SwinUNETR*,
adapting the MSPE idea of Liu et al. (https://arxiv.org/abs/2405.18240) to SwinUNETR.
"""

import importlib as _importlib

from . import swin_mspe

# Swin kernel designs 
from .swin_mspe import (
    MSPEPatchEmbedSwinNaiveRouting,   
    MSPEPatchEmbedSwinOverlapping,   
    MSPEPatchEmbedSwinDilatingK3,    
)

# Swin training / inference plumbing
from .swin_mspe import (
    mspe_swin_forward,       
    mspe_swin_train_step,    
)

# Shared helpers 
from .swin_mspe import (
    pi_resize,                          # FlexiViT pseudo-inverse kernel resize
    img_resize,                         # bilinear / trilinear image resize
    label_resize,                       # nearest-neighbour resize for labels and ROI masks
    get_aspect_preserving_target_size,  # target shape at an effective resolution, AR preserved
    get_effective_resolution,           # (prod(spatial_shape)) ** (1 / n_dims)
    find_nearest_resolution,            # snap an effective resolution to its band
    DEFAULT_RESOLUTIONS,                # [256, 384, 512, 768, 1024, 1280]
    DEFAULT_K,                          # 3
)

__version__ = "1.0.0"


_VIT_EXPORTS = {
    "MSPE_UNETR",          # ViT encoder + UNETR decoder, MSPE tokenizer
    "MSPEPatchEmbedd",     # multi-scale patch embedding for ViT
    "VanillaPatchEmbed",   # single-kernel baseline embedding
    "UNETRDecoder",
    "mspe_vit_forward",
    "mspe_vit_train_step",
}


def __getattr__(name):
    if name == "vit_mspe":
        return _importlib.import_module(".vit_mspe", __name__)
    if name in _VIT_EXPORTS:
        return getattr(_importlib.import_module(".vit_mspe", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _VIT_EXPORTS | {"vit_mspe"})


__all__ = [
    "MSPEPatchEmbedSwinNaiveRouting",
    "MSPEPatchEmbedSwinOverlapping",
    "MSPEPatchEmbedSwinDilatingK3",
    "mspe_swin_forward",
    "mspe_swin_train_step",
    "pi_resize",
    "img_resize",
    "label_resize",
    "get_aspect_preserving_target_size",
    "get_effective_resolution",
    "find_nearest_resolution",
    "DEFAULT_RESOLUTIONS",
    "DEFAULT_K",
    "MSPE_UNETR",
    "MSPEPatchEmbedd",
    "VanillaPatchEmbed",
    "UNETRDecoder",
    "mspe_vit_forward",
    "mspe_vit_train_step",
    "swin_mspe",
    "vit_mspe",
]
