"""Model definitions for stroke prognosis prediction.

This package contains:
- backbones: Individual encoders for each modality (image, text, table)
- fusion_modules: Multimodal fusion mechanisms (VDAFM)
- multimodal_model: The complete end-to-end model

Public API:
    - ImageEncoder3D: 3D ResNet-50 for MRI image encoding
    - TextEncoderBioBERT: BioBERT-based text encoder
    - TableEncoder: Residual MLP for tabular data
    - VDAFM: Vision-guided dual alignment fusion module
    - StrokePrognosisModel: Complete multimodal model for stroke prognosis
"""

from .backbones import (
    ImageEncoder,
    TextEncoder,
    TableEncoder,
    BioBERTEncoder,
    ResNet50_3D,
    ResidualMLPEncoder
)
from .fusion_modules.vdafm import (
    VDAFM,
    VDAFMConfig,
)
from .multimodal_model import StrokePrognosisModel

__all__ = [
    # Backbone encoders
    "ImageEncoder",
    "TextEncoder",
    "TableEncoder",
    "BioBERTEncoder",
    "ResNet50_3D",
    "ResidualMLPEncoder",
    # Fusion module
    "VDAFM",
    "VDAFMConfig",
    # Main model
    "StrokePrognosisModel",
]
