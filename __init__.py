"""
Stroke Multimodal Prognosis Package
===================================

A PyTorch-based framework for multi-modal stroke prognosis prediction
using clinical data, medical images, and text reports.

Author: Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Import core components
from stroke_multimodal_prognosis.models.multimodal_model import StrokePrognosisModel
from stroke_multimodal_prognosis.config.base_config import BaseConfig

__all__ = [
    "StrokePrognosisModel",
    "BaseConfig",
]
