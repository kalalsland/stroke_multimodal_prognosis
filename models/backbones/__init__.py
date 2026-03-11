"""
Backbone Networks Module

This module contains encoder implementations for different modalities:
- Image encoders: 3D medical image feature extraction
- Text encoders: Medical text feature extraction  
- Tabular encoders: Clinical tabular data encoding
"""

from typing import Dict, Any, Type
import torch.nn as nn

# Import encoders
from .text_encoder import BioBERTEncoder
from .image_encoder import ResNet50_3D, Simple3DEncoder
from .table_encoder import (
    ResidualMLPEncoder,
    SimpleMLPEncoder,
    TabularEncoderWithEmbedding
)

# Create aliases for backward compatibility and cleaner imports
TextEncoder = BioBERTEncoder
ImageEncoder3D = ResNet50_3D
ImageEncoder = ResNet50_3D  # Default image encoder
TableEncoder = ResidualMLPEncoder  # Default table encoder

__all__ = [
    # Original class names
    'BioBERTEncoder',
    'ResNet50_3D',
    'Simple3DEncoder',
    'ResidualMLPEncoder',
    'SimpleMLPEncoder',
    'TabularEncoderWithEmbedding',
    # Aliases
    'TextEncoder',
    'ImageEncoder3D',
    'ImageEncoder',
    'TableEncoder',
    # Factory function
    'get_encoder'
]


def get_encoder(modality: str, encoder_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create encoder based on modality and type.
    
    Args:
        modality: One of ['image', 'text', 'tabular']
        encoder_type: Specific encoder type (e.g., 'resnet3d', 'biobert', 'mlp')
        **kwargs: Additional arguments for encoder initialization
        
    Returns:
        Initialized encoder module
        
    Raises:
        ValueError: If modality or encoder_type is not supported
        
    Examples:
        >>> # Create image encoder
        >>> img_encoder = get_encoder('image', 'resnet3d', output_dim=256)
        >>> 
        >>> # Create text encoder
        >>> text_encoder = get_encoder('text', 'biobert', output_dim=256)
        >>> 
        >>> # Create tabular encoder
        >>> tab_encoder = get_encoder('tabular', 'residual_mlp', input_dim=50, output_dim=256)
    """
    encoders = {
        'image': {
            'resnet3d': ResNet50_3D,
            'simple3d': Simple3DEncoder
        },
        'text': {
            'biobert': BioBERTEncoder
        },
        'tabular': {
            'residual_mlp': ResidualMLPEncoder,
            'simple_mlp': SimpleMLPEncoder,
            'embedding_mlp': TabularEncoderWithEmbedding
        }
    }
    
    if modality not in encoders:
        raise ValueError(
            f"Unsupported modality '{modality}'. "
            f"Choose from {list(encoders.keys())}"
        )
    
    if encoder_type not in encoders[modality]:
        raise ValueError(
            f"Unsupported encoder type '{encoder_type}' for modality '{modality}'. "
            f"Choose from {list(encoders[modality].keys())}"
        )
    
    encoder_class = encoders[modality][encoder_type]
    return encoder_class(**kwargs)
