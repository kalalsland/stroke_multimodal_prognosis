"""
Image Encoder Module - 3D ResNet-50 based MRI feature extraction.

This module implements a 3D ResNet-50 encoder for extracting spatial-temporal
features from 3D medical imaging data (MRI scans). It supports transfer learning
from pre-trained 2D models and custom output dimensions.

Author: Research Team
Date: 2026-03-11
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import warnings


class Conv3DBlock(nn.Module):
    """
    3D Convolutional block with BatchNorm and ReLU.
    
    This is a building block for 3D ResNet architecture.
    
    Mathematical Formulation:
        output = ReLU(BatchNorm(Conv3D(input)))
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        """
        Initialize 3D convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            padding: Padding size
        """
        super(Conv3DBlock, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock3D(nn.Module):
    """
    3D Residual block (bottleneck design) for ResNet-50.
    
    Implements the bottleneck residual connection:
        F(x) = Conv3(Conv2(Conv1(x)))
        output = ReLU(F(x) + x)
    
    The bottleneck design reduces computational cost:
        - 1x1x1 conv: dimension reduction
        - 3x3x3 conv: spatial feature extraction
        - 1x1x1 conv: dimension restoration
    
    Mathematical Formulation:
        Let H(x) = F(x) + x where:
        F(x) = Conv3(1×1×1, ReLU(BN(Conv2(3×3×3, ReLU(BN(Conv1(1×1×1, x)))))))
    """
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize 3D residual block.
        
        Args:
            in_channels: Input channel dimension
            mid_channels: Middle (bottleneck) channel dimension
            out_channels: Output channel dimension
            stride: Stride for the 3x3x3 convolution
            downsample: Downsampling layer for skip connection if needed
        """
        super(ResidualBlock3D, self).__init__()
        
        # Bottleneck: 1x1x1 -> 3x3x3 -> 1x1x1
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        
        self.conv2 = nn.Conv3d(
            mid_channels, mid_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)
        
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, C_in, D, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, D', H', W')
        """
        identity = x
        
        # Bottleneck path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Skip connection with optional downsampling
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet50_3D(nn.Module):
    """
    3D ResNet-50 encoder for medical image feature extraction.
    
    Architecture follows the standard ResNet-50 design adapted to 3D:
        - Input: (B, 1, D, H, W) for single-channel medical images
        - Conv1: 7×7×7 conv + BN + ReLU + MaxPool
        - Conv2_x: 3 bottleneck blocks (64 -> 256 channels)
        - Conv3_x: 4 bottleneck blocks (128 -> 512 channels)
        - Conv4_x: 6 bottleneck blocks (256 -> 1024 channels)
        - Conv5_x: 3 bottleneck blocks (512 -> 2048 channels)
        - Global Average Pooling
        - Projection to output_dim
    
    Mathematical Formulation:
        f = Projection(GAP(Conv5_x(...Conv2_x(Conv1(x)))))
        where f ∈ R^{output_dim}
    
    Attributes:
        in_channels (int): Number of input channels (typically 1 for MRI)
        output_dim (int): Dimension of output feature vector
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 256,
        dropout: float = 0.5,
        pretrained_2d: bool = False
    ):
        """
        Initialize 3D ResNet-50 encoder.
        
        Args:
            in_channels: Number of input channels (1 for grayscale MRI)
            output_dim: Dimension of output feature vector
            dropout: Dropout probability after global pooling
            pretrained_2d: If True, initialize from 2D ResNet-50 weights
            
        Raises:
            ValueError: If invalid parameters provided
        """
        super(ResNet50_3D, self).__init__()
        
        # Validate parameters
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.dropout_prob = dropout
        
        # Initial convolution: 7x7x7, stride=2
        self.conv1 = nn.Conv3d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 256, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, 512, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 2048, blocks=3, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self._initialize_weights()
        
        if pretrained_2d:
            warnings.warn(
                "pretrained_2d=True: 2D to 3D weight inflation not yet implemented. "
                "Using random initialization."
            )
    
    def _make_layer(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a ResNet layer with multiple residual blocks.
        
        Args:
            in_channels: Input channels for first block
            mid_channels: Bottleneck channels
            out_channels: Output channels
            blocks: Number of residual blocks
            stride: Stride for first block (for downsampling)
            
        Returns:
            Sequential module containing residual blocks
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        # First block with potential downsampling
        layers.append(ResidualBlock3D(
            in_channels, mid_channels, out_channels, stride, downsample
        ))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(
                out_channels, mid_channels, out_channels
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to extract image features.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
               where B=batch size, C=channels, D=depth, H=height, W=width
            return_intermediate: If True, return intermediate feature maps
            
        Returns:
            Feature tensor of shape (B, output_dim)
            If return_intermediate=True, returns dict with intermediate features
            
        Mathematical Flow:
            x → Conv1 → MaxPool → Layer1 → Layer2 → Layer3 → Layer4 
              → GlobalAvgPool → Projection → f
        
        Example:
            >>> encoder = ResNet50_3D(in_channels=1, output_dim=256)
            >>> x = torch.randn(4, 1, 32, 128, 128)  # (B, C, D, H, W)
            >>> features = encoder(x)
            >>> print(features.shape)  # torch.Size([4, 256])
        """
        # Store intermediate features if requested
        intermediates = {}
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if return_intermediate:
            intermediates['layer0'] = x
        
        # ResNet blocks
        x = self.layer1(x)
        if return_intermediate:
            intermediates['layer1'] = x
            
        x = self.layer2(x)
        if return_intermediate:
            intermediates['layer2'] = x
            
        x = self.layer3(x)
        if return_intermediate:
            intermediates['layer3'] = x
            
        x = self.layer4(x)
        if return_intermediate:
            intermediates['layer4'] = x
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 2048)
        
        # Projection
        features = self.projection(x)  # (B, output_dim)
        
        if return_intermediate:
            intermediates['features'] = features
            return intermediates
        
        return features
    
    def get_config(self) -> Dict:
        """
        Get encoder configuration.
        
        Returns:
            Dictionary containing encoder hyperparameters
        """
        return {
            "in_channels": self.in_channels,
            "output_dim": self.output_dim,
            "dropout": self.dropout_prob,
            "architecture": "ResNet50_3D"
        }


class Simple3DEncoder(nn.Module):
    """
    Lightweight 3D CNN encoder for quick prototyping.
    
    A simpler alternative to ResNet-50 for faster training
    and debugging purposes.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 256,
        base_channels: int = 32
    ):
        """
        Initialize simple 3D encoder.
        
        Args:
            in_channels: Number of input channels
            output_dim: Output feature dimension
            base_channels: Base number of channels (doubles each layer)
        """
        super(Simple3DEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1
            Conv3DBlock(in_channels, base_channels, stride=2),
            Conv3DBlock(base_channels, base_channels),
            
            # Layer 2
            Conv3DBlock(base_channels, base_channels * 2, stride=2),
            Conv3DBlock(base_channels * 2, base_channels * 2),
            
            # Layer 3
            Conv3DBlock(base_channels * 2, base_channels * 4, stride=2),
            Conv3DBlock(base_channels * 4, base_channels * 4),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.projection = nn.Sequential(
            nn.Linear(base_channels * 4, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.projection(x)
