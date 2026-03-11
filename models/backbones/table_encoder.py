"""
Table Encoder Module - MLP-based clinical tabular data encoding.

This module implements encoders for processing structured clinical data
(e.g., patient demographics, vital signs, lab results). It includes both
simple MLP and residual MLP architectures.

Author: Research Team
Date: 2026-03-11
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import warnings


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with skip connection.
    
    Implements a residual connection to improve gradient flow:
        output = ReLU(LayerNorm(Linear(input))) + input
    
    Mathematical Formulation:
        H(x) = F(x) + x
        where F(x) = ReLU(LayerNorm(Linear(x)))
    
    This design is particularly useful for deep MLPs processing
    tabular medical data with many features.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize residual MLP block.
        
        Args:
            hidden_dim: Dimension of hidden layer
            dropout: Dropout probability for regularization
        """
        super(ResidualMLPBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        return x + self.block(x)


class ResidualMLPEncoder(nn.Module):
    """
    Residual MLP encoder for tabular clinical data.
    
    This encoder processes structured clinical features through multiple
    residual MLP blocks, enabling deeper networks while maintaining
    stable gradient flow.
    
    Architecture:
        Input → Linear(projection) → [ResidualBlock × N] → Output
    
    Mathematical Formulation:
        Let x ∈ R^{input_dim} be input features:
        1. Project: h_0 = W_0 · x + b_0 ∈ R^{hidden_dim}
        2. Residual blocks: h_i = ResBlock_i(h_{i-1}) for i=1..N
        3. Output: f = h_N ∈ R^{output_dim}
    
    Attributes:
        input_dim (int): Dimension of input clinical features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of output features
        num_blocks (int): Number of residual blocks
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize residual MLP encoder.
        
        Args:
            input_dim: Dimension of input tabular features
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_blocks: Number of residual MLP blocks
            dropout: Dropout probability
            
        Raises:
            ValueError: If dimensions are invalid
        """
        super(ResidualMLPEncoder, self).__init__()
        
        # Validate parameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to encode tabular features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_intermediate: If True, return intermediate activations
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
            If return_intermediate=True, returns dict with all layer outputs
            
        Mathematical Flow:
            x → Input_Proj → ResBlock_1 → ... → ResBlock_N → Output_Proj → f
        
        Example:
            >>> encoder = ResidualMLPEncoder(input_dim=50, output_dim=256)
            >>> x = torch.randn(32, 50)  # Batch of 32 samples
            >>> features = encoder(x)
            >>> print(features.shape)  # torch.Size([32, 256])
        """
        intermediates = {}
        
        # Input projection
        x = self.input_projection(x)
        if return_intermediate:
            intermediates['input_proj'] = x
        
        # Residual blocks
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            if return_intermediate:
                intermediates[f'residual_block_{i}'] = x
        
        # Output projection
        features = self.output_projection(x)
        
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
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_blocks": self.num_blocks,
            "architecture": "ResidualMLP"
        }


class SimpleMLPEncoder(nn.Module):
    """
    Simple feedforward MLP encoder for tabular data.
    
    A straightforward multi-layer perceptron without residual connections.
    Suitable for simpler tasks or when computational efficiency is prioritized.
    
    Architecture:
        Input → Linear → ReLU → Dropout → [repeat] → Output
    
    Mathematical Formulation:
        f = σ(W_n · σ(W_{n-1} · ... · σ(W_1 · x + b_1) ... + b_{n-1}) + b_n)
        where σ is ReLU activation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize simple MLP encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            dropout: Dropout probability
            
        Raises:
            ValueError: If dimensions are invalid
        """
        super(SimpleMLPEncoder, self).__init__()
        
        # Validate parameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        return self.encoder(x)
    
    def get_config(self) -> Dict:
        """Get encoder configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "architecture": "SimpleMLP"
        }


class TabularEncoderWithEmbedding(nn.Module):
    """
    Advanced tabular encoder with categorical embedding support.
    
    This encoder handles mixed data types:
    - Continuous features: normalized numerical values
    - Categorical features: learned embeddings
    
    It's particularly useful for clinical data containing both
    numerical measurements (age, blood pressure) and categorical
    variables (gender, hospital, diagnosis codes).
    
    Mathematical Formulation:
        For continuous features x_c ∈ R^{n_c} and categorical features x_d ∈ N^{n_d}:
        1. Embed categoricals: e_i = Embedding(x_d[i]) for i=1..n_d
        2. Concatenate: h = [x_c; e_1; ...; e_{n_d}]
        3. Process with MLP: f = MLP(h)
    """
    
    def __init__(
        self,
        continuous_dim: int,
        categorical_dims: Optional[List[int]] = None,
        embedding_dim: int = 8,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize tabular encoder with embedding support.
        
        Args:
            continuous_dim: Number of continuous features
            categorical_dims: List of cardinalities for categorical features
                             (e.g., [2, 4] means 2 categories for first feature,
                              4 for second)
            embedding_dim: Dimension for each categorical embedding
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_blocks: Number of residual blocks
            dropout: Dropout probability
        """
        super(TabularEncoderWithEmbedding, self).__init__()
        
        self.continuous_dim = continuous_dim
        self.categorical_dims = categorical_dims or []
        self.embedding_dim = embedding_dim
        
        # Create embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories in self.categorical_dims
        ])
        
        # Calculate total input dimension
        total_dim = continuous_dim + len(self.categorical_dims) * embedding_dim
        
        # Use residual MLP for main processing
        self.encoder = ResidualMLPEncoder(
            input_dim=total_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )
    
    def forward(
        self,
        continuous_features: torch.Tensor,
        categorical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with mixed feature types.
        
        Args:
            continuous_features: Continuous features (batch_size, continuous_dim)
            categorical_features: Categorical indices (batch_size, num_categorical)
                                 Each column contains category indices
            
        Returns:
            Encoded features of shape (batch_size, output_dim)
            
        Example:
            >>> encoder = TabularEncoderWithEmbedding(
            ...     continuous_dim=10,
            ...     categorical_dims=[2, 4],  # gender (2), hospital (4)
            ...     output_dim=256
            ... )
            >>> cont = torch.randn(32, 10)
            >>> cat = torch.tensor([[0, 2], [1, 1], ...])  # (32, 2)
            >>> features = encoder(cont, cat)
            >>> print(features.shape)  # torch.Size([32, 256])
        """
        features = [continuous_features]
        
        # Embed categorical features if provided
        if categorical_features is not None and len(self.embeddings) > 0:
            if categorical_features.shape[1] != len(self.embeddings):
                raise ValueError(
                    f"Expected {len(self.embeddings)} categorical features, "
                    f"got {categorical_features.shape[1]}"
                )
            
            for i, embedding in enumerate(self.embeddings):
                cat_embed = embedding(categorical_features[:, i])
                features.append(cat_embed)
        
        # Concatenate all features
        combined = torch.cat(features, dim=1)
        
        # Encode
        return self.encoder(combined)
    
    def get_config(self) -> Dict:
        """Get encoder configuration."""
        return {
            "continuous_dim": self.continuous_dim,
            "categorical_dims": self.categorical_dims,
            "embedding_dim": self.embedding_dim,
            "architecture": "TabularWithEmbedding"
        }
