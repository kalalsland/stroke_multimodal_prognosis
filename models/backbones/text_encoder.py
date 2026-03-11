"""
Text Encoder Module - Bio-BERT based clinical text feature extraction.

This module implements a BioBERT-based encoder for extracting semantic features
from clinical text data. It supports both frozen and fine-tunable configurations.

Author: Research Team
Date: 2026-03-11
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import warnings


class BioBERTEncoder(nn.Module):
    """
    BioBERT-based clinical text encoder with optional projection layer.
    
    This encoder uses pre-trained BioBERT model to extract contextual embeddings
    from clinical text. It supports:
    - Frozen or fine-tunable BioBERT weights
    - Optional linear projection to target dimension
    - CLS token pooling strategy
    
    Mathematical Formulation:
        Given input text sequence x = [x_1, ..., x_n]:
        1. Token Embeddings: E = BioBERT(x) ∈ R^{n×768}
        2. Pooling: h = E[CLS] ∈ R^{768}
        3. Projection (optional): f = W·h + b ∈ R^{d}
    
    Attributes:
        model_name (str): HuggingFace model identifier
        output_dim (int): Target feature dimension after projection
        freeze_bert (bool): Whether to freeze BioBERT parameters
        device (str): Device for computation ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        output_dim: int = 256,
        freeze_bert: bool = True,
        max_length: int = 512,
        device: str = "cuda"
    ):
        """
        Initialize BioBERT encoder.
        
        Args:
            model_name: Pre-trained BioBERT model name from HuggingFace
            output_dim: Dimension of output feature vector
            freeze_bert: If True, freeze BioBERT parameters during training
            max_length: Maximum sequence length for tokenization
            device: Computation device ('cuda' or 'cpu')
            
        Raises:
            ValueError: If output_dim <= 0
            RuntimeError: If model fails to load
        """
        super(BioBERTEncoder, self).__init__()
        
        # Validate parameters
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.max_length = max_length
        self.device = device
        
        # Load BioBERT tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            # BioBERT hidden size (typically 768)
            self.hidden_size = self.bert_model.config.hidden_size
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load BioBERT model '{model_name}': {str(e)}"
            )
        
        # Freeze BioBERT parameters if specified
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            warnings.warn(
                "BioBERT parameters are frozen. Set freeze_bert=False to fine-tune."
            )
        
        # Projection layer to target dimension
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Move model to device
        self.to(device)
    
    def forward(
        self,
        texts: list,
        return_attention_mask: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to extract text features.
        
        Args:
            texts: List of clinical text strings
            return_attention_mask: If True, return attention mask alongside features
            
        Returns:
            Text feature tensor of shape (batch_size, output_dim)
            If return_attention_mask=True, returns tuple (features, attention_mask)
            
        Mathematical Steps:
            1. Tokenize texts: tokens = Tokenizer(texts)
            2. BioBERT encoding: H = BioBERT(tokens) → (B, seq_len, 768)
            3. CLS pooling: h_cls = H[:, 0, :] → (B, 768)
            4. Projection: f = ReLU(LayerNorm(Linear(h_cls))) → (B, output_dim)
        """
        # Tokenize input texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Extract BioBERT features
        with torch.set_grad_enabled(not self.freeze_bert):
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use CLS token representation (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # Project to target dimension
        features = self.projection(cls_embeddings)  # (batch_size, output_dim)
        
        if return_attention_mask:
            return features, attention_mask
        return features
    
    def get_config(self) -> Dict:
        """
        Get encoder configuration.
        
        Returns:
            Dictionary containing encoder hyperparameters
        """
        return {
            "model_name": self.model_name,
            "output_dim": self.output_dim,
            "hidden_size": self.hidden_size,
            "freeze_bert": self.freeze_bert,
            "max_length": self.max_length,
            "device": self.device
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> "BioBERTEncoder":
        """
        Create encoder from configuration dictionary.
        
        Args:
            config: Configuration dictionary with encoder parameters
            
        Returns:
            Initialized BioBERTEncoder instance
        """
        return cls(
            model_name=config.get("model_name", "dmis-lab/biobert-v1.1"),
            output_dim=config.get("output_dim", 256),
            freeze_bert=config.get("freeze_bert", True),
            max_length=config.get("max_length", 512),
            device=config.get("device", "cuda")
        )


class SimpleTextEncoder(nn.Module):
    """
    Simple MLP-based text encoder for pre-extracted text features.
    
    This encoder is used when text features are already extracted
    (e.g., from pre-computed BioBERT embeddings stored in Excel files).
    It applies a simple feedforward network for dimension transformation.
    
    Mathematical Formulation:
        f = ReLU(LayerNorm(Linear(x)))
        where x ∈ R^{input_dim}, f ∈ R^{output_dim}
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize simple text encoder.
        
        Args:
            input_dim: Dimension of input text features
            output_dim: Dimension of output features
            dropout: Dropout probability for regularization
            
        Raises:
            ValueError: If dimensions are invalid
        """
        super(SimpleTextEncoder, self).__init__()
        
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.encoder(x)
