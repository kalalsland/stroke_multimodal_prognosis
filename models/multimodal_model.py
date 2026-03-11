"""Complete multimodal model for stroke prognosis prediction.

This module integrates image, text, and table encoders with VDAFM fusion
to predict stroke patient outcomes.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from stroke_multimodal_prognosis.models.backbones import (
    ImageEncoder3D,
    TextEncoder,
    TableEncoder,
)
from stroke_multimodal_prognosis.models.fusion_modules.vdafm import VDAFM, VDAFMConfig


class StrokePrognosisModel(nn.Module):
    """Complete multimodal stroke prognosis prediction model.
    
    Integrates three modalities (MRI images, clinical text, tabular data) using
    VDAFM fusion mechanism for binary outcome prediction.
    
    Architecture:
        1. Modality-specific encoders extract features
        2. VDAFM performs vision-guided multimodal fusion
        3. Classification head predicts prognosis
    
    Args:
        image_config: Configuration dict for ImageEncoder3D
            - in_channels: Input channels (default: 1 for MRI)
            - base_channels: Initial feature channels (default: 64)
        text_config: Configuration dict for TextEncoderBioBERT
            - pretrained_path: Path to BioBERT weights
            - embedding_dim: Output dimension (default: 768)
            - freeze_bert: Whether to freeze BERT layers (default: True)
        table_config: Configuration dict for TableEncoder
            - input_dim: Number of tabular features
            - hidden_dims: List of hidden layer sizes
            - dropout_rate: Dropout probability (default: 0.3)
        vdafm_config: VDAFMConfig instance or dict for fusion module
        num_classes: Number of output classes (default: 2 for binary classification)
        dropout_rate: Dropout before final classifier (default: 0.5)
    
    Example:
        >>> model = StrokePrognosisModel(
        ...     image_config={'in_channels': 1, 'base_channels': 64},
        ...     text_config={'pretrained_path': 'path/to/biobert', 'embedding_dim': 768},
        ...     table_config={'input_dim': 20, 'hidden_dims': [128, 256, 512]},
        ...     vdafm_config=VDAFMConfig(embed_dim=512, num_heads=8),
        ...     num_classes=2
        ... )
        >>> outputs = model(images, text_input_ids, text_attention_mask, table_data)
    """
    
    def __init__(
        self,
        image_config: Dict,
        text_config: Dict,
        table_config: Dict,
        vdafm_config: VDAFMConfig,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        # Store configurations
        self.num_classes = num_classes
        
        # Initialize modality encoders
        self.image_encoder = ImageEncoder3D(**image_config)
        self.text_encoder = TextEncoder(**text_config)
        self.table_encoder = TableEncoder(**table_config)
        
        # Get output dimensions from encoders
        self.image_dim = self.image_encoder.output_dim
        self.text_dim = self.text_encoder.embedding_dim
        self.table_dim = self.table_encoder.output_dim
        
        # Initialize VDAFM fusion module
        if isinstance(vdafm_config, dict):
            vdafm_config = VDAFMConfig(**vdafm_config)
        self.vdafm = VDAFM(vdafm_config)
        
        # Get fused feature dimension
        self.fused_dim = self.vdafm.embed_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.fused_dim // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self) -> None:
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        table_data: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            images: MRI images of shape [B, C, D, H, W]
                where B=batch_size, C=channels (1), D=depth, H=height, W=width
            text_input_ids: Tokenized text input [B, seq_len]
            text_attention_mask: Attention mask for text [B, seq_len]
            table_data: Tabular features [B, num_features]
            return_features: If True, return intermediate features
        
        Returns:
            Dictionary containing:
                - logits: Classification logits [B, num_classes]
                - probabilities: Softmax probabilities [B, num_classes]
                If return_features=True, also includes:
                    - image_features: Encoded image features [B, image_dim]
                    - text_features: Encoded text features [B, text_dim]
                    - table_features: Encoded table features [B, table_dim]
                    - fused_features: VDAFM output [B, fused_dim]
                    - alignment_loss: Alignment regularization loss (scalar)
                    - low_rank_loss: Low-rank regularization loss (scalar)
        
        Raises:
            ValueError: If input dimensions are incompatible
        """
        # Validate inputs
        batch_size = images.size(0)
        if text_input_ids.size(0) != batch_size or table_data.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: images={batch_size}, "
                f"text={text_input_ids.size(0)}, table={table_data.size(0)}"
            )
        
        # Step 1: Encode each modality
        image_features = self.image_encoder(images)  # [B, image_dim]
        text_features = self.text_encoder(
            text_input_ids, text_attention_mask
        )  # [B, text_dim]
        table_features = self.table_encoder(table_data)  # [B, table_dim]
        
        # Step 2: Multimodal fusion with VDAFM
        vdafm_output = self.vdafm(
            image_features, text_features, table_features
        )
        fused_features = vdafm_output["fused_features"]  # [B, fused_dim]
        
        # Step 3: Classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        probabilities = torch.softmax(logits, dim=-1)  # [B, num_classes]
        
        # Prepare output dictionary
        outputs = {
            "logits": logits,
            "probabilities": probabilities,
        }
        
        # Optionally include intermediate features
        if return_features:
            outputs.update({
                "image_features": image_features,
                "text_features": text_features,
                "table_features": table_features,
                "fused_features": fused_features,
                "alignment_loss": vdafm_output["alignment_loss"],
                "low_rank_loss": vdafm_output["low_rank_loss"],
            })
        
        return outputs
    
    def get_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        alignment_weight: float = 0.1,
        low_rank_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with regularization terms.
        
        Total loss = CE loss + λ_align * alignment_loss + λ_rank * low_rank_loss
        
        Args:
            outputs: Model outputs from forward() with return_features=True
            labels: Ground truth labels [B]
            alignment_weight: Weight for alignment loss (λ_align)
            low_rank_weight: Weight for low-rank loss (λ_rank)
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains:
                - total_loss: Combined loss value
                - ce_loss: Cross-entropy classification loss
                - alignment_loss: Modality alignment regularization
                - low_rank_loss: Low-rank structure regularization
        
        Raises:
            KeyError: If outputs missing required keys
        """
        # Validate required keys
        required_keys = ["logits", "alignment_loss", "low_rank_loss"]
        missing_keys = [k for k in required_keys if k not in outputs]
        if missing_keys:
            raise KeyError(f"Missing required output keys: {missing_keys}")
        
        # Classification loss
        ce_loss = nn.functional.cross_entropy(outputs["logits"], labels)
        
        # Regularization losses
        alignment_loss = outputs["alignment_loss"]
        low_rank_loss = outputs["low_rank_loss"]
        
        # Combined loss
        total_loss = (
            ce_loss 
            + alignment_weight * alignment_loss 
            + low_rank_weight * low_rank_loss
        )
        
        # Return loss breakdown
        loss_dict = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "alignment_loss": alignment_loss.item(),
            "low_rank_loss": low_rank_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def predict(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        table_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference mode prediction.
        
        Args:
            images: MRI images [B, C, D, H, W]
            text_input_ids: Tokenized text [B, seq_len]
            text_attention_mask: Attention mask [B, seq_len]
            table_data: Tabular features [B, num_features]
        
        Returns:
            Dictionary with:
                - predictions: Predicted class indices [B]
                - probabilities: Class probabilities [B, num_classes]
                - confidence: Maximum probability per sample [B]
        """
        self.eval()
        with torch.no_grad():
            outputs = self(
                images, text_input_ids, text_attention_mask, table_data,
                return_features=False
            )
            
            probabilities = outputs["probabilities"]
            predictions = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "confidence": confidence,
            }
    
    def freeze_encoders(self, freeze_image: bool = True, freeze_text: bool = True,
                       freeze_table: bool = False) -> None:
        """Freeze encoder parameters for fine-tuning.
        
        Args:
            freeze_image: Freeze image encoder weights
            freeze_text: Freeze text encoder weights (excluding projection)
            freeze_table: Freeze table encoder weights
        """
        if freeze_image:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        if freeze_text:
            self.text_encoder.freeze_bert_layers()
        
        if freeze_table:
            for param in self.table_encoder.parameters():
                param.requires_grad = False
    
    def get_num_parameters(self, trainable_only: bool = False) -> Dict[str, int]:
        """Count model parameters.
        
        Args:
            trainable_only: Only count trainable parameters
        
        Returns:
            Dictionary with parameter counts per component
        """
        def count_params(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())
        
        return {
            "image_encoder": count_params(self.image_encoder),
            "text_encoder": count_params(self.text_encoder),
            "table_encoder": count_params(self.table_encoder),
            "vdafm": count_params(self.vdafm),
            "classifier": count_params(self.classifier),
            "total": count_params(self),
        }
