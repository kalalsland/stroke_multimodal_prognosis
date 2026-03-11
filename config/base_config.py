"""Base configuration dataclasses for the stroke prognosis project.

This module defines configuration classes that replace the hardcoded `config` dict
from the legacy code. All hyperparameters, paths, and training settings are
encapsulated in typed dataclasses for better maintainability and IDE support.

Example:
    >>> from stroke_multimodal_prognosis.config.base_config import TrainingConfig
    >>> cfg = TrainingConfig(batch_size=32, learning_rate=1e-3)
    >>> print(cfg.batch_size)  # 32
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data paths and preprocessing.

    Attributes:
        clinic_path: Path to clinical tabular data (Excel file).
        image_feat_path: Path to pre-extracted image features (Excel file).
        text_feat_path: Path to pre-extracted text features (Excel file).
        test_size: Fraction of data reserved for testing (e.g., 0.2 = 20%).
        val_size: Fraction of training data reserved for validation.
        use_cross_validation: Whether to use k-fold cross-validation.
        n_splits: Number of folds for cross-validation.
    """

    clinic_path: str
    image_feat_path: str
    text_feat_path: str

    test_size: float = 0.2
    val_size: float = 0.2
    use_cross_validation: bool = False
    n_splits: int = 5

    def __post_init__(self) -> None:
        """Validate paths exist."""
        if not Path(self.clinic_path).exists():
            raise FileNotFoundError(f"Clinic data not found: {self.clinic_path}")
        if not Path(self.image_feat_path).exists():
            raise FileNotFoundError(f"Image features not found: {self.image_feat_path}")
        if not Path(self.text_feat_path).exists():
            raise FileNotFoundError(f"Text features not found: {self.text_feat_path}")


SMOTEMethod = Literal["tabular_copy", "image_copy", "none"]
SchedulerType = Literal["cosine_warmup", "plateau", "none"]


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation strategies.

    Attributes:
        use_smote: Whether to apply SMOTE for class balancing.
        smote_method: Which SMOTE strategy to use.
        use_mixup: Whether to apply mixup augmentation during training.
        mixup_alpha: Mixup interpolation strength (Beta distribution param).
        use_gaussian_noise: Whether to add Gaussian noise to features.
        noise_std: Standard deviation of Gaussian noise (if enabled).
    """

    use_smote: bool = True
    smote_method: SMOTEMethod = "tabular_copy"
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_gaussian_noise: bool = True
    noise_std: float = 0.01


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.

    Attributes:
        batch_size: Batch size for DataLoader.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for optimizer.
        min_lr: Minimum learning rate (for schedulers).
        weight_decay: L2 regularization coefficient.
        lambda_cl: Weight for contrastive loss term.
        lambda_align: Weight for alignment loss term.
        patience: Early stopping patience (epochs without improvement).
        grad_clip: Gradient clipping threshold (0 = disabled).
        use_lr_scheduler: Whether to use a learning rate scheduler.
        scheduler_type: Type of LR scheduler.
        warmup_epochs: Number of warmup epochs (for cosine_warmup scheduler).
        num_random_seeds: Number of random seeds to try (for robustness).
        initial_seed: Starting random seed value.
    """

    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    min_lr: float = 1e-7
    weight_decay: float = 0.0

    lambda_cl: float = 0.02
    lambda_align: float = 0.2

    patience: int = 30
    grad_clip: float = 1.0

    use_lr_scheduler: bool = True
    scheduler_type: SchedulerType = "cosine_warmup"
    warmup_epochs: int = 15

    num_random_seeds: int = 4
    initial_seed: int = 39


@dataclass
class ModelConfig:
    """Configuration for VDAFM model architecture.

    Attributes:
        hidden_dim: Hidden dimension for vision-guided encoder.
        projection_dim: Low-rank projection dimension (bottleneck).
        num_heads: Number of attention heads in Transformer encoder.
        dropout_rate: Dropout probability.
        fusion_strategy: How to fuse tabular/image/text features.
        num_classes: Number of output classes (binary = 2).
    """

    hidden_dim: int = 512
    projection_dim: int = 256
    num_heads: int = 32
    dropout_rate: float = 0.5

    fusion_strategy: str = "concat_reduce_concat"
    num_classes: int = 2


@dataclass
class ExperimentConfig:
    """Top-level configuration combining all sub-configs.

    This is the main config object passed around in training scripts.

    Attributes:
        data: Data paths and split settings.
        augmentation: Augmentation strategies.
        training: Training hyperparameters.
        model: Model architecture settings.
        output_dir: Directory to save checkpoints and logs.
        save_best_model: Whether to save the best model during training.
        device: Device to use ('cuda' or 'cpu').
    """

    data: DataConfig
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    output_dir: str = "experiments"
    save_best_model: bool = True
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Create output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
