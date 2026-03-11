"""Main training script for stroke prognosis model.

This script demonstrates the complete training pipeline including:
- Configuration loading
- Data preparation
- Model initialization
- Training with validation
- Model evaluation and saving

Usage:
    python train.py --config configs/default/base.yaml
    python train.py --config configs/custom_config.yaml --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from models import StrokePrognosisModel
from data import StrokeDataset, get_train_val_split
from data.augmentation import get_train_transforms, get_val_transforms
from training import Trainer
from utils.helpers import set_seed, get_device, create_output_dir


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
    
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


def setup_data_loaders(config: Dict[str, Any]) -> tuple:
    """Set up train and validation data loaders.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Get data paths
    data_dir = data_config.get('data_dir', 'data')
    image_dir = data_config.get('image_dir', os.path.join(data_dir, 'images'))
    text_dir = data_config.get('text_dir', os.path.join(data_dir, 'texts'))
    table_file = data_config.get('table_file', os.path.join(data_dir, 'clinical_data.csv'))
    
    # Get data split parameters
    val_ratio = data_config.get('val_ratio', 0.2)
    random_seed = training_config.get('seed', 42)
    
    # Get transforms
    train_transforms = get_train_transforms(
        image_size=data_config.get('image_size', (128, 128, 128)),
        use_augmentation=data_config.get('use_augmentation', True),
    )
    
    val_transforms = get_val_transforms(
        image_size=data_config.get('image_size', (128, 128, 128)),
    )
    
    # Create full dataset
    full_dataset = StrokeDataset(
        image_dir=image_dir,
        text_dir=text_dir,
        table_file=table_file,
        transform=None,  # Will be set per split
    )
    
    # Split into train and validation
    train_dataset, val_dataset = get_train_val_split(
        full_dataset,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )
    
    # Set transforms
    train_dataset.transform = train_transforms
    val_dataset.transform = val_transforms
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset, val_dataset


def setup_model(config: Dict[str, Any], device: torch.device) -> StrokePrognosisModel:
    """Set up the stroke prognosis model.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
    
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    # Get model parameters
    model = StrokePrognosisModel(
        # Image encoder
        image_encoder_name=model_config.get('image_encoder', 'resnet50_3d'),
        image_pretrained=model_config.get('image_pretrained', True),
        image_feature_dim=model_config.get('image_feature_dim', 512),
        
        # Text encoder
        text_encoder_name=model_config.get('text_encoder', 'biobert'),
        text_pretrained=model_config.get('text_pretrained', True),
        text_feature_dim=model_config.get('text_feature_dim', 768),
        
        # Table encoder
        table_input_dim=model_config.get('table_input_dim', 20),
        table_hidden_dims=model_config.get('table_hidden_dims', [128, 256, 256]),
        table_feature_dim=model_config.get('table_feature_dim', 256),
        
        # Fusion
        fusion_method=model_config.get('fusion_method', 'vdafm'),
        fusion_hidden_dim=model_config.get('fusion_hidden_dim', 512),
        low_rank=model_config.get('low_rank', 64),
        
        # Classification
        num_classes=model_config.get('num_classes', 1),
        dropout=model_config.get('dropout', 0.3),
    )
    
    model = model.to(device)
    
    return model


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train stroke prognosis model')
    parser.add_argument('--config', type=str, default='configs/default/base.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID (if not specified, auto-select)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Directory for saving outputs')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    training_config = config.get('training', {})
    seed = training_config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.experiment_name)
    print(f"Output directory: {output_dir}")
    
    # Save configuration to output directory
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")
    
    # Setup data loaders
    print("\nSetting up data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = setup_data_loaders(config)
    
    # Setup model
    print("\nInitializing model...")
    model = setup_model(config, device)
    
    # Print model summary
    from utils.helpers import print_model_summary
    print_model_summary(model, input_shapes={
        'image': (1, 1, 128, 128, 128),
        'text': (1, 512),
        'table': (1, 20),
    })
    
    # Setup trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=str(output_dir),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        from utils.helpers import load_checkpoint
        checkpoint_info = load_checkpoint(
            args.resume,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device=device,
        )
        trainer.start_epoch = checkpoint_info['epoch'] + 1
        print(f"Resuming from epoch {trainer.start_epoch}")
    
    # Train model
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model state...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            metrics=trainer.best_metrics,
            filename='interrupted_checkpoint.pth',
        )
        print("Checkpoint saved. You can resume training with --resume")
    
    # Load best model and evaluate
    print("\n" + "="*80)
    print("Training completed! Evaluating best model...")
    print("="*80 + "\n")
    
    best_checkpoint_path = output_dir / 'best_model.pth'
    if best_checkpoint_path.exists():
        from utils.helpers import load_checkpoint
        load_checkpoint(
            str(best_checkpoint_path),
            model=model,
            device=device,
        )
        
        # Final evaluation
        val_loss, val_metrics = trainer.validate()
        
        print("\nFinal Validation Results:")
        print(f"  Loss: {val_loss:.4f}")
        for metric_name, metric_value in val_metrics.items():
            print(f"  {metric_name.capitalize()}: {metric_value:.4f}")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
