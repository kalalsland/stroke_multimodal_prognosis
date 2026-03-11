"""Example usage of the stroke prognosis model.

This script demonstrates various usage patterns:
1. Simple training with default configuration
2. Custom model configuration
3. Inference on new data
4. Model evaluation
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models import StrokePrognosisModel
from data import StrokeDataset
from data.augmentation import get_val_transforms
from utils.helpers import set_seed, get_device, load_checkpoint
from utils.metrics import compute_metrics


def example_1_simple_training():
    """Example 1: Simple training with default settings."""
    print("="*80)
    print("Example 1: Simple Training")
    print("="*80)
    
    # This is the simplest way to train the model
    # Just run the train.py script with default config
    print("""
    To train with default settings:
    
    $ python train.py --config configs/default/base.yaml
    
    This will:
    - Load data from paths specified in config
    - Initialize model with default architecture
    - Train for specified epochs
    - Save results to experiments/ directory
    """)


def example_2_custom_configuration():
    """Example 2: Creating model with custom configuration."""
    print("="*80)
    print("Example 2: Custom Model Configuration")
    print("="*80)
    
    # Set random seed
    set_seed(42)
    device = get_device()
    
    # Create model with custom parameters
    model = StrokePrognosisModel(
        # Image encoder configuration
        image_encoder_name='resnet50_3d',
        image_pretrained=True,
        image_feature_dim=512,
        
        # Text encoder configuration
        text_encoder_name='biobert',
        text_pretrained=True,
        text_feature_dim=768,
        
        # Table encoder configuration
        table_input_dim=20,  # Number of clinical features
        table_hidden_dims=[128, 256, 256],
        table_feature_dim=256,
        
        # Fusion configuration
        fusion_method='vdafm',  # Options: 'vdafm', 'concat', 'attention'
        fusion_hidden_dim=512,
        low_rank=64,  # For VDAFM low-rank projection
        
        # Classification head
        num_classes=1,  # Binary classification (good/poor prognosis)
        dropout=0.3,
    ).to(device)
    
    print(f"\nModel created successfully!")
    print(f"Device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy input to test forward pass
    batch_size = 2
    dummy_image = torch.randn(batch_size, 1, 128, 128, 128).to(device)
    dummy_text = torch.randn(batch_size, 512).to(device)  # Pre-encoded text features
    dummy_table = torch.randn(batch_size, 20).to(device)  # Clinical features
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_image, dummy_text, dummy_table)
    
    print(f"\nForward pass successful!")
    print(f"Input shapes:")
    print(f"  Image: {dummy_image.shape}")
    print(f"  Text: {dummy_text.shape}")
    print(f"  Table: {dummy_table.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().cpu().numpy()}")


def example_3_inference():
    """Example 3: Inference on new data."""
    print("\n" + "="*80)
    print("Example 3: Model Inference")
    print("="*80)
    
    device = get_device()
    
    # Load trained model
    model = StrokePrognosisModel(
        image_encoder_name='resnet50_3d',
        text_encoder_name='biobert',
        table_input_dim=20,
        num_classes=1,
    ).to(device)
    
    # In practice, you would load from checkpoint:
    # checkpoint_path = 'experiments/my_experiment/best_model.pth'
    # load_checkpoint(checkpoint_path, model=model, device=device)
    
    print("\nModel loaded successfully!")
    
    # Prepare single sample for inference
    # In practice, these would come from your dataset
    image = torch.randn(1, 1, 128, 128, 128).to(device)
    text = torch.randn(1, 512).to(device)
    table = torch.randn(1, 20).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        logit = model(image, text, table)
        probability = torch.sigmoid(logit)
    
    print(f"\nInference Results:")
    print(f"  Logit: {logit.item():.4f}")
    print(f"  Probability: {probability.item():.4f}")
    print(f"  Prediction: {'Poor Prognosis' if probability.item() > 0.5 else 'Good Prognosis'}")


def example_4_batch_evaluation():
    """Example 4: Batch evaluation with metrics."""
    print("\n" + "="*80)
    print("Example 4: Batch Evaluation")
    print("="*80)
    
    device = get_device()
    
    # Create model
    model = StrokePrognosisModel(
        image_encoder_name='resnet50_3d',
        text_encoder_name='biobert',
        table_input_dim=20,
        num_classes=1,
    ).to(device)
    
    # Simulate batch of predictions
    batch_size = 10
    images = torch.randn(batch_size, 1, 128, 128, 128).to(device)
    texts = torch.randn(batch_size, 512).to(device)
    tables = torch.randn(batch_size, 20).to(device)
    labels = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(images, texts, tables).squeeze()
        probabilities = torch.sigmoid(logits)
    
    # Compute metrics
    metrics = compute_metrics(
        y_true=labels.cpu().numpy(),
        y_pred=probabilities.cpu().numpy(),
    )
    
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.upper()}: {metric_value:.4f}")


def example_5_custom_dataset():
    """Example 5: Creating custom dataset."""
    print("\n" + "="*80)
    print("Example 5: Custom Dataset Usage")
    print("="*80)
    
    print("""
    To use your own data:
    
    1. Organize your data:
       data/
       ├── images/
       │   ├── patient_001.nii.gz
       │   ├── patient_002.nii.gz
       │   └── ...
       ├── texts/
       │   ├── patient_001.txt
       │   ├── patient_002.txt
       │   └── ...
       └── clinical_data.csv
    
    2. clinical_data.csv should have columns:
       - patient_id: Unique identifier
       - label: 0 (good prognosis) or 1 (poor prognosis)
       - feature_1, feature_2, ...: Clinical features
    
    3. Create dataset:
    """)
    
    from data import StrokeDataset
    from data.augmentation import get_val_transforms
    
    # Example of creating dataset
    dataset = StrokeDataset(
        image_dir='data/images',
        text_dir='data/texts',
        table_file='data/clinical_data.csv',
        transform=get_val_transforms(image_size=(128, 128, 128)),
    )
    
    print(f"\n    dataset = StrokeDataset(...)")
    print(f"    # Total samples: {len(dataset)}")
    print(f"\n    # Get single sample:")
    print(f"    sample = dataset[0]")
    print(f"    image = sample['image']  # Shape: (1, 128, 128, 128)")
    print(f"    text = sample['text']    # Shape: (768,) - BioBERT features")
    print(f"    table = sample['table']  # Shape: (20,) - Clinical features")
    print(f"    label = sample['label']  # 0 or 1")


def example_6_training_with_validation():
    """Example 6: Training with validation monitoring."""
    print("\n" + "="*80)
    print("Example 6: Training with Validation")
    print("="*80)
    
    print("""
    For training with proper validation:
    
    1. Modify config file (configs/default/base.yaml):
       
       data:
         data_dir: 'path/to/your/data'
         val_ratio: 0.2  # 20% for validation
       
       training:
         epochs: 100
         batch_size: 16
         learning_rate: 0.0001
         early_stopping_patience: 15
    
    2. Run training:
       
       $ python train.py --config configs/default/base.yaml \\
                        --experiment_name my_experiment \\
                        --gpu 0
    
    3. Monitor training:
       - Loss curves saved to: experiments/my_experiment/loss_curves.png
       - Best model saved to: experiments/my_experiment/best_model.pth
       - Training log: experiments/my_experiment/training_log.txt
    
    4. Resume training if interrupted:
       
       $ python train.py --resume experiments/my_experiment/best_model.pth
    """)


def example_7_hyperparameter_tuning():
    """Example 7: Hyperparameter tuning tips."""
    print("\n" + "="*80)
    print("Example 7: Hyperparameter Tuning")
    print("="*80)
    
    print("""
    Key hyperparameters to tune:
    
    1. Model Architecture:
       - fusion_hidden_dim: [256, 512, 1024]
       - low_rank: [32, 64, 128]
       - dropout: [0.1, 0.3, 0.5]
    
    2. Training:
       - learning_rate: [1e-5, 1e-4, 1e-3]
       - batch_size: [8, 16, 32]
       - weight_decay: [1e-5, 1e-4, 1e-3]
    
    3. Data Augmentation:
       - rotation_range: [0, 5, 10] degrees
       - scale_range: [0.9, 1.0, 1.1]
       - use_augmentation: [True, False]
    
    Tips:
    - Start with default configuration
    - Tune one parameter at a time
    - Use validation set to select best configuration
    - Consider using tools like Optuna for automated search
    """)


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("STROKE PROGNOSIS MODEL - USAGE EXAMPLES")
    print("="*80)
    
    # Run examples
    example_1_simple_training()
    example_2_custom_configuration()
    example_3_inference()
    example_4_batch_evaluation()
    example_5_custom_dataset()
    example_6_training_with_validation()
    example_7_hyperparameter_tuning()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("For more information, see README.md")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
