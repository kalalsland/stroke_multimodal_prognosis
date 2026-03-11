"""Test script to verify the installation and setup.

This script performs basic checks to ensure all components are properly configured.
"""
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from stroke_multimodal_prognosis.models import StrokePrognosisModel
        from stroke_multimodal_prognosis.models.backbones import ImageEncoder, TextEncoder, TableEncoder
        from stroke_multimodal_prognosis.models.fusion_modules import VDAFM
        from stroke_multimodal_prognosis.data import StrokeDataset
        from stroke_multimodal_prognosis.data.augmentation import get_train_transforms, get_val_transforms
        from stroke_multimodal_prognosis.training import Trainer
        from stroke_multimodal_prognosis.utils.metrics import compute_metrics
        from stroke_multimodal_prognosis.utils.helpers import set_seed, get_device
        
        print("[PASS] All imports successful!")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from stroke_multimodal_prognosis.models import StrokePrognosisModel
        from stroke_multimodal_prognosis.models.fusion_modules.vdafm import VDAFMConfig
        
        # Configure model components
        image_config = {
            'encoder_name': 'simple_3d',
            'in_channels': 1,
            'base_channels': 32,
            'output_dim': 512,
        }
        
        text_config = {
            'encoder_name': 'simple',
            'input_dim': 768,
            'output_dim': 512,
        }
        
        table_config = {
            'encoder_name': 'simple_mlp',
            'input_dim': 20,
            'hidden_dims': [128, 256],
            'output_dim': 512,
            'dropout_rate': 0.3,
        }
        
        vdafm_config = VDAFMConfig(
            embed_dim=512,
            num_heads=8,
            low_rank_dim=64,
        )
        
        model = StrokePrognosisModel(
            image_config=image_config,
            text_config=text_config,
            table_config=table_config,
            vdafm_config=vdafm_config,
            num_classes=2,
        )
        
        print(f"[PASS] Model created successfully!")
        param_counts = model.get_num_parameters()
        print(f"  Total parameters: {param_counts['total']:,}")
        for component, count in param_counts.items():
            if component != 'total':
                print(f"    {component}: {count:,}")
        return True
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")
    
    try:
        from stroke_multimodal_prognosis.models import StrokePrognosisModel
        from stroke_multimodal_prognosis.models.fusion_modules.vdafm import VDAFMConfig
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")
        
        # Configure model
        image_config = {
            'encoder_name': 'simple_3d',
            'in_channels': 1,
            'base_channels': 32,
            'output_dim': 512,
        }
        
        text_config = {
            'encoder_name': 'simple',
            'input_dim': 768,
            'output_dim': 512,
        }
        
        table_config = {
            'encoder_name': 'simple_mlp',
            'input_dim': 20,
            'hidden_dims': [128, 256],
            'output_dim': 512,
            'dropout_rate': 0.3,
        }
        
        vdafm_config = VDAFMConfig(
            embed_dim=512,
            num_heads=8,
            low_rank_dim=64,
        )
        
        model = StrokePrognosisModel(
            image_config=image_config,
            text_config=text_config,
            table_config=table_config,
            vdafm_config=vdafm_config,
            num_classes=2,
        ).to(device)
        
        # Create dummy input
        batch_size = 2
        dummy_image = torch.randn(batch_size, 1, 32, 32, 32).to(device)  # Small size for test
        dummy_text = torch.randn(batch_size, 768).to(device)
        dummy_table = torch.randn(batch_size, 20).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_image, dummy_text, dummy_table)
        
        print(f"[PASS] Forward pass successful!")
        print(f"  Input shapes:")
        print(f"    Image: {dummy_image.shape}")
        print(f"    Text: {dummy_text.shape}")
        print(f"    Table: {dummy_table.shape}")
        print(f"  Output shape: {outputs['logits'].shape}")
        print(f"  Output keys: {list(outputs.keys())}")
        return True
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoders():
    """Test individual encoders."""
    print("\nTesting individual encoders...")
    
    try:
        from stroke_multimodal_prognosis.models.backbones import ImageEncoder, TextEncoder, TableEncoder
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test image encoder
        image_encoder = ImageEncoder(
            encoder_name='resnet50_3d',
            pretrained=False,
            feature_dim=512,
        ).to(device)
        
        dummy_image = torch.randn(1, 1, 32, 32, 32).to(device)
        image_feat = image_encoder(dummy_image)
        print(f"[PASS] Image encoder: {dummy_image.shape} -> {image_feat.shape}")
        
        # Test text encoder
        text_encoder = TextEncoder(
            encoder_name='biobert',
            pretrained=False,
            feature_dim=768,
        ).to(device)
        
        dummy_text = torch.randn(1, 512).to(device)
        text_feat = text_encoder(dummy_text)
        print(f"[PASS] Text encoder: {dummy_text.shape} -> {text_feat.shape}")
        
        # Test table encoder
        table_encoder = TableEncoder(
            input_dim=20,
            hidden_dims=[128, 256],
            feature_dim=256,
            dropout=0.3,
        ).to(device)
        
        dummy_table = torch.randn(1, 20).to(device)
        table_feat = table_encoder(dummy_table)
        print(f"[PASS] Table encoder: {dummy_table.shape} -> {table_feat.shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_module():
    """Test VDAFM fusion module."""
    print("\nTesting VDAFM fusion module...")
    
    try:
        from stroke_multimodal_prognosis.models.fusion_modules import VDAFM
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        fusion = VDAFM(
            input_dims={'image': 512, 'text': 768, 'table': 256},
            hidden_dim=512,
            low_rank=64,
        ).to(device)
        
        # Dummy features
        image_feat = torch.randn(2, 512).to(device)
        text_feat = torch.randn(2, 768).to(device)
        table_feat = torch.randn(2, 256).to(device)
        
        fused = fusion(image_feat, text_feat, table_feat)
        
        print(f"[PASS] VDAFM fusion: ({image_feat.shape}, {text_feat.shape}, {table_feat.shape}) -> {fused.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")
    
    try:
        from stroke_multimodal_prognosis.utils.metrics import compute_metrics
        
        # Dummy predictions
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0.2, 0.8, 0.3, 0.9, 0.7, 0.1, 0.6, 0.4])
        
        metrics = compute_metrics(y_true, y_pred)
        
        print(f"[PASS] Metrics computed successfully!")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        return True
    except Exception as e:
        print(f"[FAIL] Metrics test failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from stroke_multimodal_prognosis.utils.helpers import set_seed, get_device
        
        # Test set_seed
        set_seed(42)
        r1 = torch.rand(1).item()
        set_seed(42)
        r2 = torch.rand(1).item()
        assert r1 == r2, "set_seed not working properly"
        print(f"[PASS] set_seed working correctly")
        
        # Test get_device
        device = get_device()
        print(f"[PASS] get_device: {device}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Utility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("STROKE MULTIMODAL PROGNOSIS - SETUP TEST")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Encoders", test_encoders),
        ("Fusion Module", test_fusion_module),
        ("Forward Pass", test_forward_pass),
        ("Metrics", test_metrics),
        ("Utilities", test_utils),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name} encountered unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\nAll tests passed! Your setup is ready.")
    else:
        print("\nSome tests failed. Please check the errors above.")
    
    print("="*80)


if __name__ == '__main__':
    main()
