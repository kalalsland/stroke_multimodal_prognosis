"""Dataset classes for stroke prognosis multimodal data loading.

This module provides PyTorch Dataset classes for loading and processing
MRI images, clinical text, and tabular data for stroke patients.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import nibabel as nib


class StrokeMultimodalDataset(Dataset):
    """Dataset for multimodal stroke prognosis data.
    
    Loads and processes three modalities:
        1. MRI images (NIfTI format)
        2. Clinical text reports
        3. Tabular clinical features
    
    Args:
        data_csv: Path to CSV file containing patient information with columns:
            - patient_id: Unique identifier
            - image_path: Path to MRI NIfTI file (relative to image_dir)
            - clinical_text: Clinical report text
            - label: Binary outcome label (0=good, 1=poor)
            - Additional columns for tabular features
        image_dir: Root directory for MRI images
        tabular_features: List of column names to use as tabular features
        tokenizer_name: HuggingFace tokenizer name (default: "dmis-lab/biobert-v1.1")
        max_text_length: Maximum text sequence length (default: 512)
        image_size: Target image size (D, H, W) for resizing (default: (32, 128, 128))
        transform: Optional image augmentation transform
        normalize_tabular: Whether to normalize tabular features (default: True)
    
    Example:
        >>> dataset = StrokeMultimodalDataset(
        ...     data_csv='data/train.csv',
        ...     image_dir='data/mri_scans/',
        ...     tabular_features=['age', 'nihss_score', 'glucose'],
        ...     tokenizer_name='dmis-lab/biobert-v1.1',
        ...     max_text_length=512
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['image', 'text_input_ids', 'text_attention_mask', 
                   'tabular', 'label', 'patient_id'])
    """
    
    def __init__(
        self,
        data_csv: str,
        image_dir: str,
        tabular_features: List[str],
        tokenizer_name: str = "dmis-lab/biobert-v1.1",
        max_text_length: int = 512,
        image_size: Tuple[int, int, int] = (32, 128, 128),
        transform: Optional[object] = None,
        normalize_tabular: bool = True,
    ):
        super().__init__()
        
        # Load data CSV
        self.data_df = pd.read_csv(data_csv)
        self.image_dir = Path(image_dir)
        self.tabular_features = tabular_features
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.transform = transform
        
        # Validate required columns
        required_cols = ['patient_id', 'image_path', 'clinical_text', 'label']
        missing_cols = [col for col in required_cols if col not in self.data_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate tabular features exist
        missing_features = [f for f in tabular_features if f not in self.data_df.columns]
        if missing_features:
            raise ValueError(f"Missing tabular features: {missing_features}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Normalize tabular features if requested
        if normalize_tabular:
            self._normalize_tabular_features()
        
        # Store statistics
        self.num_samples = len(self.data_df)
        self.num_tabular_features = len(tabular_features)
    
    def _normalize_tabular_features(self) -> None:
        """Normalize tabular features using z-score normalization.
        
        Computes mean and std for each feature and applies standardization:
            z = (x - μ) / σ
        """
        self.tabular_mean = self.data_df[self.tabular_features].mean().values
        self.tabular_std = self.data_df[self.tabular_features].std().values
        
        # Avoid division by zero
        self.tabular_std = np.where(self.tabular_std == 0, 1.0, self.tabular_std)
        
        # Apply normalization
        normalized_values = (
            self.data_df[self.tabular_features].values - self.tabular_mean
        ) / self.tabular_std
        
        self.data_df[self.tabular_features] = normalized_values
    
    def _load_mri_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess MRI image from NIfTI file.
        
        Args:
            image_path: Relative path to NIfTI file
        
        Returns:
            Preprocessed image array of shape (D, H, W)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image loading fails
        """
        full_path = self.image_dir / image_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"MRI image not found: {full_path}")
        
        try:
            # Load NIfTI image
            nii_img = nib.load(str(full_path))
            img_data = nii_img.get_fdata()
            
            # Handle 4D images (take first volume)
            if img_data.ndim == 4:
                img_data = img_data[..., 0]
            
            # Normalize intensity to [0, 1]
            img_data = self._normalize_intensity(img_data)
            
            # Resize to target size
            img_data = self._resize_image(img_data, self.image_size)
            
            return img_data
            
        except Exception as e:
            raise ValueError(f"Failed to load image {full_path}: {str(e)}")
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity using percentile-based normalization.
        
        Args:
            image: Raw image array
        
        Returns:
            Normalized image in range [0, 1]
        """
        # Clip extreme values using percentiles
        p_low, p_high = np.percentile(image, [1, 99])
        image = np.clip(image, p_low, p_high)
        
        # Min-max normalization
        image_min = image.min()
        image_max = image.max()
        
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        
        return image
    
    def _resize_image(
        self, image: np.ndarray, target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Resize 3D image to target size using scipy interpolation.
        
        Args:
            image: Input image of shape (D, H, W)
            target_size: Target dimensions (D', H', W')
        
        Returns:
            Resized image of shape target_size
        """
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = [
            target_size[i] / image.shape[i] for i in range(3)
        ]
        
        # Apply interpolation
        resized = zoom(image, zoom_factors, order=1)
        
        return resized
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single multimodal sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary containing:
                - image: MRI tensor [1, D, H, W]
                - text_input_ids: Tokenized text [seq_len]
                - text_attention_mask: Attention mask [seq_len]
                - tabular: Clinical features [num_features]
                - label: Binary outcome label (scalar)
                - patient_id: Patient identifier (string)
        
        Raises:
            IndexError: If idx is out of range
            FileNotFoundError: If MRI image file is missing
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Get row data
        row = self.data_df.iloc[idx]
        
        # Load MRI image
        image = self._load_mri_image(row['image_path'])
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Tokenize clinical text
        text_encoding = self.tokenizer(
            row['clinical_text'],
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract tabular features
        tabular = torch.tensor(
            row[self.tabular_features].values.astype(np.float32)
        )
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'image': image,
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'tabular': tabular,
            'label': label,
            'patient_id': row['patient_id'],
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Compute class distribution for balancing.
        
        Returns:
            Dictionary with class counts and percentages
        """
        label_counts = self.data_df['label'].value_counts().to_dict()
        total = sum(label_counts.values())
        
        return {
            'counts': label_counts,
            'percentages': {k: v / total * 100 for k, v in label_counts.items()},
            'total': total,
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with various statistics
        """
        return {
            'num_samples': self.num_samples,
            'num_tabular_features': self.num_tabular_features,
            'tabular_feature_names': self.tabular_features,
            'image_size': self.image_size,
            'max_text_length': self.max_text_length,
            'class_distribution': self.get_class_distribution(),
        }


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    image_dir: str,
    tabular_features: List[str],
    batch_size: int = 16,
    num_workers: int = 4,
    tokenizer_name: str = "dmis-lab/biobert-v1.1",
    max_text_length: int = 512,
    image_size: Tuple[int, int, int] = (32, 128, 128),
    train_transform: Optional[object] = None,
    val_transform: Optional[object] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train, validation, and test data loaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        image_dir: Root directory for MRI images
        tabular_features: List of tabular feature column names
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        tokenizer_name: HuggingFace tokenizer name
        max_text_length: Maximum text sequence length
        image_size: Target image dimensions (D, H, W)
        train_transform: Optional augmentation for training
        val_transform: Optional transform for validation/test
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Example:
        >>> train_loader, val_loader, test_loader = create_data_loaders(
        ...     train_csv='data/train.csv',
        ...     val_csv='data/val.csv',
        ...     test_csv='data/test.csv',
        ...     image_dir='data/mri_scans/',
        ...     tabular_features=['age', 'nihss_score'],
        ...     batch_size=16
        ... )
    """
    # Create datasets
    train_dataset = StrokeMultimodalDataset(
        data_csv=train_csv,
        image_dir=image_dir,
        tabular_features=tabular_features,
        tokenizer_name=tokenizer_name,
        max_text_length=max_text_length,
        image_size=image_size,
        transform=train_transform,
        normalize_tabular=True,
    )
    
    val_dataset = StrokeMultimodalDataset(
        data_csv=val_csv,
        image_dir=image_dir,
        tabular_features=tabular_features,
        tokenizer_name=tokenizer_name,
        max_text_length=max_text_length,
        image_size=image_size,
        transform=val_transform,
        normalize_tabular=True,
    )
    
    test_dataset = StrokeMultimodalDataset(
        data_csv=test_csv,
        image_dir=image_dir,
        tabular_features=tabular_features,
        tokenizer_name=tokenizer_name,
        max_text_length=max_text_length,
        image_size=image_size,
        transform=val_transform,
        normalize_tabular=True,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
