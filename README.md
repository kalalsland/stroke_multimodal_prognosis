# Stroke Multimodal Prognosis Prediction

A PyTorch-based deep learning framework for stroke prognosis prediction using multimodal medical data (MRI images, clinical reports, and tabular data).

## 🌟 Features

- **Multimodal Learning**: Integrates 3D MRI images, clinical text reports, and tabular clinical data
- **Advanced Fusion**: Variable-Density Adaptive Fusion Module (VDAFM) with low-rank projection
- **Professional Architecture**: Clean, modular, and extensible codebase following best practices
- **Type Safety**: Full type annotations for better code quality
- **Comprehensive Documentation**: Google-style docstrings with mathematical explanations
- **Flexible Configuration**: YAML-based configuration system
- **Training Pipeline**: Complete training, validation, and evaluation pipeline
- **Data Augmentation**: Built-in augmentation strategies for medical images

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage Examples](#usage-examples)
- [Advanced Topics](#advanced-topics)
- [Contributing](#contributing)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kalalsland/stroke_multimodal_prognosis.git
cd stroke_multimodal_prognosis

# Install required packages
pip install -r requirements.txt
```

### Additional Setup

1. **Download Pre-trained Models**:
   - BioBERT: Place in `pretrained/biobert/`
   - ResNet-50 3D: Place in `pretrained/resnet50_3d/`

2. **Prepare Your Data**:
   ```
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
   ```

## 📁 Project Structure

```
stroke_multimodal_prognosis/
├── models/                      # Model definitions
│   ├── backbones/              # Encoder modules
│   │   ├── image_encoder.py   # 3D ResNet-50
│   │   ├── text_encoder.py    # BioBERT
│   │   └── table_encoder.py   # Residual MLP
│   ├── fusion_modules/         # Fusion strategies
│   │   └── vdafm.py           # VDAFM module
│   └── multimodal_model.py    # Complete model
├── data/                       # Data handling
│   ├── datasets.py            # Dataset classes
│   └── augmentation.py        # Data augmentation
├── training/                   # Training utilities
│   └── trainer.py             # Trainer class
├── utils/                      # Utility functions
│   ├── metrics.py             # Evaluation metrics
│   └── helpers.py             # Helper functions
├── configs/                    # Configuration files
│   └── default/               # Default configs
├── train.py                    # Main training script
├── example_usage.py           # Usage examples
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python train.py --config configs/default/base.yaml

# Train with custom GPU
python train.py --config configs/default/base.yaml --gpu 0

# Train with custom experiment name
python train.py --config configs/default/base.yaml --experiment_name my_experiment
```

### 2. Custom Configuration

Create a custom config file:

```yaml
# configs/my_config.yaml
data:
  data_dir: 'path/to/your/data'
  image_size: [128, 128, 128]
  val_ratio: 0.2

model:
  fusion_method: 'vdafm'
  low_rank: 64
  dropout: 0.3

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
```

Then train:

```bash
python train.py --config configs/my_config.yaml
```

### 3. Resume Training

```bash
python train.py --resume experiments/my_experiment/best_model.pth
```

## ⚙️ Configuration

The configuration system uses YAML files organized hierarchically:

### Data Configuration (`configs/default/data.yaml`)

```yaml
data:
  data_dir: 'data'                    # Root data directory
  image_dir: 'data/images'            # MRI images
  text_dir: 'data/texts'              # Clinical reports
  table_file: 'data/clinical_data.csv' # Tabular data
  
  image_size: [128, 128, 128]         # Target image size
  val_ratio: 0.2                       # Validation split ratio
  
  # Data augmentation
  use_augmentation: true
  rotation_range: 5                    # Degrees
  scale_range: [0.9, 1.1]
  noise_std: 0.01
```

### Model Configuration (`configs/default/model.yaml`)

```yaml
model:
  # Image encoder
  image_encoder: 'resnet50_3d'
  image_pretrained: true
  image_feature_dim: 512
  
  # Text encoder
  text_encoder: 'biobert'
  text_pretrained: true
  text_feature_dim: 768
  
  # Table encoder
  table_input_dim: 20
  table_hidden_dims: [128, 256, 256]
  table_feature_dim: 256
  
  # Fusion module
  fusion_method: 'vdafm'  # Options: vdafm, concat, attention
  fusion_hidden_dim: 512
  low_rank: 64
  
  # Classification
  num_classes: 1
  dropout: 0.3
```

### Training Configuration (`configs/default/training.yaml`)

```yaml
training:
  # Optimization
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 0.0001
  
  # Learning rate schedule
  lr_scheduler: 'cosine'  # Options: step, cosine, plateau
  lr_warmup_epochs: 5
  lr_min: 0.00001
  
  # Early stopping
  early_stopping_patience: 15
  
  # Loss weights
  loss_weights:
    classification: 1.0
    contrastive: 0.1
    alignment: 0.1
  
  # Other
  seed: 42
  num_workers: 4
  log_interval: 10
```

## 🏗️ Model Architecture

### Overview

The model consists of three main components:

1. **Encoders**: Extract features from each modality
2. **Fusion Module**: Combine multimodal features
3. **Classification Head**: Predict prognosis

```
┌─────────────┐
│ MRI Image   │─┐
│ (3D Volume) │ │  
└─────────────┘ │
                ├─→ ┌──────────┐     ┌─────────────┐     ┌──────────┐
┌─────────────┐ │   │          │     │   VDAFM     │     │Classifier│
│  Clinical   │─┼─→ │ Encoders │ ──→ │   Fusion    │ ──→ │  Head    │ ──→ Prediction
│   Report    │ │   │          │     │   Module    │     │          │
└─────────────┘ │   └──────────┘     └─────────────┘     └──────────┘
                │
┌─────────────┐ │
│  Clinical   │─┘
│    Data     │
└─────────────┘
```

### Encoders

#### 1. Image Encoder (3D ResNet-50)
- Processes 3D MRI volumes
- Pre-trained on medical imaging data
- Output: 512-dimensional feature vector

#### 2. Text Encoder (BioBERT)
- Processes clinical reports
- Pre-trained on biomedical literature
- Output: 768-dimensional feature vector

#### 3. Table Encoder (Residual MLP)
- Processes structured clinical data
- Multiple residual blocks with batch normalization
- Output: 256-dimensional feature vector

### VDAFM Fusion Module

The Variable-Density Adaptive Fusion Module (VDAFM) uses:

1. **Low-Rank Projection**: Reduces feature dimensionality while preserving information
   ```
   Z_i = W_i^low · X_i
   where W_i^low ∈ R^{d×r}, r << d
   ```

2. **Projection Calibration**: Adaptively weights modality importance
   ```
   α_i = softmax(MLP(Z_i))
   Z_i' = α_i ⊙ Z_i
   ```

3. **Cross-Modal Attention**: Captures inter-modality relationships

## 🎓 Training

### Basic Training

```bash
python train.py --config configs/default/base.yaml
```

### Monitor Training

Training progress is logged to:
- Console output
- `experiments/{experiment_name}/training_log.txt`
- Loss curves: `experiments/{experiment_name}/loss_curves.png`

### Training Outputs

```
experiments/{experiment_name}/
├── config.yaml              # Configuration used
├── best_model.pth          # Best model checkpoint
├── latest_checkpoint.pth   # Latest checkpoint
├── training_log.txt        # Training logs
├── loss_curves.png         # Loss visualization
└── metrics.json            # Training metrics
```

### Early Stopping

Training automatically stops if validation loss doesn't improve for `early_stopping_patience` epochs.

## 📊 Evaluation

### Metrics

The following metrics are computed:

- **AUC-ROC**: Area under the ROC curve
- **Accuracy**: Classification accuracy
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Average of sensitivity and specificity

### Evaluate Trained Model

```python
from models import StrokePrognosisModel
from utils.helpers import load_checkpoint
from utils.metrics import compute_metrics

# Load model
model = StrokePrognosisModel(...)
load_checkpoint('experiments/my_experiment/best_model.pth', model=model)

# Get predictions
model.eval()
with torch.no_grad():
    predictions = model(images, texts, tables)

# Compute metrics
metrics = compute_metrics(y_true, y_pred)
print(metrics)
```

## 💡 Usage Examples

### Example 1: Simple Inference

```python
import torch
from models import StrokePrognosisModel
from utils.helpers import load_checkpoint

# Load model
model = StrokePrognosisModel(num_classes=1)
load_checkpoint('best_model.pth', model=model)

# Prepare input
image = torch.randn(1, 1, 128, 128, 128)  # MRI scan
text = torch.randn(1, 512)                 # Encoded report
table = torch.randn(1, 20)                 # Clinical features

# Inference
model.eval()
with torch.no_grad():
    logit = model(image, text, table)
    probability = torch.sigmoid(logit)

print(f"Prognosis probability: {probability.item():.2%}")
```

### Example 2: Custom Dataset

```python
from data import StrokeDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = StrokeDataset(
    image_dir='data/images',
    text_dir='data/texts',
    table_file='data/clinical_data.csv',
)

# Create data loader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate
for batch in loader:
    images = batch['image']
    texts = batch['text']
    tables = batch['table']
    labels = batch['label']
    # ... training code ...
```

### Example 3: Custom Training Loop

```python
from training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device,
    output_dir='experiments/my_experiment',
)

trainer.train()
```

See `example_usage.py` for more detailed examples.

## 🔬 Advanced Topics

### Custom Fusion Strategies

Implement your own fusion module:

```python
from models.fusion_modules import BaseFusion

class MyFusion(BaseFusion):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        # Your implementation
    
    def forward(self, image_feat, text_feat, table_feat):
        # Your fusion logic
        return fused_features
```

### Hyperparameter Tuning

Use tools like Optuna for automated hyperparameter search:

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    # ... train and return validation metric
    return val_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Multi-GPU Training

```python
# Wrap model with DataParallel
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### Export for Deployment

```python
# Export to ONNX
torch.onnx.export(
    model,
    (dummy_image, dummy_text, dummy_table),
    'model.onnx',
    input_names=['image', 'text', 'table'],
    output_names=['prediction'],
)
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function arguments and returns
- Write Google-style docstrings
- Add unit tests for new features

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Multimodal Deep Learning for Stroke Prognosis Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- BioBERT: https://github.com/dmis-lab/biobert
- 3D ResNet: https://github.com/Tencent/MedicalNet
- PyTorch: https://pytorch.org/

## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: your.email@example.com

## 🔄 Updates

### Version 1.0.0 (2024-03-11)
- Initial release
- Complete multimodal architecture
- VDAFM fusion module
- Training and evaluation pipeline
- Comprehensive documentation

---

**Happy modeling! 🎉**
