# Cleaner AI - CAPTCHA Noise Removal System

Production-ready AI system for cleaning CAPTCHA images from noise, lines, artifacts, and background distortions using PyTorch and modern deep learning techniques.

## Features

- **Advanced Neural Architecture**: U-Net with residual blocks, attention mechanisms (CBAM), and feature pyramids
- **Comprehensive Dataset Generation**: Synthetic CAPTCHA generation with realistic noise, distortions, and artifacts
- **Multi-mode Training**: Support for GAN-based training with discriminator network
- **Real-time Inference**: Fast image cleaning with batch processing support
- **Professional GUI**: PyQt5-based interface with dark theme
- **Mixed Precision Training**: FP16 support for faster training and reduced memory usage
- **Checkpoint Management**: Automatic checkpoint saving and resuming
- **TensorBoard Integration**: Real-time training visualization
- **CUDA/CPU Support**: Automatic device detection and optimization

## Project Structure

```
cleaner_ai/
├── model.py              # Neural network architectures
├── train.py              # Training pipeline
├── infer.py              # Inference engines
├── captcha_generator.py  # Dataset generation
├── data_loader.py        # Data loading and augmentation
├── utils.py              # Utilities and helpers
├── cleaner_ai.py         # Main GUI application
├── run.py                # Entry point
├── requirements.txt      # Dependencies
└── data/
    ├── inp/              # Noisy CAPTCHA images
    ├── out/              # Noise+lines only
    ├── otv/              # Clean target images
    ├── models/           # Saved models
    └── logs/             # Training logs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application

```bash
python run.py
```

### Command Line Training

```python
from train import Trainer, Config

config = Config()
config.epochs = 100
config.batch_size = 16
config.learning_rate = 0.001

trainer = Trainer(config)
trainer.train()
```

### Dataset Generation

```python
from captcha_generator import CaptchaGenerator

generator = CaptchaGenerator(image_size=256, output_dir='data')
generator.generate_dataset(num_images=10000)
```

### Inference

```python
from infer import Inferencer

inferencer = Inferencer()
result = inferencer.infer_image('path/to/noisy/image.png')
```

## GUI Tabs

### Train
- Configure training parameters (epochs, batch size, learning rate, image size)
- Start/stop training
- Real-time loss monitoring
- Mixed precision training support

### Generate
- Generate synthetic CAPTCHA dataset
- Configure dataset size and image dimensions
- Progress tracking

### Test
- Load and test individual images
- Real-time inference
- Side-by-side comparison of input and output

### Viewer
- Browse generated dataset
- View inp/out/otv image triplets
- Navigate through dataset

### Logs
- View training logs
- Monitor system performance

## Model Architecture

### Generator (CleanerAIModel)
- Input: Noisy CAPTCHA image (3 channels)
- Encoder: 4 levels with residual blocks and CBAM attention
- Bottleneck: Dilated residual blocks with attention
- Decoder: 4 levels with skip connections
- Feature Pyramid: Multi-scale feature fusion
- Output: Clean CAPTCHA image (3 channels)

### Discriminator
- PatchGAN-style discriminator
- 5 convolutional blocks
- Binary classification (real vs fake)

## Training Details

- **Loss Functions**:
  - L1 Loss: 0.5 weight
  - L2 Loss: 0.3 weight
  - Perceptual Loss: 0.1 weight
  - Adversarial Loss: 0.1 weight

- **Optimization**:
  - AdamW optimizer
  - Cosine annealing learning rate scheduler
  - Gradient clipping (max norm: 1.0)
  - EMA weight averaging

- **Data Augmentation**:
  - Random brightness/contrast
  - Random blur and rotation
  - Random HSV shifts
  - Random sharpening
  - Random pixel noise

## Performance

- Supports batch processing for faster inference
- Mixed precision training reduces memory usage by ~50%
- Automatic checkpoint management
- Real-time progress tracking

## System Requirements

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for training)
- 10GB+ disk space for dataset

## Configuration

Edit `Config` class in `utils.py` to customize:
- Training hyperparameters
- Model architecture
- Data paths
- Device selection

## License

Proprietary - Cleaner AI System
