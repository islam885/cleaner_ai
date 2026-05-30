# Cleaner AI - Architecture Documentation

## System Overview

Cleaner AI is a production-ready deep learning system for removing noise, artifacts, and distortions from CAPTCHA images. The system consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Cleaner AI System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Generator  │  │   Training   │  │  Inference   │       │
│  │   (Synthetic)│  │   (PyTorch)  │  │  (Real-time) │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                  │                  │               │
│         └──────────────────┼──────────────────┘               │
│                            │                                  │
│                    ┌───────▼────────┐                        │
│                    │   Data Loader  │                        │
│                    │  (Augmentation)│                        │
│                    └────────────────┘                        │
│                            │                                  │
│         ┌──────────────────┼──────────────────┐              │
│         │                  │                  │              │
│    ┌────▼────┐        ┌────▼────┐       ┌────▼────┐         │
│    │  Model  │        │   Loss  │       │  Utils  │         │
│    │ (U-Net) │        │ Function│       │ (Config)│         │
│    └─────────┘        └─────────┘       └─────────┘         │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              GUI Interface (PyQt5)                   │   │
│  │  ┌────────┬────────┬────────┬────────┬────────┐     │   │
│  │  │ Train  │Generate│ Test   │Viewer  │ Logs   │     │   │
│  │  └────────┴────────┴────────┴────────┴────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Architecture (model.py)

#### Generator Network (CleanerAIModel)

**Encoder Path:**
- Input: 3-channel RGB image (256x256)
- 4 encoder blocks with progressive downsampling
- Each block: Conv → BatchNorm → ReLU → CBAM Attention → MaxPool
- Channel progression: 64 → 64 → 128 → 256 → 512

**Bottleneck:**
- 3 dilated residual blocks (dilation: 1, 2, 4)
- CBAM attention module
- Preserves spatial information while capturing context

**Decoder Path:**
- 4 decoder blocks with progressive upsampling
- Skip connections from encoder
- Each block: ConvTranspose → Concat → Conv → BatchNorm → ReLU → CBAM
- Channel progression: 512 → 256 → 128 → 64 → 64

**Feature Pyramid:**
- Multi-scale feature fusion
- Lateral convolutions for dimension matching
- Smooth convolutions for refinement

**Output:**
- 3-channel RGB image (256x256)
- Tanh activation for normalized output [-1, 1]

#### Attention Mechanisms

**CBAM (Convolutional Block Attention Module):**
- Channel Attention: Adaptive pooling + FC layers
- Spatial Attention: Conv on concatenated avg/max pooling
- Applied at each encoder/decoder block

**Discriminator Network:**
- PatchGAN-style architecture
- 5 convolutional blocks
- Binary classification (real vs fake)
- Used for adversarial training

### 2. Training Pipeline (train.py)

**Loss Functions:**
```
Total Loss = 0.5*L1 + 0.3*L2 + 0.1*Perceptual + 0.1*Adversarial
```

- **L1 Loss**: Pixel-level reconstruction
- **L2 Loss**: Smooth reconstruction
- **Perceptual Loss**: Feature-level similarity
- **Adversarial Loss**: GAN-based realism

**Optimization:**
- AdamW optimizer with β₁=0.9, β₂=0.999
- Cosine annealing learning rate scheduler
- Gradient clipping (max norm: 1.0)
- EMA weight averaging for stability

**Training Features:**
- Mixed precision (FP16) support
- Automatic checkpoint saving
- TensorBoard logging
- Validation on separate dataset
- Early stopping capability

### 3. Data Generation (captcha_generator.py)

**Three-Image Generation:**

1. **inp/ (Noisy CAPTCHA)**
   - Clean text + noise + lines + background + distortion
   - Realistic CAPTCHA appearance

2. **out/ (Noise+Lines Only)**
   - Clean text + noise + lines
   - White background (no distortion)
   - Used for intermediate training

3. **otv/ (Clean Target)**
   - Only text on white background
   - Ground truth for training

**Noise Types:**
- Gaussian noise
- Salt-and-pepper noise
- Scratches and lines
- Grid patterns
- Wave distortions
- Compression artifacts
- Blur effects
- Perspective distortion

**Metadata:**
- Noise coordinates and types
- Line coordinates and IDs
- Bounding boxes
- Noise level
- Random seed for reproducibility

### 4. Data Loading (data_loader.py)

**CaptchaDataset:**
- Loads inp/out/otv triplets
- Automatic augmentation pipeline
- Normalization to [-1, 1]
- Caching support

**Augmentation Pipeline:**
- Random brightness/contrast
- Random blur and rotation
- Random HSV shifts
- Random sharpening
- Random pixel noise
- Perspective transforms

**DataLoaderFactory:**
- Train loader: shuffled, augmented
- Validation loader: non-shuffled, no augmentation
- Test loader: batch size 1

### 5. Inference Engine (infer.py)

**Inferencer:**
- Single image inference
- Batch processing
- Directory processing
- Visualization support

**RealtimeInferencer:**
- Frame-by-frame processing
- Video processing support
- Optimized for speed

**BatchInferencer:**
- Optimized batch processing
- Memory-efficient
- Configurable batch size

### 6. Utilities (utils.py)

**Configuration Management:**
- Config class for hyperparameters
- Directory management
- Checkpoint handling

**Data Processing:**
- Augmentation pipeline
- Dataset caching
- Metadata management

**Training Utilities:**
- Learning rate scheduling
- EMA weight averaging
- Gradient clipping
- Seed management

### 7. GUI Application (cleaner_ai.py)

**Tabs:**

1. **Train Tab**
   - Configure training parameters
   - Start/stop training
   - Real-time loss monitoring
   - Progress tracking

2. **Generate Tab**
   - Dataset generation
   - Progress bar
   - Configurable parameters

3. **Test Tab**
   - Load and test images
   - Real-time inference
   - Before/after comparison

4. **Viewer Tab**
   - Browse dataset
   - View inp/out/otv triplets
   - Navigation controls

5. **Logs Tab**
   - View training logs
   - System monitoring

## Data Flow

### Training Flow
```
Raw Images → Augmentation → Normalization → Model → Loss Calculation
                                              ↓
                                         Backprop → Optimizer Update
                                              ↓
                                         Checkpoint Save
```

### Inference Flow
```
Input Image → Resize → Normalize → Model → Denormalize → Output Image
```

## Performance Characteristics

### Memory Usage
- Model: ~150MB
- Batch size 16: ~4GB VRAM
- Batch size 1: ~1GB VRAM

### Speed
- Inference (256x256): ~50ms per image (GPU)
- Inference (256x256): ~500ms per image (CPU)
- Training: ~2-3 hours per epoch (GPU, batch size 16)

### Accuracy Metrics
- PSNR: 25-35 dB (typical)
- SSIM: 0.8-0.95 (typical)
- MSE: 100-500 (typical)

## Extension Points

### Custom Loss Functions
Modify `LossCalculator` in `train.py`

### Custom Architectures
Extend `CleanerAIModel` in `model.py`

### Custom Augmentations
Add methods to `AugmentationPipeline` in `utils.py`

### Custom Noise Types
Extend `CaptchaGenerator` in `captcha_generator.py`

## Production Deployment

### Model Export
```python
from export_model import ModelExporter
exporter = ModelExporter()
exporter.export_all()  # TorchScript, ONNX, Quantized
```

### Benchmarking
```python
from benchmark import ModelBenchmark
benchmark = ModelBenchmark()
benchmark.run_full_benchmark()
```

### Validation
```python
from validate import ModelValidator
validator = ModelValidator()
metrics = validator.validate_and_report()
```

## Configuration

Edit `config.json` or `Config` class in `utils.py`:
- Training hyperparameters
- Model architecture
- Data paths
- Device selection
- Batch sizes
- Learning rates

## Dependencies

- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+
- PyQt5 5.15+
- TensorBoard 2.14+
- SciPy 1.11+
- Pillow 10.0+
