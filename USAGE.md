# Cleaner AI - Usage Guide

## Quick Start

### 1. Installation

```bash
# Windows
install.bat

# Linux/Mac
chmod +x install.sh
./install.sh

# Or manual
pip install -r requirements.txt
python check_setup.py
```

### 2. Launch GUI

```bash
python run.py
```

### 3. Generate Dataset

In GUI → Generate tab:
- Set "Number of Images" (e.g., 1000)
- Set "Image Size" (default 256)
- Click "Generate Dataset"

### 4. Train Model

In GUI → Train tab:
- Configure parameters:
  - Epochs: 100
  - Batch Size: 16
  - Learning Rate: 0.001
  - Image Size: 256
- Click "Start Training"
- Monitor loss in real-time

### 5. Test Model

In GUI → Test tab:
- Click "Load Image"
- Select a noisy CAPTCHA image
- Click "Run Inference"
- View before/after comparison

## Command Line Usage

### Generate Dataset

```python
from captcha_generator import CaptchaGenerator

generator = CaptchaGenerator(image_size=256, output_dir='data')
generator.generate_dataset(num_images=10000)
```

### Train Model

```python
from train import Trainer, Config

config = Config()
config.epochs = 100
config.batch_size = 16
config.learning_rate = 0.001

trainer = Trainer(config)
trainer.train()
```

### Run Inference

```python
from infer import Inferencer
import cv2

inferencer = Inferencer()
result = inferencer.infer_image('path/to/image.png')
cv2.imwrite('output.png', result)
```

### Batch Processing

```python
from infer import BatchInferencer

batch_inferencer = BatchInferencer(batch_size=32)
image_paths = ['img1.png', 'img2.png', 'img3.png']
results = batch_inferencer.infer_batch_optimized(image_paths)
```

### Video Processing

```python
from infer import RealtimeInferencer

realtime = RealtimeInferencer()
realtime.process_video('input.mp4', 'output.mp4')
```

## Advanced Usage

### Custom Configuration

Edit `config.json` or modify `Config` class:

```python
from utils import Config

config = Config()
config.epochs = 200
config.batch_size = 32
config.learning_rate = 0.0005
config.image_size = 512
config.mixed_precision = True
```

### Model Export

```python
from export_model import ModelExporter

exporter = ModelExporter()
exporter.export_torchscript('models/model.pt')
exporter.export_onnx('models/model.onnx')
exporter.export_quantized('models/model_quantized.pt')
```

### Model Benchmarking

```python
from benchmark import ModelBenchmark

benchmark = ModelBenchmark()
benchmark.run_full_benchmark()
```

### Model Validation

```python
from validate import ModelValidator

validator = ModelValidator()
metrics = validator.validate_and_report(num_samples=100)
```

### Custom Training Loop

```python
from train import Trainer, Config
from data_loader import DataLoaderFactory

config = Config()
trainer = Trainer(config)

train_loader = DataLoaderFactory.create_train_loader(
    batch_size=16,
    image_size=256
)

for epoch in range(config.epochs):
    train_loss = trainer.train_epoch(train_loader, epoch)
    print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
```

## GUI Features

### Train Tab
- **Epochs**: Number of training iterations
- **Batch Size**: Images per batch (higher = faster but more memory)
- **Learning Rate**: Optimization step size
- **Image Size**: Input/output resolution
- **Mixed Precision**: FP16 training for speed
- **Progress Bar**: Training progress
- **Log Window**: Real-time training output

### Generate Tab
- **Number of Images**: Dataset size
- **Image Size**: CAPTCHA resolution
- **Progress Bar**: Generation progress
- **Log Window**: Generation details

### Test Tab
- **Load Image**: Select test image
- **Run Inference**: Process image
- **Input Viewer**: Original noisy image
- **Output Viewer**: Cleaned image
- **Log Window**: Processing details

### Viewer Tab
- **Load Dataset**: Select data directory
- **Previous/Next**: Navigate images
- **Three Viewers**: inp, out, otv images
- **Image Counter**: Current position

### Logs Tab
- **Refresh Logs**: Update log display
- **Clear Logs**: Clear log window
- **Log Window**: Training/system logs

## Tips & Tricks

### Faster Training
- Increase batch size (if GPU memory allows)
- Use mixed precision (FP16)
- Reduce image size to 128x128
- Use fewer workers if CPU bottleneck

### Better Results
- Generate larger dataset (10000+ images)
- Train for more epochs (200+)
- Use lower learning rate (0.0001)
- Enable data augmentation

### Memory Optimization
- Reduce batch size
- Reduce image size
- Enable mixed precision
- Use gradient accumulation

### GPU Optimization
- Check CUDA availability: `torch.cuda.is_available()`
- Monitor GPU: `nvidia-smi`
- Use TensorBoard: `tensorboard --logdir=logs`

## Troubleshooting

### CUDA Out of Memory
```python
config.batch_size = 8  # Reduce batch size
config.image_size = 128  # Reduce image size
config.mixed_precision = True  # Enable FP16
```

### Slow Training
```python
config.num_workers = 8  # Increase data loading workers
config.batch_size = 32  # Increase batch size
```

### Model Not Improving
```python
config.learning_rate = 0.0001  # Lower learning rate
config.epochs = 200  # Train longer
# Generate more diverse dataset
```

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Performance Metrics

### Typical Results
- PSNR: 25-35 dB
- SSIM: 0.8-0.95
- MSE: 100-500

### Inference Speed
- GPU (RTX 3080): ~50ms per image
- GPU (RTX 2080): ~100ms per image
- CPU (i7-10700K): ~500ms per image

### Training Time
- 100 epochs, batch size 16: ~2-3 hours (GPU)
- 100 epochs, batch size 8: ~4-5 hours (GPU)

## File Structure

```
cleaner_ai/
├── model.py                 # Neural network
├── train.py                 # Training pipeline
├── infer.py                 # Inference engines
├── captcha_generator.py     # Dataset generation
├── data_loader.py           # Data loading
├── utils.py                 # Utilities
├── cleaner_ai.py            # GUI application
├── run.py                   # Entry point
├── quick_start.py           # Quick start script
├── check_setup.py           # Setup verification
├── benchmark.py             # Performance benchmarking
├── validate.py              # Model validation
├── export_model.py          # Model export
├── demo.py                  # Demonstration
├── config.json              # Configuration
├── requirements.txt         # Dependencies
├── README.md                # Project overview
├── ARCHITECTURE.md          # Architecture details
├── USAGE.md                 # This file
└── data/
    ├── inp/                 # Noisy images
    ├── out/                 # Noise+lines
    ├── otv/                 # Clean targets
    ├── models/              # Saved models
    └── logs/                # Training logs
```

## Support

For issues or questions:
1. Check ARCHITECTURE.md for technical details
2. Run check_setup.py to verify installation
3. Check logs in data/logs/ directory
4. Review error messages in GUI log window

## License

Proprietary - Cleaner AI System
