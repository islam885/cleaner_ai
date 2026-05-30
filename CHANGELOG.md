# Changelog

All notable changes to Cleaner AI will be documented in this file.

## [1.0.0] - 2026-05-29

### Added

#### Core Features
- U-Net architecture with attention mechanisms and residual blocks
- Mixed precision training (FP16) for faster GPU training
- EMA (Exponential Moving Average) weights for model stability
- Cosine learning rate scheduler with warmup
- Gradient clipping for training stability
- Automatic checkpoint saving and loading

#### Dataset Generation
- Powerful CAPTCHA generator with 20+ augmentation types
- Gaussian, Salt-Pepper, Speckle, and Poisson noise
- Random lines (straight, curve, zigzag)
- Multiple distortion types (blur, rotation, perspective)
- Background generation (gradient, texture, pattern)
- Metadata tracking with unique noise IDs
- Three-stage generation (inp/out/otv)

#### GUI Application
- Modern dark theme interface using CustomTkinter
- Real-time training visualization
- Live preview of input/output/target images
- Training statistics (epoch, loss, PSNR, FPS, LR)
- Progress bars and status indicators
- Dataset generation tab with preview
- Test tab for single image cleaning
- Viewer tab with noise overlay visualization
- Drag & drop support (planned)

#### CLI Tools
- `train_cli.py` - Command-line training
- `infer_cli.py` - Batch image processing
- `generate_cli.py` - Dataset generation
- `validate.py` - Model validation
- `benchmark.py` - Performance testing
- `export_model.py` - Model export (ONNX, TorchScript)

#### Utilities
- Dataset caching system
- Configuration management
- TensorBoard integration
- PSNR calculation
- Image conversion utilities
- Device detection (CUDA/CPU)

#### Documentation
- README.md - Project overview
- ARCHITECTURE.md - Technical documentation
- USAGE.md - User guide
- CONTRIBUTING.md - Development guide
- CHANGELOG.md - Version history

#### Setup & Installation
- `requirements.txt` - Python dependencies
- `quick_start.py` - Automated setup
- `setup.bat` - Windows installer
- `run.bat` - Windows launcher
- `.gitignore` - Git configuration

### Technical Details

#### Model
- Input: RGB images (3 channels)
- Output: RGB images (3 channels)
- Architecture: U-Net with 4 encoder/decoder levels
- Parameters: ~31M
- Attention: Self-attention at multiple scales
- Residual: Skip connections throughout

#### Training
- Loss: L1 (Mean Absolute Error)
- Optimizer: AdamW
- Learning Rate: 0.001 (default)
- Batch Size: 8 (default)
- Image Size: 256x256 (default)
- Epochs: 100 (default)

#### Performance
- GPU Training: ~10-15 FPS (RTX 3080)
- GPU Inference: ~50-100 FPS (RTX 3080)
- CPU Inference: ~2-5 FPS
- Memory: ~4GB VRAM (batch=8, size=256)

### Known Issues
- None

### Future Plans
- Transformer-based architecture
- Perceptual loss function
- Adversarial training
- Real CAPTCHA dataset support
- Mobile deployment (TFLite)
- Web interface
- REST API
- Docker container

## [0.1.0] - Development

### Initial Development
- Project structure
- Basic model architecture
- Simple training loop
- Dataset generator prototype
