# Cleaner AI - Getting Started

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
# Windows
install.bat

# Linux/Mac
chmod +x install.sh
./install.sh

# Or manual
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python check_setup.py
```

### Step 3: Launch GUI
```bash
python run.py
```

### Step 4: Generate Dataset
- Go to "Generate" tab
- Set "Number of Images" to 100
- Click "Generate Dataset"
- Wait for completion

### Step 5: Train Model
- Go to "Train" tab
- Set "Epochs" to 10 (for quick test)
- Click "Start Training"
- Monitor loss in real-time

### Step 6: Test Model
- Go to "Test" tab
- Click "Load Image"
- Select an image from `data/inp/`
- Click "Run Inference"
- View results

## 📋 System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended
- Python 3.10+
- 16GB RAM
- GPU with CUDA support
- 50GB disk space

## 🔧 Installation Options

### Option 1: Automated (Recommended)
```bash
# Windows
install.bat

# Linux/Mac
./install.sh
```

### Option 2: Manual Installation
```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_setup.py
```

### Option 3: Docker
```bash
docker build -t cleaner-ai .
docker run -it --gpus all cleaner-ai python run.py
```

## 🎯 Common Tasks

### Generate Dataset
```bash
python quick_start.py
# Select option 1
```

### Train Model
```bash
python quick_start.py
# Select option 2
```

### Test Inference
```bash
python quick_start.py
# Select option 3
```

### Run Full Pipeline
```bash
python quick_start.py
# Select option 4
```

### Launch GUI
```bash
python run.py
```

### Run Benchmarks
```bash
python benchmark.py
```

### Validate Model
```bash
python validate.py
```

### Export Model
```bash
python export_model.py
```

## 📊 GUI Tabs

### Train Tab
- Configure training parameters
- Start/stop training
- Monitor loss in real-time
- View training logs

### Generate Tab
- Generate synthetic CAPTCHA dataset
- Configure dataset size
- Track generation progress

### Test Tab
- Load and test images
- Run inference
- View before/after comparison

### Viewer Tab
- Browse generated dataset
- View inp/out/otv triplets
- Navigate through images

### Logs Tab
- View training logs
- Monitor system performance

## 🐛 Troubleshooting

### Issue: CUDA not found
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory
```python
# In config or GUI:
config.batch_size = 8      # Reduce batch size
config.image_size = 128    # Reduce image size
config.mixed_precision = True  # Enable FP16
```

### Issue: Slow training
```python
# Increase workers and batch size
config.num_workers = 8
config.batch_size = 32
```

### Issue: Model not improving
```python
# Lower learning rate and train longer
config.learning_rate = 0.0001
config.epochs = 200
```

## 📚 Documentation

- **README.md** - Project overview
- **USAGE.md** - Detailed usage guide
- **ARCHITECTURE.md** - Technical architecture
- **EXAMPLES.md** - Code examples
- **PROJECT_INFO.md** - Project information

## 🎓 Learning Path

### Beginner
1. Read README.md
2. Run install.bat/install.sh
3. Launch GUI (python run.py)
4. Generate dataset
5. Train for 10 epochs
6. Test inference

### Intermediate
1. Read USAGE.md
2. Run command-line examples
3. Modify training parameters
4. Generate custom dataset
5. Train for 100 epochs
6. Validate model

### Advanced
1. Read ARCHITECTURE.md
2. Study model.py and train.py
3. Implement custom loss functions
4. Create custom architectures
5. Run benchmarks
6. Export models

## 🚀 Performance Tips

### Faster Training
- Increase batch size (if GPU memory allows)
- Enable mixed precision (FP16)
- Reduce image size to 128x128
- Use more workers

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

## 📈 Expected Results

### After 10 epochs
- PSNR: 20-25 dB
- SSIM: 0.7-0.8
- Training time: 10-15 minutes

### After 100 epochs
- PSNR: 25-30 dB
- SSIM: 0.8-0.9
- Training time: 2-3 hours

### After 200 epochs
- PSNR: 28-35 dB
- SSIM: 0.85-0.95
- Training time: 4-6 hours

## 🔗 Useful Commands

```bash
# Check setup
python check_setup.py

# Generate dataset
python quick_start.py

# Launch GUI
python run.py

# Run demo
python demo.py

# Benchmark model
python benchmark.py

# Validate model
python validate.py

# Export model
python export_model.py

# View TensorBoard logs
tensorboard --logdir=logs
```

## 📞 Support

### Getting Help
1. Check documentation files
2. Review EXAMPLES.md
3. Run check_setup.py
4. Check logs in data/logs/

### Common Issues
- See Troubleshooting section above
- Check GPU availability
- Verify Python version (3.8+)
- Ensure all dependencies installed

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (pip install -r requirements.txt)
- [ ] Setup verified (python check_setup.py)
- [ ] CUDA available (optional but recommended)
- [ ] Disk space available (10GB+)
- [ ] RAM available (8GB+)

## 🎉 Next Steps

1. **Generate Dataset**: Create synthetic CAPTCHA images
2. **Train Model**: Train the neural network
3. **Test Inference**: Clean real CAPTCHA images
4. **Validate Results**: Check model performance
5. **Export Model**: Save for deployment

## 📖 Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- OpenCV Documentation: https://docs.opencv.org/
- PyQt5 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/

---

**Ready to start?** Run `python run.py` to launch the GUI!

For detailed information, see USAGE.md or ARCHITECTURE.md
