# Cleaner AI - Troubleshooting Guide

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError: No module named 'PyQt5.QtChart'

**Error:**
```
ModuleNotFoundError: No module named 'PyQt5.QtChart'
```

**Solution:**
```bash
# Run the fix installation script
fix_install.bat

# Or manually install PyQt5
pip install --upgrade PyQt5 PyQt5-sip
```

### Issue 2: RuntimeError in DataLoader - weights.ndim mismatch

**Error:**
```
RuntimeError: weights.ndim (2) must match len(axes) (3)
```

**Solution:**
This is fixed in the latest version. Make sure you have the updated `utils.py` file.

### Issue 3: ModuleNotFoundError: No module named 'torch'

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 4: ModuleNotFoundError: No module named 'cv2'

**Error:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```bash
pip install opencv-python
```

### Issue 5: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# In GUI or code, reduce batch size
config.batch_size = 8  # Instead of 16
config.image_size = 128  # Instead of 256
config.mixed_precision = True  # Enable FP16
```

### Issue 6: No CUDA available (CPU mode)

**Warning:**
```
CUDA not available (CPU mode)
```

**Solution:**
This is normal if you don't have a GPU. Training will be slower but still works.

To use GPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 7: GUI doesn't start

**Error:**
```
GUI window doesn't appear
```

**Solution:**
```bash
# Test imports first
python test_imports.py

# Test CLI functionality
python test_cli.py

# Try running with verbose output
python -u run.py
```

### Issue 8: Data loading is slow

**Solution:**
```python
# Increase number of workers
config.num_workers = 8  # Instead of 4

# Or disable workers if on Windows
config.num_workers = 0
```

### Issue 9: Training doesn't start

**Error:**
```
Training error: 'NoneType' object has no attribute 'copy'
```

**Solution:**
1. Make sure dataset is generated first
2. Check that data/inp/, data/out/, data/otv/ directories have images
3. Run: `python quick_start.py` (option 1) to generate dataset

### Issue 10: Model not improving

**Solution:**
```python
# Lower learning rate
config.learning_rate = 0.0001  # Instead of 0.001

# Train longer
config.epochs = 200  # Instead of 100

# Generate more diverse dataset
generator.generate_dataset(num_images=10000)
```

---

## Quick Fixes

### Fix 1: Complete Reinstall
```bash
# Remove old installation
pip uninstall torch torchvision torchaudio PyQt5 opencv-python -y

# Reinstall everything
fix_install.bat
```

### Fix 2: Test Everything
```bash
# Test imports
python test_imports.py

# Test CLI
python test_cli.py

# Test GUI
python run.py
```

### Fix 3: Clear Cache
```bash
# Remove cache directories
rmdir /s /q __pycache__
rmdir /s /q .pytest_cache
rmdir /s /q cache

# Remove old models
rmdir /s /q models
```

### Fix 4: Verify Setup
```bash
python check_setup.py
```

---

## Installation Troubleshooting

### Problem: pip not found
```bash
# Use python -m pip instead
python -m pip install -r requirements.txt
```

### Problem: Permission denied
```bash
# Run as administrator or use --user flag
pip install --user -r requirements.txt
```

### Problem: Long installation time
```bash
# Use pre-built wheels
pip install --only-binary :all: -r requirements.txt
```

---

## Performance Troubleshooting

### Slow Training
- Increase batch size (if memory allows)
- Enable mixed precision (FP16)
- Reduce image size
- Increase num_workers

### Slow Inference
- Use batch processing instead of single images
- Reduce image size
- Enable mixed precision
- Use GPU instead of CPU

### High Memory Usage
- Reduce batch size
- Reduce image size
- Enable mixed precision
- Disable augmentation

---

## GPU Troubleshooting

### Check CUDA
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Check GPU Memory
```bash
# Windows
nvidia-smi

# Linux
nvidia-smi
```

### Monitor Training
```bash
# In another terminal
tensorboard --logdir=logs
```

---

## Data Troubleshooting

### No images generated
1. Check data/ directory exists
2. Check write permissions
3. Run: `python quick_start.py` (option 1)

### Images not loading
1. Check image format (PNG, JPG)
2. Check image size (256x256)
3. Check file permissions

### Metadata missing
1. Check .json files in data/inp/
2. Regenerate dataset
3. Check disk space

---

## Model Troubleshooting

### Model not saving
1. Check models/ directory exists
2. Check write permissions
3. Check disk space

### Model not loading
1. Check checkpoint file exists
2. Check file is not corrupted
3. Check PyTorch version compatibility

### Model not improving
1. Check learning rate (too high/low)
2. Check dataset quality
3. Check training parameters
4. Train longer

---

## GUI Troubleshooting

### GUI freezes
1. Training is running (normal)
2. Check system resources
3. Reduce batch size
4. Disable augmentation

### Buttons not responding
1. Wait for current operation to complete
2. Check logs for errors
3. Restart GUI

### Images not displaying
1. Check image format
2. Check image size
3. Check file permissions

---

## Command Line Troubleshooting

### Script doesn't run
```bash
# Check Python version
python --version

# Check file exists
dir cleaner_ai.py

# Run with verbose output
python -u cleaner_ai.py
```

### Import errors
```bash
# Check installed packages
pip list

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

---

## Getting Help

### Step 1: Check Documentation
- START.md - Quick start
- USAGE.md - Usage guide
- ARCHITECTURE.md - Technical details

### Step 2: Run Diagnostics
```bash
python check_setup.py
python test_imports.py
python test_cli.py
```

### Step 3: Check Logs
- data/logs/ - Training logs
- GUI log window - Real-time output

### Step 4: Review Examples
- EXAMPLES.md - Code examples
- demo.py - Demonstrations

---

## Advanced Troubleshooting

### Enable Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Performance
```bash
python -m cProfile -s cumtime train.py
```

### Check Memory Usage
```python
import tracemalloc
tracemalloc.start()
# ... run code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f}MB")
print(f"Peak: {peak / 1024 / 1024:.1f}MB")
```

---

## Still Having Issues?

1. **Check all documentation files**
2. **Run diagnostic scripts** (check_setup.py, test_imports.py, test_cli.py)
3. **Review error messages carefully**
4. **Check system resources** (RAM, disk space, GPU)
5. **Try complete reinstall** (fix_install.bat)

---

**Last Updated**: May 2024
**Version**: 1.0.0
