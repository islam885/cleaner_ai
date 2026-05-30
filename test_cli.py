import sys
from pathlib import Path

print("=" * 60)
print("Cleaner AI - CLI Test")
print("=" * 60)

print("\n1. Testing imports...")
try:
    from model import CleanerAIModel
    from train import Trainer, Config
    from infer import Inferencer
    from captcha_generator import CaptchaGenerator
    from data_loader import DataLoaderFactory
    from utils import DirectoryManager, set_seed
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n2. Creating directories...")
try:
    dir_manager = DirectoryManager()
    print("✓ Directories created")
except Exception as e:
    print(f"✗ Directory creation failed: {e}")
    sys.exit(1)

print("\n3. Testing model creation...")
try:
    import torch
    model = CleanerAIModel()
    print(f"✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

print("\n4. Testing config...")
try:
    config = Config()
    print(f"✓ Config created")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

print("\n5. Testing data generation...")
try:
    generator = CaptchaGenerator(image_size=256)
    print("✓ Generator created")
    print("  Generating 1 sample image...")
    filename, metadata = generator.generate_captcha(0)
    print(f"✓ Generated: {filename}")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    sys.exit(1)

print("\n6. Testing data loading...")
try:
    train_loader = DataLoaderFactory.create_train_loader(
        batch_size=2,
        image_size=256,
        num_workers=0
    )
    print(f"✓ Data loader created")
    
    batch = next(iter(train_loader))
    print(f"  Batch shapes:")
    print(f"    inp: {batch['inp'].shape}")
    print(f"    out: {batch['out'].shape}")
    print(f"    otv: {batch['otv'].shape}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python run.py (for GUI)")
print("2. Or: python quick_start.py (for CLI)")
