import sys
import importlib
from pathlib import Path


def check_python_version():
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True


def check_dependencies():
    dependencies = [
        'torch',
        'torchvision',
        'numpy',
        'cv2',
        'PIL',
        'PyQt5',
        'tensorboard',
        'scipy',
        'tqdm'
    ]
    
    print("\nChecking dependencies:")
    all_ok = True
    
    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {dep}: {version}")
        except ImportError:
            print(f"✗ {dep}: NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_cuda():
    print("\nChecking CUDA:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
        else:
            print("⚠ CUDA not available (CPU mode)")
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")


def check_directories():
    print("\nChecking directories:")
    dirs = ['data', 'models', 'logs']
    
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"✓ {d}/ exists")
        else:
            print(f"⚠ {d}/ does not exist (will be created)")


def check_files():
    print("\nChecking project files:")
    files = [
        'model.py',
        'train.py',
        'infer.py',
        'captcha_generator.py',
        'data_loader.py',
        'utils.py',
        'cleaner_ai.py',
        'requirements.txt'
    ]
    
    all_ok = True
    for f in files:
        path = Path(f)
        if path.exists():
            print(f"✓ {f}")
        else:
            print(f"✗ {f} NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    print("=" * 60)
    print("Cleaner AI - Setup Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA", check_cuda),
        ("Directories", check_directories),
        ("Project Files", check_files)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result if isinstance(result, bool) else True))
        except Exception as e:
            print(f"✗ {name} check failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ OK" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_ok = all(result for _, result in results)
    
    if all_ok:
        print("\n✓ All checks passed! Ready to use Cleaner AI")
        print("\nNext steps:")
        print("1. Run: python quick_start.py")
        print("2. Or run: python run.py (for GUI)")
    else:
        print("\n✗ Some checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")


if __name__ == '__main__':
    main()
