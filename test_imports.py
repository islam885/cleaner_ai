import sys

print("Testing imports...")

try:
    import torch
    print("✓ torch")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import cv2
    print("✓ cv2")
except ImportError as e:
    print(f"✗ cv2: {e}")

try:
    import numpy
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    from PIL import Image
    print("✓ PIL")
except ImportError as e:
    print(f"✗ PIL: {e}")

try:
    from PyQt5.QtWidgets import QApplication
    print("✓ PyQt5.QtWidgets")
except ImportError as e:
    print(f"✗ PyQt5.QtWidgets: {e}")

try:
    from PyQt5.QtCore import Qt
    print("✓ PyQt5.QtCore")
except ImportError as e:
    print(f"✗ PyQt5.QtCore: {e}")

try:
    from PyQt5.QtGui import QPixmap
    print("✓ PyQt5.QtGui")
except ImportError as e:
    print(f"✗ PyQt5.QtGui: {e}")

try:
    from tensorboard import program
    print("✓ tensorboard")
except ImportError as e:
    print(f"✗ tensorboard: {e}")

try:
    import scipy
    print("✓ scipy")
except ImportError as e:
    print(f"✗ scipy: {e}")

try:
    import tqdm
    print("✓ tqdm")
except ImportError as e:
    print(f"✗ tqdm: {e}")

print("\nAll critical imports tested!")
