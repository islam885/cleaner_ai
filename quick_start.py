import os
import sys
from pathlib import Path

from captcha_generator import CaptchaGenerator
from utils import DirectoryManager, Config, set_seed
from train import Trainer
from infer import Inferencer


def setup_project():
    print("Setting up Cleaner AI project...")
    dir_manager = DirectoryManager()
    print("✓ Directories created")


def generate_sample_dataset(num_images=100):
    print(f"\nGenerating sample dataset ({num_images} images)...")
    generator = CaptchaGenerator(image_size=256, output_dir='data')
    generator.generate_dataset(num_images=num_images)
    print("✓ Dataset generated")


def train_model(epochs=10):
    print(f"\nTraining model for {epochs} epochs...")
    config = Config()
    config.epochs = epochs
    config.batch_size = 8
    config.learning_rate = 0.001
    config.image_size = 256
    
    trainer = Trainer(config)
    trainer.train()
    print("✓ Training complete")


def test_inference():
    print("\nTesting inference...")
    test_image = Path('data/inp/captcha_00000.png')
    
    if not test_image.exists():
        print("✗ No test image found. Generate dataset first.")
        return
    
    try:
        inferencer = Inferencer()
        result = inferencer.infer_image(str(test_image))
        output_path = Path('output_test.png')
        import cv2
        cv2.imwrite(str(output_path), result)
        print(f"✓ Inference complete. Output saved to {output_path}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")


def main():
    print("=" * 60)
    print("Cleaner AI - Quick Start")
    print("=" * 60)
    
    setup_project()
    
    print("\nOptions:")
    print("1. Generate sample dataset")
    print("2. Train model")
    print("3. Test inference")
    print("4. Full pipeline (generate + train + test)")
    print("5. Launch GUI")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        num = input("Number of images (default 100): ").strip()
        generate_sample_dataset(int(num) if num else 100)
    
    elif choice == '2':
        epochs = input("Number of epochs (default 10): ").strip()
        train_model(int(epochs) if epochs else 10)
    
    elif choice == '3':
        test_inference()
    
    elif choice == '4':
        generate_sample_dataset(100)
        train_model(5)
        test_inference()
    
    elif choice == '5':
        from cleaner_ai import main as gui_main
        gui_main()
    
    else:
        print("Invalid option")


if __name__ == '__main__':
    main()
