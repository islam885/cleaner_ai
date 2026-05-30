import cv2
import numpy as np
from pathlib import Path

from captcha_generator import CaptchaGenerator
from infer import Inferencer, RealtimeInferencer
from utils import DirectoryManager


def demo_generation():
    print("=" * 60)
    print("Demo: CAPTCHA Generation")
    print("=" * 60)
    
    generator = CaptchaGenerator(image_size=256, output_dir='data')
    
    print("\nGenerating 5 sample CAPTCHAs...")
    for i in range(5):
        filename, metadata = generator.generate_captcha(i)
        print(f"  Generated: {filename}")
        print(f"    Text: {metadata['text']}")
        print(f"    Noise level: {metadata['noise_level']:.2f}")
    
    print("\n✓ Generation demo complete!")


def demo_inference():
    print("\n" + "=" * 60)
    print("Demo: Image Inference")
    print("=" * 60)
    
    inp_dir = Path('data/inp')
    if not list(inp_dir.glob('*.png')):
        print("No images found. Run generation demo first.")
        return
    
    try:
        inferencer = Inferencer()
        
        test_images = list(inp_dir.glob('*.png'))[:3]
        
        for img_path in test_images:
            print(f"\nProcessing: {img_path.name}")
            result = inferencer.infer_image(str(img_path))
            output_path = Path('demo_output') / img_path.name
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), result)
            print(f"  Saved to: {output_path}")
        
        print("\n✓ Inference demo complete!")
    
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        print("  Make sure to train a model first!")


def demo_batch_inference():
    print("\n" + "=" * 60)
    print("Demo: Batch Inference")
    print("=" * 60)
    
    from infer import BatchInferencer
    
    inp_dir = Path('data/inp')
    if not list(inp_dir.glob('*.png')):
        print("No images found. Run generation demo first.")
        return
    
    try:
        batch_inferencer = BatchInferencer(batch_size=4)
        
        test_images = [str(p) for p in list(inp_dir.glob('*.png'))[:10]]
        
        print(f"Processing {len(test_images)} images in batches...")
        results = batch_inferencer.infer_batch_optimized(test_images)
        
        output_dir = Path('demo_batch_output')
        output_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            output_path = output_dir / f'batch_result_{i:03d}.png'
            cv2.imwrite(str(output_path), result)
        
        print(f"✓ Processed {len(results)} images")
        print(f"  Saved to: {output_dir}")
    
    except Exception as e:
        print(f"✗ Batch inference failed: {e}")


def demo_comparison():
    print("\n" + "=" * 60)
    print("Demo: Before/After Comparison")
    print("=" * 60)
    
    inp_dir = Path('data/inp')
    otv_dir = Path('data/otv')
    
    if not list(inp_dir.glob('*.png')):
        print("No images found. Run generation demo first.")
        return
    
    try:
        inferencer = Inferencer()
        
        test_image = list(inp_dir.glob('*.png'))[0]
        otv_image = otv_dir / test_image.name
        
        print(f"\nComparing: {test_image.name}")
        
        original, result, comparison = inferencer.infer_with_visualization(str(test_image))
        
        output_path = Path('demo_comparison.png')
        cv2.imwrite(str(output_path), comparison)
        
        print(f"  Original shape: {original.shape}")
        print(f"  Result shape: {result.shape}")
        print(f"  Comparison saved to: {output_path}")
        
        print("\n✓ Comparison demo complete!")
    
    except Exception as e:
        print(f"✗ Comparison failed: {e}")


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Cleaner AI - Demonstration".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    DirectoryManager().create_all()
    
    print("\nAvailable demos:")
    print("1. CAPTCHA Generation")
    print("2. Image Inference")
    print("3. Batch Inference")
    print("4. Before/After Comparison")
    print("5. Run all demos")
    
    choice = input("\nSelect demo (1-5): ").strip()
    
    if choice == '1':
        demo_generation()
    elif choice == '2':
        demo_inference()
    elif choice == '3':
        demo_batch_inference()
    elif choice == '4':
        demo_comparison()
    elif choice == '5':
        demo_generation()
        demo_inference()
        demo_batch_inference()
        demo_comparison()
    else:
        print("Invalid option")


if __name__ == '__main__':
    main()
