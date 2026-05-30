import argparse
from pathlib import Path
from infer import CaptchaCleaner
import cv2
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Clean CAPTCHA images using Cleaner AI')
    
    parser.add_argument('input', type=str, help='Input image or directory')
    parser.add_argument('--output', type=str, default='data/cleaner/images', help='Output directory')
    parser.add_argument('--model', type=str, default='data/models/best_model.pth', help='Model path')
    parser.add_argument('--size', type=int, default=256, help='Processing size')
    parser.add_argument('--batch', action='store_true', help='Process directory')
    parser.add_argument('--benchmark', action='store_true', help='Show timing info')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Loading model...")
    cleaner = CaptchaCleaner(args.model)
    print("Model loaded!")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.batch or input_path.is_dir():
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        print(f"\nProcessing {len(image_files)} images...")
        
        total_time = 0
        
        for img_file in image_files:
            start_time = time.time()
            
            cleaned = cleaner.clean_image(str(img_file), output_size=args.size)
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            output_file = output_path / f"cleaned_{img_file.name}"
            cv2.imwrite(str(output_file), cleaned)
            
            if args.benchmark:
                print(f"  {img_file.name}: {elapsed*1000:.2f}ms")
        
        print(f"\nProcessed {len(image_files)} images in {total_time:.2f}s")
        print(f"Average: {total_time/len(image_files)*1000:.2f}ms per image")
        print(f"FPS: {len(image_files)/total_time:.2f}")
    
    else:
        print(f"\nProcessing {input_path.name}...")
        
        start_time = time.time()
        cleaned = cleaner.clean_image(str(input_path), output_size=args.size)
        elapsed = time.time() - start_time
        
        output_file = output_path / f"cleaned_{input_path.name}"
        cv2.imwrite(str(output_file), cleaned)
        
        print(f"Saved to: {output_file}")
        
        if args.benchmark:
            print(f"Time: {elapsed*1000:.2f}ms")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
