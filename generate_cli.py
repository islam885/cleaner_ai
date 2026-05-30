import argparse
from captcha_generator import CaptchaGenerator
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Generate CAPTCHA dataset')
    
    parser.add_argument('--count', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--start', type=int, default=0, help='Starting index')
    parser.add_argument('--inp-dir', type=str, default='data/inp', help='Input directory')
    parser.add_argument('--out-dir', type=str, default='data/out', help='Output directory')
    parser.add_argument('--otv-dir', type=str, default='data/otv', help='Target directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dirs = {
        'inp': args.inp_dir,
        'out': args.out_dir,
        'otv': args.otv_dir
    }
    
    generator = CaptchaGenerator(output_dirs)
    
    print(f"Generating {args.count} CAPTCHA samples...")
    print(f"Starting from index: {args.start}")
    
    for i in tqdm(range(args.count)):
        generator.generate_captcha_set(args.start + i)
    
    print(f"\nGenerated {args.count} samples successfully!")
    print(f"Input (dirty): {args.inp_dir}")
    print(f"Output (no bg): {args.out_dir}")
    print(f"Target (clean): {args.otv_dir}")


if __name__ == '__main__':
    main()
