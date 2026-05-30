import argparse
from train import Trainer
from utils import get_device
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cleaner AI Model')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=256, help='Image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--cache', action='store_true', help='Use dataset cache')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = get_device()
    print(f"Using device: {device.type.upper()}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'image_size': args.image_size,
        'inp_dir': 'data/inp',
        'otv_dir': 'data/otv',
        'model_dir': 'data/models',
        'log_dir': 'data/logs',
        'checkpoint_path': args.resume or 'data/models/best_model.pth',
        'num_workers': args.workers,
        'use_cache': args.cache
    }
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    
    trainer = Trainer(config)
    trainer.train()
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
