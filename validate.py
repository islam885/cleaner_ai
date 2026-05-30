import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from infer import Inferencer
from utils import DirectoryManager, get_device


class ModelValidator:
    def __init__(self, model_path=None):
        self.device = get_device()
        self.inferencer = Inferencer(model_path, self.device)

    def calculate_psnr(self, img1, img2):
        return peak_signal_noise_ratio(img1, img2, data_range=255)

    def calculate_ssim(self, img1, img2):
        return structural_similarity(img1, img2, channel_axis=2, data_range=255)

    def calculate_mse(self, img1, img2):
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    def validate_dataset(self, data_dir='data', num_samples=None):
        print("Validating model on dataset...")
        
        inp_dir = Path(data_dir) / 'inp'
        otv_dir = Path(data_dir) / 'otv'
        
        inp_images = sorted(list(inp_dir.glob('*.png')))
        
        if num_samples:
            inp_images = inp_images[:num_samples]
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'mse': []
        }
        
        for inp_path in tqdm(inp_images, desc="Validating"):
            otv_path = otv_dir / inp_path.name
            
            if not otv_path.exists():
                continue
            
            output = self.inferencer.infer_image(str(inp_path))
            target = cv2.imread(str(otv_path))
            
            if output.shape != target.shape:
                output = cv2.resize(output, (target.shape[1], target.shape[0]))
            
            psnr = self.calculate_psnr(target, output)
            ssim = self.calculate_ssim(target, output)
            mse = self.calculate_mse(target, output)
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['mse'].append(mse)
        
        return metrics

    def print_metrics(self, metrics):
        print("\n" + "=" * 60)
        print("Validation Results")
        print("=" * 60)
        
        for metric_name, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                print(f"\n{metric_name.upper()}:")
                print(f"  Mean: {mean_val:.4f}")
                print(f"  Std:  {std_val:.4f}")
                print(f"  Min:  {min_val:.4f}")
                print(f"  Max:  {max_val:.4f}")

    def validate_and_report(self, data_dir='data', num_samples=None):
        print("=" * 60)
        print("Cleaner AI - Model Validation")
        print("=" * 60)
        
        metrics = self.validate_dataset(data_dir, num_samples)
        self.print_metrics(metrics)
        
        return metrics


if __name__ == '__main__':
    validator = ModelValidator()
    validator.validate_and_report(num_samples=100)
