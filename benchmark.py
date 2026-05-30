import torch
import time
import numpy as np
from pathlib import Path

from model import CleanerAIModel
from utils import get_device, count_parameters


class ModelBenchmark:
    def __init__(self, model_path=None):
        self.device = get_device()
        self.model = CleanerAIModel().to(self.device)
        self.model.eval()
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)

    def benchmark_inference(self, image_size=256, batch_size=1, num_iterations=100):
        print(f"\nBenchmarking inference (batch_size={batch_size}, image_size={image_size}):")
        
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = batch_size / avg_time
        
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per batch: {avg_time*1000:.2f}ms")
        print(f"  FPS: {fps:.2f}")
        
        return fps, avg_time

    def benchmark_memory(self, image_size=256, batch_size=1):
        print(f"\nBenchmarking memory (batch_size={batch_size}, image_size={image_size}):")
        
        if not torch.cuda.is_available():
            print("  CUDA not available, skipping memory benchmark")
            return
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  Peak memory: {peak_memory:.2f}MB")
        
        return peak_memory

    def benchmark_throughput(self, image_size=256, batch_sizes=[1, 4, 8, 16, 32]):
        print(f"\nBenchmarking throughput (image_size={image_size}):")
        print(f"{'Batch Size':<15} {'FPS':<15} {'Time/Batch (ms)':<20}")
        print("-" * 50)
        
        results = {}
        for batch_size in batch_sizes:
            try:
                fps, avg_time = self.benchmark_inference(image_size, batch_size, num_iterations=50)
                results[batch_size] = (fps, avg_time)
                print(f"{batch_size:<15} {fps:<15.2f} {avg_time*1000:<20.2f}")
            except RuntimeError as e:
                print(f"{batch_size:<15} {'OOM':<15} {'N/A':<20}")
        
        return results

    def print_model_info(self):
        print("\nModel Information:")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {count_parameters(self.model):,}")
        print(f"  Model size: {sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2):.2f}MB")

    def run_full_benchmark(self):
        print("=" * 60)
        print("Cleaner AI - Model Benchmark")
        print("=" * 60)
        
        self.print_model_info()
        self.benchmark_memory(256, 1)
        self.benchmark_throughput(256, [1, 2, 4, 8, 16])
        
        print("\n" + "=" * 60)
        print("Benchmark complete!")


if __name__ == '__main__':
    benchmark = ModelBenchmark()
    benchmark.run_full_benchmark()
