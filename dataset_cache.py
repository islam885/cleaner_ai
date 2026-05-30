import torch
import pickle
import hashlib
from pathlib import Path
import numpy as np


class DatasetCache:
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.max_memory_items = 1000
    
    def _get_cache_key(self, file_path, transform_params=None):
        key_str = str(file_path)
        if transform_params:
            key_str += str(transform_params)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, file_path, transform_params=None):
        cache_key = self._get_cache_key(file_path, transform_params)
        
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                
                if len(self.memory_cache) < self.max_memory_items:
                    self.memory_cache[cache_key] = data
                
                return data
        
        return None
    
    def set(self, file_path, data, transform_params=None):
        cache_key = self._get_cache_key(file_path, transform_params)
        
        if len(self.memory_cache) < self.max_memory_items:
            self.memory_cache[cache_key] = data
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def clear_memory(self):
        self.memory_cache.clear()
    
    def clear_disk(self):
        for cache_file in self.cache_dir.glob('*.pkl'):
            cache_file.unlink()
    
    def clear_all(self):
        self.clear_memory()
        self.clear_disk()
    
    def get_size(self):
        memory_size = len(self.memory_cache)
        disk_size = len(list(self.cache_dir.glob('*.pkl')))
        return {'memory': memory_size, 'disk': disk_size}


class PreloadedDataset:
    def __init__(self, dataset, device='cpu', preload_all=False):
        self.dataset = dataset
        self.device = device
        self.preloaded_data = {}
        
        if preload_all:
            self.preload_all()
    
    def preload_all(self):
        print("Preloading entire dataset...")
        for idx in range(len(self.dataset)):
            data = self.dataset[idx]
            if isinstance(data, tuple):
                data = tuple(d.to(self.device) if isinstance(d, torch.Tensor) else d for d in data)
            elif isinstance(data, torch.Tensor):
                data = data.to(self.device)
            self.preloaded_data[idx] = data
        print(f"Preloaded {len(self.preloaded_data)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx in self.preloaded_data:
            return self.preloaded_data[idx]
        
        data = self.dataset[idx]
        if isinstance(data, tuple):
            data = tuple(d.to(self.device) if isinstance(d, torch.Tensor) else d for d in data)
        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)
        
        return data
