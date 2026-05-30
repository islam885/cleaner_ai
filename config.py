import json
from pathlib import Path


class Config:
    DEFAULT_CONFIG = {
        'model': {
            'in_channels': 3,
            'out_channels': 3,
            'base_channels': 64,
            'use_attention': True,
            'use_residual': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 8,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
            'min_lr': 1e-6,
            'gradient_clip': 1.0,
            'use_amp': True,
            'ema_decay': 0.999
        },
        'data': {
            'image_size': 256,
            'num_workers': 4,
            'use_cache': False,
            'train_split': 0.9,
            'augmentation': True
        },
        'paths': {
            'inp_dir': 'data/inp',
            'out_dir': 'data/out',
            'otv_dir': 'data/otv',
            'model_dir': 'data/models',
            'log_dir': 'data/logs',
            'cache_dir': 'data/cache'
        },
        'generator': {
            'min_width': 150,
            'max_width': 300,
            'min_height': 50,
            'max_height': 100,
            'min_text_length': 4,
            'max_text_length': 8,
            'min_font_size': 30,
            'max_font_size': 50,
            'num_noises': [3, 8],
            'num_lines': [5, 15]
        },
        'validation': {
            'interval': 5,
            'save_samples': True,
            'num_samples': 10
        }
    }
    
    def __init__(self, config_path='config.json'):
        self.config_path = Path(config_path)
        self.config = self.DEFAULT_CONFIG.copy()
        
        if self.config_path.exists():
            self.load()
        else:
            self.save()
    
    def load(self):
        with open(self.config_path, 'r') as f:
            loaded_config = json.load(f)
            self._update_nested_dict(self.config, loaded_config)
    
    def save(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _update_nested_dict(self, base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, *keys):
        result = self.config
        for key in keys:
            result = result[key]
        return result
    
    def set(self, value, *keys):
        result = self.config
        for key in keys[:-1]:
            result = result[key]
        result[keys[-1]] = value
        self.save()
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
        self.save()


if __name__ == '__main__':
    config = Config()
    print("Configuration loaded:")
    print(json.dumps(config.config, indent=2))
