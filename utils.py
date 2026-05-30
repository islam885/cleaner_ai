import os
import json
import random
import string
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime


class Config:
    def __init__(self):
        self.epochs = 100
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.image_size = 256
        self.dataset_size = 10000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = False
        self.num_workers = 0
        self.checkpoint_dir = 'models'
        self.data_dir = 'data'
        self.log_dir = 'logs'
        self.seed = 42

    def to_dict(self):
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'image_size': self.image_size,
            'dataset_size': self.dataset_size,
            'device': str(self.device),
            'mixed_precision': self.mixed_precision,
            'num_workers': self.num_workers,
            'seed': self.seed
        }

    def from_dict(self, d):
        for key, value in d.items():
            if hasattr(self, key) and key != 'device':
                setattr(self, key, value)


class DirectoryManager:
    def __init__(self, base_path='data'):
        self.base_path = Path(base_path)
        self.inp_dir = self.base_path / 'inp'
        self.out_dir = self.base_path / 'out'
        self.otv_dir = self.base_path / 'otv'
        self.models_dir = Path('models')
        self.logs_dir = Path('logs')
        self.images_dir = self.base_path / 'images'
        self.cleaner_dir = self.base_path / 'cleaner'
        
        self.create_all()

    def create_all(self):
        for directory in [self.inp_dir, self.out_dir, self.otv_dir, 
                         self.models_dir, self.logs_dir, self.images_dir, self.cleaner_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_latest_checkpoint(self):
        if not self.models_dir.exists():
            return None
        checkpoints = list(self.models_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            return None
        return max(checkpoints, key=os.path.getctime)

    def save_checkpoint(self, model, optimizer, epoch, loss, discriminator=None):
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        if discriminator is not None:
            checkpoint['discriminator_state'] = discriminator.state_dict()
        
        path = self.models_dir / f'checkpoint_{epoch:05d}.pt'
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, model, optimizer, path=None):
        if path is None:
            path = self.get_latest_checkpoint()
        
        if path is None:
            return 0, float('inf')
        
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        return checkpoint['epoch'], checkpoint['loss']


class NoiseIDGenerator:
    @staticmethod
    def generate_id():
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    @staticmethod
    def encode_noise_info(noise_type, thickness, strength, layer, transparency):
        info = {
            'type': noise_type,
            'thickness': thickness,
            'strength': strength,
            'layer': layer,
            'transparency': transparency
        }
        return info

    @staticmethod
    def decode_noise_info(noise_id, metadata):
        if 'noises' in metadata and noise_id in metadata['noises']:
            return metadata['noises'][noise_id]
        return None


class MetadataManager:
    @staticmethod
    def create_metadata(image_name, noise_coords, line_coords, noise_types, 
                       line_ids, noise_ids, bboxes, noise_level, seed):
        metadata = {
            'image_name': image_name,
            'timestamp': datetime.now().isoformat(),
            'noise_coordinates': noise_coords,
            'line_coordinates': line_coords,
            'noise_types': noise_types,
            'line_ids': line_ids,
            'noise_ids': noise_ids,
            'bboxes': bboxes,
            'noise_level': noise_level,
            'random_seed': seed
        }
        return metadata

    @staticmethod
    def save_metadata(metadata, path):
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_metadata(path):
        with open(path, 'r') as f:
            return json.load(f)


class AugmentationPipeline:
    def __init__(self, image_size=256):
        self.image_size = image_size

    def random_brightness(self, image, factor=0.3):
        brightness = np.random.uniform(1 - factor, 1 + factor)
        return np.clip(image * brightness, 0, 255).astype(np.uint8)

    def random_contrast(self, image, factor=0.3):
        contrast = np.random.uniform(1 - factor, 1 + factor)
        mean = image.mean()
        return np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)

    def random_blur(self, image, kernel_size_range=(3, 7)):
        import cv2
        kernel_size = np.random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def random_rotation(self, image, angle_range=15):
        import cv2
        h, w = image.shape[:2]
        angle = np.random.uniform(-angle_range, angle_range)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h)).astype(np.uint8)

    def random_perspective(self, image):
        import cv2
        h, w = image.shape[:2]
        points1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = np.random.randint(-20, 20, (4, 2))
        points2 = points1 + offset
        
        matrix = cv2.getPerspectiveTransform(points1, points2)
        return cv2.warpPerspective(image, matrix, (w, h)).astype(np.uint8)

    def random_hsv(self, image):
        import cv2
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-30, 30)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def random_sharpen(self, image):
        from scipy.ndimage import convolve
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[2]):
                result[:, :, i] = convolve(image[:, :, i].astype(np.float32), kernel)
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            return np.clip(convolve(image.astype(np.float32), kernel), 0, 255).astype(np.uint8)

    def random_jpeg_noise(self, image, quality_range=(50, 95)):
        import cv2
        quality = np.random.randint(*quality_range)
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def random_pixel_noise(self, image, intensity=0.1):
        noise = np.random.normal(0, intensity * 255, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def apply_augmentation(self, image, augmentation_type='all'):
        augmentations = [
            self.random_brightness,
            self.random_contrast,
            self.random_blur,
            self.random_rotation,
            self.random_hsv,
            self.random_sharpen,
            self.random_pixel_noise
        ]
        
        if augmentation_type == 'all':
            num_augmentations = np.random.randint(1, 4)
            selected = np.random.choice(augmentations, num_augmentations, replace=False)
            for aug in selected:
                image = aug(image)
        else:
            image = augmentations[augmentation_type](image)
        
        return image


class DatasetCache:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_path(self, key):
        return self.cache_dir / f"{key}.pt"

    def save(self, key, data):
        torch.save(data, self.get_cache_path(key))

    def load(self, key):
        path = self.get_cache_path(key)
        if path.exists():
            return torch.load(path)
        return None

    def exists(self, key):
        return self.get_cache_path(key).exists()

    def clear(self):
        for file in self.cache_dir.glob('*.pt'):
            file.unlink()


class LRScheduler:
    def __init__(self, optimizer, base_lr, total_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch):
        return self.base_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))

    def step(self, epoch):
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EMAWeights:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
