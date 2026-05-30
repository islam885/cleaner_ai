import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from utils import AugmentationPipeline


class CaptchaDataset(Dataset):
    def __init__(self, data_dir='data', image_size=256, augment=True, mode='train'):
        self.data_dir = Path(data_dir)
        self.inp_dir = self.data_dir / 'inp'
        self.out_dir = self.data_dir / 'out'
        self.otv_dir = self.data_dir / 'otv'
        self.image_size = image_size
        self.augment = augment
        self.mode = mode
        
        self.augmentation = AugmentationPipeline(image_size)
        
        self.inp_images = sorted(list(self.inp_dir.glob('*.png')))
        self.out_images = sorted(list(self.out_dir.glob('*.png')))
        self.otv_images = sorted(list(self.otv_dir.glob('*.png')))
        
        self.length = len(self.inp_images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inp_path = self.inp_images[idx]
        out_path = self.out_dir / inp_path.name
        otv_path = self.otv_dir / inp_path.name
        
        inp_image = cv2.imread(str(inp_path))
        out_image = cv2.imread(str(out_path))
        otv_image = cv2.imread(str(otv_path))
        
        if inp_image is None or out_image is None or otv_image is None:
            return self.__getitem__((idx + 1) % self.length)
        
        inp_image = cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
        otv_image = cv2.cvtColor(otv_image, cv2.COLOR_BGR2RGB)
        
        inp_image = cv2.resize(inp_image, (self.image_size, self.image_size))
        out_image = cv2.resize(out_image, (self.image_size, self.image_size))
        otv_image = cv2.resize(otv_image, (self.image_size, self.image_size))
        
        if self.augment and self.mode == 'train':
            if np.random.random() < 0.5:
                inp_image = self.augmentation.apply_augmentation(inp_image)
            if np.random.random() < 0.3:
                out_image = self.augmentation.apply_augmentation(out_image)
        
        inp_tensor = torch.from_numpy(inp_image).permute(2, 0, 1).float() / 255.0
        out_tensor = torch.from_numpy(out_image).permute(2, 0, 1).float() / 255.0
        otv_tensor = torch.from_numpy(otv_image).permute(2, 0, 1).float() / 255.0
        
        inp_tensor = inp_tensor * 2.0 - 1.0
        out_tensor = out_tensor * 2.0 - 1.0
        otv_tensor = otv_tensor * 2.0 - 1.0
        
        return {
            'inp': inp_tensor,
            'out': out_tensor,
            'otv': otv_tensor,
            'name': inp_path.name
        }


class DataLoaderFactory:
    @staticmethod
    def create_train_loader(data_dir='data', batch_size=16, image_size=256, 
                           num_workers=4, shuffle=True):
        dataset = CaptchaDataset(data_dir, image_size, augment=True, mode='train')
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

    @staticmethod
    def create_val_loader(data_dir='data', batch_size=16, image_size=256, 
                         num_workers=4):
        dataset = CaptchaDataset(data_dir, image_size, augment=False, mode='val')
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    @staticmethod
    def create_test_loader(data_dir='data', batch_size=1, image_size=256):
        dataset = CaptchaDataset(data_dir, image_size, augment=False, mode='test')
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )


class SingleImageLoader:
    @staticmethod
    def load_image(image_path, image_size=256):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
        
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor * 2.0 - 1.0
        
        return tensor.unsqueeze(0)

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor.squeeze(0).cpu().detach()
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        image = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
