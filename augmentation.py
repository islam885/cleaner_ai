import albumentations as A
import cv2
import numpy as np


class AugmentationPipeline:
    def __init__(self, image_size=256):
        self.image_size = image_size
        
        self.train_transform = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.MultiplicativeNoise(),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
        ])
    
    def apply_train(self, image, target):
        transformed = self.train_transform(image=image, mask=target)
        return transformed['image'], transformed['mask']
    
    def apply_val(self, image, target):
        transformed = self.val_transform(image=image, mask=target)
        return transformed['image'], transformed['mask']


class AdvancedNoiseGenerator:
    @staticmethod
    def add_jpeg_artifacts(image, quality=30):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)
    
    @staticmethod
    def add_scratches(image, num_scratches=10):
        img = image.copy()
        h, w = img.shape[:2]
        
        for _ in range(num_scratches):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            thickness = np.random.randint(1, 3)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        return img
    
    @staticmethod
    def add_dots(image, num_dots=50):
        img = image.copy()
        h, w = img.shape[:2]
        
        for _ in range(num_dots):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(1, 4)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(img, (x, y), radius, color, -1)
        
        return img
    
    @staticmethod
    def add_wave_distortion(image, amplitude=10, frequency=0.1):
        img = image.copy()
        h, w = img.shape[:2]
        
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                map_x[i, j] = j + amplitude * np.sin(2 * np.pi * i * frequency)
                map_y[i, j] = i + amplitude * np.cos(2 * np.pi * j * frequency)
        
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    @staticmethod
    def add_grid_pattern(image, grid_size=20, line_width=1):
        img = image.copy()
        h, w = img.shape[:2]
        color = tuple(np.random.randint(100, 200, 3).tolist())
        
        for i in range(0, h, grid_size):
            cv2.line(img, (0, i), (w, i), color, line_width)
        
        for j in range(0, w, grid_size):
            cv2.line(img, (j, 0), (j, h), color, line_width)
        
        return img
    
    @staticmethod
    def add_dirty_texture(image, intensity=0.3):
        img = image.copy().astype(np.float32)
        h, w = img.shape[:2]
        
        texture = np.random.randn(h, w, 3) * 50 * intensity
        img = np.clip(img + texture, 0, 255)
        
        return img.astype(np.uint8)
