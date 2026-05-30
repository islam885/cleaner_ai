import os
import json
import random
import string
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io


class CaptchaGenerator:
    def __init__(self, image_size=256, output_dir='data'):
        self.image_size = image_size
        self.output_dir = Path(output_dir)
        self.inp_dir = self.output_dir / 'inp'
        self.out_dir = self.output_dir / 'out'
        self.otv_dir = self.output_dir / 'otv'
        
        for d in [self.inp_dir, self.out_dir, self.otv_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.fonts = self._load_fonts()
        self.noise_ids = {}
        self.line_ids = {}

    def _load_fonts(self):
        fonts = []
        font_paths = [
            'C:\\Windows\\Fonts\\arial.ttf',
            'C:\\Windows\\Fonts\\times.ttf',
            'C:\\Windows\\Fonts\\cour.ttf',
            'C:\\Windows\\Fonts\\verdana.ttf',
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    fonts.append(path)
                except:
                    pass
        
        return fonts if fonts else [None]

    def _generate_text(self, length=6):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))

    def _get_random_font(self, size=40):
        if self.fonts[0] is None:
            return None
        font_path = random.choice(self.fonts)
        try:
            return ImageFont.truetype(font_path, size)
        except:
            return ImageFont.load_default()

    def _add_noise(self, image, noise_level=0.3):
        noise_coords = []
        noise_types = []
        noise_ids = []
        
        h, w = image.shape[:2]
        
        if random.random() < 0.7:
            num_dots = int(h * w * noise_level * 0.01)
            for _ in range(num_dots):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.circle(image, (x, y), random.randint(1, 3), color, -1)
                noise_coords.append([x, y])
                noise_types.append('dot')
                noise_ids.append(self._generate_noise_id())

        if random.random() < 0.6:
            num_scratches = int(noise_level * 5)
            for _ in range(num_scratches):
                x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
                x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
                color = tuple(random.randint(0, 255) for _ in range(3))
                thickness = random.randint(1, 3)
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                noise_coords.append([[x1, y1], [x2, y2]])
                noise_types.append('scratch')
                noise_ids.append(self._generate_noise_id())

        if random.random() < 0.5:
            num_grids = random.randint(1, 3)
            for _ in range(num_grids):
                grid_size = random.randint(10, 30)
                color = tuple(random.randint(100, 200) for _ in range(3))
                for i in range(0, w, grid_size):
                    cv2.line(image, (i, 0), (i, h), color, 1)
                for i in range(0, h, grid_size):
                    cv2.line(image, (0, i), (w, i), color, 1)
                noise_types.append('grid')
                noise_ids.append(self._generate_noise_id())

        if random.random() < 0.4:
            num_waves = random.randint(1, 2)
            for _ in range(num_waves):
                amplitude = random.randint(2, 8)
                frequency = random.randint(1, 3)
                color = tuple(random.randint(100, 200) for _ in range(3))
                for x in range(w):
                    y = int(h / 2 + amplitude * np.sin(2 * np.pi * frequency * x / w))
                    if 0 <= y < h:
                        cv2.circle(image, (x, y), 1, color, -1)
                noise_types.append('wave')
                noise_ids.append(self._generate_noise_id())

        return image, noise_coords, noise_types, noise_ids

    def _add_lines(self, image):
        line_coords = []
        line_ids = []
        
        h, w = image.shape[:2]
        num_lines = random.randint(2, 8)
        
        for _ in range(num_lines):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            x2 = random.randint(0, w - 1)
            y2 = random.randint(0, h - 1)
            
            color = tuple(random.randint(50, 200) for _ in range(3))
            thickness = random.randint(1, 4)
            
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
            
            line_coords.append([[x1, y1], [x2, y2]])
            line_ids.append(self._generate_noise_id())
        
        return image, line_coords, line_ids

    def _add_background(self, image):
        h, w = image.shape[:2]
        
        bg_type = random.choice(['solid', 'gradient', 'texture', 'pattern'])
        
        if bg_type == 'solid':
            bg_color = tuple(random.randint(100, 255) for _ in range(3))
            bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
        
        elif bg_type == 'gradient':
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            color1 = np.array(random.choices(range(100, 255), k=3))
            color2 = np.array(random.choices(range(100, 255), k=3))
            for i in range(h):
                ratio = i / h
                bg[i] = (color1 * (1 - ratio) + color2 * ratio).astype(np.uint8)
        
        elif bg_type == 'texture':
            bg = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
            bg = cv2.GaussianBlur(bg, (5, 5), 0)
        
        else:
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(random.randint(50, 200)):
                x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
                x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
                color = tuple(random.randint(100, 200) for _ in range(3))
                cv2.line(bg, (x1, y1), (x2, y2), color, 1)
        
        alpha = random.uniform(0.3, 0.7)
        return cv2.addWeighted(image, 1 - alpha, bg, alpha, 0)

    def _add_distortion(self, image):
        h, w = image.shape[:2]
        
        if random.random() < 0.5:
            map_x = np.arange(w)
            map_y = np.arange(h)
            map_x, map_y = np.meshgrid(map_x, map_y)
            
            amplitude = random.uniform(5, 15)
            frequency = random.uniform(0.01, 0.05)
            
            map_x = map_x + amplitude * np.sin(2 * np.pi * frequency * map_y)
            map_y = map_y + amplitude * np.cos(2 * np.pi * frequency * map_x)
            
            map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
            map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
            
            image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
        
        return image

    def _add_blur(self, image):
        if random.random() < 0.6:
            kernel_size = random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        if random.random() < 0.4:
            image = cv2.medianBlur(image, random.choice([3, 5]))
        
        return image

    def _add_compression_artifacts(self, image):
        if random.random() < 0.5:
            quality = random.randint(30, 80)
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        return image

    def _generate_noise_id(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    def generate_captcha(self, index):
        text = self._generate_text()
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        
        h, w = self.image_size, self.image_size
        
        clean_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        font = self._get_random_font(size=random.randint(30, 50))
        text_color = tuple(random.randint(0, 100) for _ in range(3))
        
        pil_image = Image.fromarray(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (w - text_width) // 2
        y = (h - text_height) // 2
        
        draw.text((x, y), text, fill=text_color, font=font)
        clean_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        otv_image = clean_image.copy()
        
        out_image = clean_image.copy()
        noise_coords, noise_types, noise_ids = [], [], []
        line_coords, line_ids = [], []
        
        noise_level = random.uniform(0.1, 0.5)
        
        out_image, noise_coords, noise_types, noise_ids = self._add_noise(out_image, noise_level)
        out_image, line_coords, line_ids = self._add_lines(out_image)
        
        inp_image = out_image.copy()
        inp_image = self._add_background(inp_image)
        inp_image = self._add_distortion(inp_image)
        inp_image = self._add_blur(inp_image)
        inp_image = self._add_compression_artifacts(inp_image)
        
        if random.random() < 0.3:
            brightness = random.uniform(0.7, 1.3)
            inp_image = cv2.convertScaleAbs(inp_image, alpha=brightness, beta=0)
        
        if random.random() < 0.3:
            contrast = random.uniform(0.7, 1.3)
            inp_image = cv2.convertScaleAbs(inp_image, alpha=contrast, beta=0)
        
        filename = f'captcha_{index:05d}.png'
        
        cv2.imwrite(str(self.inp_dir / filename), inp_image)
        cv2.imwrite(str(self.out_dir / filename), out_image)
        cv2.imwrite(str(self.otv_dir / filename), otv_image)
        
        metadata = {
            'image_name': filename,
            'text': text,
            'noise_coordinates': noise_coords,
            'line_coordinates': line_coords,
            'noise_types': noise_types,
            'line_ids': line_ids,
            'noise_ids': noise_ids,
            'noise_level': float(noise_level),
            'random_seed': seed,
            'bbox': [int(x), int(y), int(x + text_width), int(y + text_height)]
        }
        
        metadata_path = self.inp_dir / f'{filename[:-4]}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filename, metadata

    def generate_dataset(self, num_images=100):
        print(f"Generating {num_images} CAPTCHA images...")
        for i in range(num_images):
            self.generate_captcha(i)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_images} images")
        print("Dataset generation complete!")


if __name__ == '__main__':
    generator = CaptchaGenerator(image_size=256, output_dir='data')
    generator.generate_dataset(num_images=100)
