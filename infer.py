import torch
import cv2
import numpy as np
from pathlib import Path
from model import CleanerAIModel
from data_loader import SingleImageLoader
from utils import DirectoryManager, get_device


class Inferencer:
    def __init__(self, model_path=None, device=None):
        self.device = device or get_device()
        self.model = CleanerAIModel().to(self.device)
        self.model.eval()
        
        if model_path is None:
            dir_manager = DirectoryManager()
            model_path = dir_manager.get_latest_checkpoint()
        
        if model_path is None:
            raise ValueError("No model found. Please train a model first.")
        
        self.load_model(model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")

    def infer_image(self, image_path, image_size=256):
        tensor = SingleImageLoader.load_image(image_path, image_size)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
        
        result_image = SingleImageLoader.tensor_to_image(output)
        return result_image

    def infer_batch(self, image_paths, image_size=256):
        results = []
        for image_path in image_paths:
            result = self.infer_image(image_path, image_size)
            results.append(result)
        return results

    def infer_directory(self, input_dir, output_dir, image_size=256):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
        
        for image_path in image_files:
            result = self.infer_image(str(image_path), image_size)
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), result)
            print(f"Processed: {image_path.name}")

    def infer_with_visualization(self, image_path, image_size=256):
        original = cv2.imread(str(image_path))
        result = self.infer_image(image_path, image_size)
        
        original_resized = cv2.resize(original, (image_size, image_size))
        
        comparison = np.hstack([original_resized, result])
        
        return original_resized, result, comparison


class RealtimeInferencer:
    def __init__(self, model_path=None, device=None):
        self.inferencer = Inferencer(model_path, device)
        self.device = self.inferencer.device
        self.model = self.inferencer.model

    def process_frame(self, frame, image_size=256):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (image_size, image_size))
        
        tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
        
        output = (output + 1.0) / 2.0
        output = torch.clamp(output, 0, 1)
        output_image = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        return output_image

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = self.process_frame(frame)
            
            if output_path:
                out.write(processed)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"Video processing complete! Total frames: {frame_count}")


class BatchInferencer:
    def __init__(self, model_path=None, device=None, batch_size=32):
        self.inferencer = Inferencer(model_path, device)
        self.device = self.inferencer.device
        self.model = self.inferencer.model
        self.batch_size = batch_size

    def infer_batch_optimized(self, image_paths, image_size=256):
        results = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                tensor = SingleImageLoader.load_image(path, image_size)
                batch_tensors.append(tensor)
            
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
            
            for output in outputs:
                result_image = SingleImageLoader.tensor_to_image(output.unsqueeze(0))
                results.append(result_image)
        
        return results


if __name__ == '__main__':
    inferencer = Inferencer()
    
    test_image = 'data/inp/captcha_00000.png'
    if Path(test_image).exists():
        result = inferencer.infer_image(test_image)
        cv2.imwrite('output.png', result)
        print("Inference complete!")
