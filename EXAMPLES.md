# Cleaner AI - Usage Examples

## Example 1: Basic Training

```python
from train import Trainer, Config

config = Config()
config.epochs = 50
config.batch_size = 16
config.learning_rate = 0.001

trainer = Trainer(config)
trainer.train()
```

## Example 2: Generate Custom Dataset

```python
from captcha_generator import CaptchaGenerator

generator = CaptchaGenerator(image_size=512, output_dir='data')
generator.generate_dataset(num_images=5000)
```

## Example 3: Single Image Inference

```python
from infer import Inferencer
import cv2

inferencer = Inferencer()
result = inferencer.infer_image('noisy_captcha.png')
cv2.imwrite('cleaned_captcha.png', result)
```

## Example 4: Batch Processing

```python
from infer import BatchInferencer
from pathlib import Path

batch_inferencer = BatchInferencer(batch_size=32)

image_paths = [str(p) for p in Path('data/inp').glob('*.png')]
results = batch_inferencer.infer_batch_optimized(image_paths)

for i, result in enumerate(results):
    cv2.imwrite(f'output_{i:04d}.png', result)
```

## Example 5: Video Processing

```python
from infer import RealtimeInferencer

realtime = RealtimeInferencer()
realtime.process_video('input_video.mp4', 'output_video.mp4')
```

## Example 6: Model Benchmarking

```python
from benchmark import ModelBenchmark

benchmark = ModelBenchmark()
benchmark.run_full_benchmark()

fps, avg_time = benchmark.benchmark_inference(
    image_size=256,
    batch_size=16,
    num_iterations=100
)
print(f"FPS: {fps:.2f}, Time: {avg_time*1000:.2f}ms")
```

## Example 7: Model Validation

```python
from validate import ModelValidator

validator = ModelValidator()
metrics = validator.validate_and_report(
    data_dir='data',
    num_samples=100
)
```

## Example 8: Model Export

```python
from export_model import ModelExporter

exporter = ModelExporter()
exporter.export_torchscript('models/model.pt')
exporter.export_onnx('models/model.onnx')
exporter.export_quantized('models/model_quantized.pt')
```

## Example 9: Custom Training Loop

```python
import torch
from train import Trainer, Config, LossCalculator
from data_loader import DataLoaderFactory

config = Config()
trainer = Trainer(config)

train_loader = DataLoaderFactory.create_train_loader(
    batch_size=16,
    image_size=256
)

for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        inp = batch['inp'].to(trainer.device)
        otv = batch['otv'].to(trainer.device)
        
        output = trainer.model(inp)
        disc_output = trainer.discriminator(output)
        
        loss_g, _ = trainer.loss_calculator.calculate_generator_loss(
            output, otv, disc_output
        )
        
        trainer.optimizer_g.zero_grad()
        loss_g.backward()
        trainer.optimizer_g.step()
        
        total_loss += loss_g.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")
```

## Example 10: Directory Inference

```python
from infer import Inferencer
from pathlib import Path

inferencer = Inferencer()
inferencer.infer_directory(
    input_dir='data/inp',
    output_dir='data/cleaned',
    image_size=256
)
```

## Example 11: Real-time Webcam Processing

```python
import cv2
from infer import RealtimeInferencer

realtime = RealtimeInferencer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = realtime.process_frame(frame)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Cleaned', processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Example 12: Custom Configuration

```python
from utils import Config
from train import Trainer

config = Config()
config.epochs = 200
config.batch_size = 32
config.learning_rate = 0.0001
config.image_size = 512
config.mixed_precision = True
config.num_workers = 8

trainer = Trainer(config)
trainer.train()
```

## Example 13: Data Augmentation

```python
from utils import AugmentationPipeline
import cv2

augmentation = AugmentationPipeline(image_size=256)

image = cv2.imread('image.png')

augmented = augmentation.apply_augmentation(image, 'all')
cv2.imwrite('augmented.png', augmented)
```

## Example 14: Checkpoint Management

```python
from utils import DirectoryManager
from train import Trainer, Config

dir_manager = DirectoryManager()

latest_checkpoint = dir_manager.get_latest_checkpoint()
print(f"Latest checkpoint: {latest_checkpoint}")

config = Config()
trainer = Trainer(config)
trainer.load_checkpoint()

trainer.train()
```

## Example 15: Performance Profiling

```python
import torch
import time
from model import CleanerAIModel

model = CleanerAIModel().cuda()
model.eval()

dummy_input = torch.randn(1, 3, 256, 256).cuda()

with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)

torch.cuda.synchronize()
end = time.time()

print(f"Time: {(end-start)*1000:.2f}ms")
print(f"FPS: {100 / (end-start):.2f}")
```

## Example 16: Metadata Extraction

```python
from utils import MetadataManager
from pathlib import Path
import json

metadata_path = Path('data/inp/captcha_00000.json')
metadata = MetadataManager.load_metadata(metadata_path)

print(f"Image: {metadata['image_name']}")
print(f"Text: {metadata['text']}")
print(f"Noise level: {metadata['noise_level']}")
print(f"Noise types: {metadata['noise_types']}")
```

## Example 17: Dataset Caching

```python
from utils import DatasetCache
import torch

cache = DatasetCache('cache')

data = torch.randn(100, 3, 256, 256)
cache.save('dataset_batch_1', data)

loaded_data = cache.load('dataset_batch_1')
print(f"Cached data shape: {loaded_data.shape}")
```

## Example 18: Learning Rate Scheduling

```python
from utils import LRScheduler
import torch.optim as optim
from model import CleanerAIModel

model = CleanerAIModel()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

scheduler = LRScheduler(optimizer, base_lr=0.001, total_epochs=100)

for epoch in range(100):
    lr = scheduler.step(epoch)
    print(f"Epoch {epoch}: LR = {lr:.6f}")
```

## Example 19: EMA Weight Averaging

```python
from utils import EMAWeights
from model import CleanerAIModel
import torch

model = CleanerAIModel()
ema = EMAWeights(model, decay=0.999)

for step in range(1000):
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    
    ema.update()

ema.apply_shadow()
torch.save(model.state_dict(), 'ema_model.pt')
ema.restore()
```

## Example 20: Full Pipeline

```python
from captcha_generator import CaptchaGenerator
from train import Trainer, Config
from infer import Inferencer
from validate import ModelValidator
import cv2

print("Step 1: Generate dataset")
generator = CaptchaGenerator(image_size=256)
generator.generate_dataset(num_images=1000)

print("Step 2: Train model")
config = Config()
config.epochs = 50
trainer = Trainer(config)
trainer.train()

print("Step 3: Validate model")
validator = ModelValidator()
metrics = validator.validate_and_report(num_samples=100)

print("Step 4: Test inference")
inferencer = Inferencer()
result = inferencer.infer_image('data/inp/captcha_00000.png')
cv2.imwrite('final_result.png', result)

print("Pipeline complete!")
```
