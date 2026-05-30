import torch
import torch.onnx
from pathlib import Path

from model import CleanerAIModel
from utils import DirectoryManager, get_device


class ModelExporter:
    def __init__(self, model_path=None):
        self.device = get_device()
        self.model = CleanerAIModel().to(self.device)
        self.model.eval()
        
        if model_path is None:
            dir_manager = DirectoryManager()
            model_path = dir_manager.get_latest_checkpoint()
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")

    def export_torchscript(self, output_path='models/cleaner_ai.pt'):
        print(f"Exporting to TorchScript: {output_path}")
        
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(output_path)
        
        print(f"✓ Exported to {output_path}")
        return output_path

    def export_onnx(self, output_path='models/cleaner_ai.onnx', image_size=256):
        print(f"Exporting to ONNX: {output_path}")
        
        dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"✓ Exported to {output_path}")
        return output_path

    def export_quantized(self, output_path='models/cleaner_ai_quantized.pt'):
        print(f"Exporting quantized model: {output_path}")
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        torch.save(quantized_model.state_dict(), output_path)
        
        print(f"✓ Exported to {output_path}")
        return output_path

    def export_all(self, output_dir='models'):
        Path(output_dir).mkdir(exist_ok=True)
        
        print("=" * 60)
        print("Exporting model in all formats...")
        print("=" * 60)
        
        self.export_torchscript(f'{output_dir}/cleaner_ai.pt')
        self.export_onnx(f'{output_dir}/cleaner_ai.onnx')
        
        print("\n✓ All exports complete!")


if __name__ == '__main__':
    exporter = ModelExporter()
    exporter.export_all()
