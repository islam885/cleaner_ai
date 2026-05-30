import sys
import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QTextEdit, QFileDialog, QProgressBar, QComboBox, QCheckBox,
    QSlider, QScrollArea, QGridLayout, QFrame, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRect
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QIcon

from model import CleanerAIModel
from train import Trainer, Config
from infer import Inferencer, RealtimeInferencer
from captcha_generator import CaptchaGenerator
from data_loader import DataLoaderFactory, SingleImageLoader
from utils import DirectoryManager, set_seed


class TrainingThread(QThread):
    progress = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trainer = None
        self.is_running = True

    def run(self):
        try:
            self.trainer = Trainer(self.config)
            self.trainer.train()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False


class GeneratorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, num_images, image_size):
        super().__init__()
        self.num_images = num_images
        self.image_size = image_size

    def run(self):
        try:
            generator = CaptchaGenerator(self.image_size, 'data')
            for i in range(self.num_images):
                generator.generate_captcha(i)
                self.progress.emit(i + 1)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class InferenceThread(QThread):
    result = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_path, model_path=None):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path

    def run(self):
        try:
            inferencer = Inferencer(self.model_path)
            result = inferencer.infer_image(self.image_path)
            self.result.emit(result)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setMinimumSize(512, 512)
        self.label.setStyleSheet("background-color: #1e1e1e; border: 2px solid #3e3e3e;")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.current_image = None

    def set_image(self, image_path):
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
        
        if image is None:
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        if w > 512 or h > 512:
            scale = min(512 / w, 512 / h)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        h, w = image.shape[:2]
        qt_image = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap)
        self.current_image = image


class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.training_thread = None
        self.config = Config()

    def init_ui(self):
        layout = QVBoxLayout()
        
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setValue(100)
        self.epochs_spin.setMaximum(10000)
        grid.addWidget(self.epochs_spin, 0, 1)
        
        grid.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setValue(16)
        self.batch_spin.setMaximum(256)
        grid.addWidget(self.batch_spin, 1, 1)
        
        grid.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setMinimum(0.000001)
        grid.addWidget(self.lr_spin, 2, 1)
        
        grid.addWidget(QLabel("Image Size:"), 3, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setValue(256)
        self.size_spin.setMaximum(1024)
        grid.addWidget(self.size_spin, 3, 1)
        
        grid.addWidget(QLabel("Mixed Precision:"), 4, 0)
        self.mixed_precision_check = QCheckBox()
        self.mixed_precision_check.setChecked(True)
        grid.addWidget(self.mixed_precision_check, 4, 1)
        
        layout.addLayout(grid)
        
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Courier;")
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)

    def start_training(self):
        self.config.epochs = self.epochs_spin.value()
        self.config.batch_size = self.batch_spin.value()
        self.config.learning_rate = self.lr_spin.value()
        self.config.image_size = self.size_spin.value()
        self.config.mixed_precision = self.mixed_precision_check.isChecked()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
        self.log_text.append(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")
        
        self.training_thread = TrainingThread(self.config)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.error.connect(self.on_training_error)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training stopped by user")

    def on_training_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed!")

    def on_training_error(self, error):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {error}")


class GenerateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.generator_thread = None

    def init_ui(self):
        layout = QVBoxLayout()
        
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Number of Images:"), 0, 0)
        self.num_images_spin = QSpinBox()
        self.num_images_spin.setValue(100)
        self.num_images_spin.setMaximum(100000)
        grid.addWidget(self.num_images_spin, 0, 1)
        
        grid.addWidget(QLabel("Image Size:"), 1, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setValue(256)
        self.size_spin.setMaximum(1024)
        grid.addWidget(self.size_spin, 1, 1)
        
        layout.addLayout(grid)
        
        self.generate_btn = QPushButton("Generate Dataset")
        self.generate_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_btn)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Courier;")
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)

    def start_generation(self):
        num_images = self.num_images_spin.value()
        image_size = self.size_spin.value()
        
        self.generate_btn.setEnabled(False)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Generating {num_images} images...")
        
        self.generator_thread = GeneratorThread(num_images, image_size)
        self.generator_thread.progress.connect(self.on_progress)
        self.generator_thread.finished.connect(self.on_finished)
        self.generator_thread.error.connect(self.on_error)
        self.generator_thread.start()

    def on_progress(self, count):
        self.progress_bar.setValue(count)
        if count % 10 == 0:
            self.log_text.append(f"Generated {count} images...")

    def on_finished(self):
        self.generate_btn.setEnabled(True)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Generation complete!")

    def on_error(self, error):
        self.generate_btn.setEnabled(True)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {error}")


class TestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.inference_thread = None

    def init_ui(self):
        layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.infer_btn = QPushButton("Run Inference")
        self.infer_btn.clicked.connect(self.run_inference)
        self.infer_btn.setEnabled(False)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.infer_btn)
        layout.addLayout(button_layout)
        
        self.image_path_label = QLabel("No image loaded")
        layout.addWidget(self.image_path_label)
        
        viewer_layout = QHBoxLayout()
        self.input_viewer = ImageViewer()
        self.output_viewer = ImageViewer()
        viewer_layout.addWidget(self.input_viewer)
        viewer_layout.addWidget(self.output_viewer)
        layout.addLayout(viewer_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Courier;")
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        self.current_image_path = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.current_image_path = file_path
            self.image_path_label.setText(f"Loaded: {Path(file_path).name}")
            self.input_viewer.set_image(file_path)
            self.infer_btn.setEnabled(True)
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Image loaded: {file_path}")

    def run_inference(self):
        if not self.current_image_path:
            return
        
        self.infer_btn.setEnabled(False)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Running inference...")
        
        self.inference_thread = InferenceThread(self.current_image_path)
        self.inference_thread.result.connect(self.on_inference_result)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()

    def on_inference_result(self, result):
        self.output_viewer.set_image(result)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Inference complete!")

    def on_inference_finished(self):
        self.infer_btn.setEnabled(True)

    def on_inference_error(self, error):
        self.infer_btn.setEnabled(True)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {error}")


class ViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.load_dataset)
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.show_next)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        layout.addLayout(button_layout)
        
        self.index_label = QLabel("No dataset loaded")
        layout.addWidget(self.index_label)
        
        viewer_layout = QHBoxLayout()
        self.inp_viewer = ImageViewer()
        self.out_viewer = ImageViewer()
        self.otv_viewer = ImageViewer()
        viewer_layout.addWidget(self.inp_viewer)
        viewer_layout.addWidget(self.out_viewer)
        viewer_layout.addWidget(self.otv_viewer)
        layout.addLayout(viewer_layout)
        
        self.setLayout(layout)
        self.current_index = 0
        self.images = []

    def load_dataset(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            inp_dir = Path(dir_path) / 'inp'
            if inp_dir.exists():
                self.images = sorted(list(inp_dir.glob('*.png')))
                self.current_index = 0
                self.show_image()

    def show_image(self):
        if not self.images:
            return
        
        image_path = self.images[self.current_index]
        out_path = image_path.parent.parent / 'out' / image_path.name
        otv_path = image_path.parent.parent / 'otv' / image_path.name
        
        self.inp_viewer.set_image(str(image_path))
        self.out_viewer.set_image(str(out_path))
        self.otv_viewer.set_image(str(otv_path))
        
        self.index_label.setText(f"Image {self.current_index + 1}/{len(self.images)}: {image_path.name}")

    def show_previous(self):
        if self.images:
            self.current_index = (self.current_index - 1) % len(self.images)
            self.show_image()

    def show_next(self):
        if self.images:
            self.current_index = (self.current_index + 1) % len(self.images)
            self.show_image()


class LogsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Logs")
        self.refresh_btn.clicked.connect(self.refresh_logs)
        self.clear_btn = QPushButton("Clear Logs")
        self.clear_btn.clicked.connect(self.clear_logs)
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.clear_btn)
        layout.addLayout(button_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Courier;")
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)

    def refresh_logs(self):
        log_dir = Path('logs')
        if log_dir.exists():
            self.log_text.clear()
            for log_file in sorted(log_dir.glob('*.log')):
                with open(log_file, 'r') as f:
                    self.log_text.append(f"=== {log_file.name} ===")
                    self.log_text.append(f.read())

    def clear_logs(self):
        self.log_text.clear()


class CleanerAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cleaner AI - CAPTCHA Noise Removal")
        self.setGeometry(100, 100, 1400, 900)
        
        self.set_dark_theme()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        self.train_tab = TrainTab()
        self.generate_tab = GenerateTab()
        self.test_tab = TestTab()
        self.viewer_tab = ViewerTab()
        self.logs_tab = LogsTab()
        
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.generate_tab, "Generate")
        self.tabs.addTab(self.test_tab, "Test")
        self.tabs.addTab(self.viewer_tab, "Viewer")
        self.tabs.addTab(self.logs_tab, "Logs")
        
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        
        DirectoryManager().create_all()

    def set_dark_theme(self):
        dark_stylesheet = """
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e3e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 5px 15px;
                border: 1px solid #3e3e3e;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1084d7;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3e3e3e;
                padding: 5px;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                border: 1px solid #3e3e3e;
                font-family: Courier;
            }
            QProgressBar {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
            QLabel {
                color: #ffffff;
            }
            QCheckBox {
                color: #ffffff;
            }
        """
        self.setStyleSheet(dark_stylesheet)


def main():
    app = QApplication(sys.argv)
    window = CleanerAIApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
