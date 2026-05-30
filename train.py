import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

from model import CleanerAIModel, Discriminator
from data_loader import DataLoaderFactory
from utils import (
    Config, DirectoryManager, set_seed, count_parameters,
    LRScheduler, EMAWeights, get_device
)


class LossCalculator:
    def __init__(self, device):
        self.device = device
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = self._build_perceptual_loss()

    def _build_perceptual_loss(self):
        class PerceptualLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                return self.features(x)

        return PerceptualLoss().to(self.device)

    def calculate_generator_loss(self, output, target, discriminator_output):
        l1 = self.l1_loss(output, target)
        l2 = self.l2_loss(output, target)
        
        perceptual = self.perceptual_loss(output)
        perceptual_target = self.perceptual_loss(target)
        perceptual_loss = self.l1_loss(perceptual, perceptual_target)
        
        adversarial_loss = -discriminator_output.mean()
        
        total_loss = l1 * 0.5 + l2 * 0.3 + perceptual_loss * 0.1 + adversarial_loss * 0.1
        
        return total_loss, {
            'l1': l1.item(),
            'l2': l2.item(),
            'perceptual': perceptual_loss.item(),
            'adversarial': adversarial_loss.item()
        }

    def calculate_discriminator_loss(self, real_output, fake_output):
        real_loss = -real_output.mean()
        fake_loss = fake_output.mean()
        total_loss = real_loss + fake_loss
        return total_loss, {'real': real_loss.item(), 'fake': fake_loss.item()}


class Trainer:
    def __init__(self, config=None):
        self.config = config or Config()
        set_seed(self.config.seed)
        
        self.device = self.config.device
        self.dir_manager = DirectoryManager(self.config.data_dir)
        
        self.model = CleanerAIModel().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        self.optimizer_g = optim.AdamW(self.model.parameters(), 
                                       lr=self.config.learning_rate, 
                                       betas=(0.9, 0.999))
        self.optimizer_d = optim.AdamW(self.discriminator.parameters(), 
                                       lr=self.config.learning_rate * 0.1, 
                                       betas=(0.9, 0.999))
        
        self.lr_scheduler_g = LRScheduler(self.optimizer_g, self.config.learning_rate, 
                                         self.config.epochs)
        self.lr_scheduler_d = LRScheduler(self.optimizer_d, self.config.learning_rate * 0.1, 
                                         self.config.epochs)
        
        self.ema_weights = EMAWeights(self.model, decay=0.999)
        
        self.loss_calculator = LossCalculator(self.device)
        
        if self.config.mixed_precision and torch.cuda.is_available():
            self.scaler_g = GradScaler('cuda')
            self.scaler_d = GradScaler('cuda')
        else:
            self.scaler_g = None
            self.scaler_d = None
        
        self.writer = SummaryWriter(str(self.dir_manager.logs_dir))
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Discriminator parameters: {count_parameters(self.discriminator):,}")
        print(f"Device: {self.device}")

    def load_checkpoint(self):
        checkpoint_path = self.dir_manager.get_latest_checkpoint()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state'])
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
            print(f"Loaded checkpoint from epoch {self.start_epoch}")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.discriminator.train()
        
        total_loss_g = 0
        total_loss_d = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            inp = batch['inp'].to(self.device)
            out = batch['out'].to(self.device)
            otv = batch['otv'].to(self.device)
            
            self.optimizer_g.zero_grad()
            
            if self.config.mixed_precision and self.scaler_g is not None:
                with autocast('cuda'):
                    output = self.model(inp)
                    disc_output = self.discriminator(output)
                    loss_g, loss_dict_g = self.loss_calculator.calculate_generator_loss(
                        output, otv, disc_output
                    )
                
                self.scaler_g.scale(loss_g).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                output = self.model(inp)
                disc_output = self.discriminator(output)
                loss_g, loss_dict_g = self.loss_calculator.calculate_generator_loss(
                    output, otv, disc_output
                )
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer_g.step()
            
            self.optimizer_d.zero_grad()
            
            if self.config.mixed_precision and self.scaler_d is not None:
                with autocast('cuda'):
                    real_output = self.discriminator(otv.detach())
                    fake_output = self.discriminator(output.detach())
                    loss_d, loss_dict_d = self.loss_calculator.calculate_discriminator_loss(
                        real_output, fake_output
                    )
                
                self.scaler_d.scale(loss_d).backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                real_output = self.discriminator(otv.detach())
                fake_output = self.discriminator(output.detach())
                loss_d, loss_dict_d = self.loss_calculator.calculate_discriminator_loss(
                    real_output, fake_output
                )
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.optimizer_d.step()
            
            self.ema_weights.update()
            
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss_g': loss_g.item(),
                'loss_d': loss_d.item(),
                'lr': self.optimizer_g.param_groups[0]['lr']
            })
        
        avg_loss_g = total_loss_g / num_batches
        avg_loss_d = total_loss_d / num_batches
        
        self.writer.add_scalar('Loss/train_generator', avg_loss_g, epoch)
        self.writer.add_scalar('Loss/train_discriminator', avg_loss_d, epoch)
        self.writer.add_scalar('LR/generator', self.optimizer_g.param_groups[0]['lr'], epoch)
        
        return avg_loss_g, avg_loss_d

    def validate(self, val_loader, epoch):
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inp = batch['inp'].to(self.device)
                otv = batch['otv'].to(self.device)
                
                output = self.model(inp)
                loss = self.loss_calculator.l1_loss(output, otv)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        
        return avg_loss

    def train(self):
        train_loader = DataLoaderFactory.create_train_loader(
            self.config.data_dir,
            self.config.batch_size,
            self.config.image_size,
            self.config.num_workers
        )
        
        val_loader = DataLoaderFactory.create_val_loader(
            self.config.data_dir,
            self.config.batch_size,
            self.config.image_size,
            self.config.num_workers
        )
        
        self.load_checkpoint()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            self.lr_scheduler_g.step(epoch)
            self.lr_scheduler_d.step(epoch)
            
            train_loss_g, train_loss_d = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            
            self.history['train_loss'].append(train_loss_g)
            self.history['val_loss'].append(val_loss)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_best_checkpoint(epoch)
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
            
            print(f"Epoch {epoch + 1}: Train Loss G={train_loss_g:.4f}, "
                  f"Train Loss D={train_loss_d:.4f}, Val Loss={val_loss:.4f}")
        
        self.writer.close()
        print("Training complete!")

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }
        path = self.dir_manager.models_dir / f'checkpoint_{epoch:05d}.pt'
        torch.save(checkpoint, path)

    def save_best_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }
        path = self.dir_manager.models_dir / 'best_model.pt'
        torch.save(checkpoint, path)


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.train()
