"""Training pipeline for multimodal stroke prognosis model.

This module provides a comprehensive trainer class that handles:
- Training loop with mixed precision support
- Validation and early stopping
- Checkpointing and model saving
- Metrics tracking and logging
- Learning rate scheduling
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.metrics import compute_metrics
from ..utils.helpers import set_random_seed, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for multimodal stroke prognosis model.
    
    This class encapsulates the complete training pipeline including:
    - Training and validation loops
    - Loss computation and backpropagation
    - Metrics tracking
    - Model checkpointing
    - Early stopping
    
    Args:
        model: The multimodal model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for model parameters
        criterion: Loss function
        device: Device to run training on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
        max_epochs: Maximum number of training epochs
        early_stopping_patience: Epochs to wait before early stopping
        checkpoint_dir: Directory to save model checkpoints
        use_amp: Whether to use automatic mixed precision
        log_interval: Steps between logging updates
        save_best_only: Only save checkpoints that improve validation metric
        
    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     criterion=nn.BCEWithLogitsLoss(),
        ...     device='cuda',
        ...     max_epochs=100,
        ...     early_stopping_patience=10
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        scheduler: Optional[_LRScheduler] = None,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = './checkpoints',
        use_amp: bool = True,
        log_interval: int = 10,
        save_best_only: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.log_interval = log_interval
        self.save_best_only = save_best_only
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_acc': [],
            'learning_rates': [],
        }
        
        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Using AMP: {self.use_amp}")
    
    def train(self) -> Dict[str, list]:
        """Execute complete training loop.
        
        Returns:
            Dictionary containing training history with keys:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch
                - 'val_auc': List of validation AUC scores per epoch
                - 'val_acc': List of validation accuracies per epoch
                - 'learning_rates': List of learning rates per epoch
        
        Example:
            >>> history = trainer.train()
            >>> print(f"Best val AUC: {max(history['val_auc']):.4f}")
        """
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_metrics = self._validate_epoch()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            # Model checkpointing
            is_best = val_metrics['auc'] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics['auc']
                self.patience_counter = 0
                logger.info(f"New best validation AUC: {self.best_val_metric:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if not self.save_best_only or is_best:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    path=checkpoint_path,
                )
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save best model separately
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pt'
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    path=best_path,
                )
                logger.info(f"Best model saved: {best_path}")
            
            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs. "
                    f"Best val AUC: {self.best_val_metric:.4f}"
                )
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation AUC: {self.best_val_metric:.4f}")
        
        return self.history
    
    def _train_epoch(self) -> float:
        """Execute one training epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            texts = batch['text'].to(self.device)
            tables = batch['table'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with AMP
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, texts, tables)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, texts, tables)
                loss = self.criterion(outputs, labels)
                
                # Standard backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Execute validation for one epoch.
        
        Returns:
            Dictionary containing validation metrics:
                - 'loss': Average validation loss
                - 'auc': Area under ROC curve
                - 'accuracy': Classification accuracy
                - 'precision': Precision score
                - 'recall': Recall score
                - 'f1': F1 score
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            images = batch['image'].to(self.device)
            texts = batch['text'].to(self.device)
            tables = batch['table'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, texts, tables)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images, texts, tables)
                loss = self.criterion(outputs, labels)
            
            # Collect predictions
            total_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
        
        # Concatenate all predictions
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(all_outputs, all_labels)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Dictionary containing test metrics
        
        Example:
            >>> test_metrics = trainer.evaluate(test_loader)
            >>> print(f"Test AUC: {test_metrics['auc']:.4f}")
        """
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch to device
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)
                tables = batch['table'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, texts, tables)
                else:
                    outputs = self.model(images, texts, tables)
                
                # Collect predictions
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all predictions
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(all_outputs, all_labels)
        
        logger.info("Test Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional metadata to save
        """
        checkpoint_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
            'patience_counter': self.patience_counter,
            'history': self.history,
        }
        checkpoint_data.update(kwargs)
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics={'best_val_auc': self.best_val_metric},
            path=path,
            **checkpoint_data,
        )
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            path=path,
            device=self.device,
        )
        
        # Restore training state
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        if 'best_val_metric' in checkpoint:
            self.best_val_metric = checkpoint['best_val_metric']
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded: {path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
