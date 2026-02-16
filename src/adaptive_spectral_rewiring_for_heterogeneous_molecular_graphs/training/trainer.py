"""Training loop with early stopping and learning rate scheduling."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for molecular GNN with adaptive spectral rewiring."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        scheduler_params: Optional[Dict] = None,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 20,
        save_dir: str = "./models",
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            optimizer_type: Type of optimizer ('adam' or 'adamw').
            scheduler_type: Type of LR scheduler ('cosine', 'step', 'plateau').
            scheduler_params: Parameters for the scheduler.
            gradient_clip: Gradient clipping value.
            early_stopping_patience: Patience for early stopping.
            save_dir: Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        if optimizer_type.lower() == "adam":
            self.optimizer = Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "adamw":
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler
        if scheduler_params is None:
            scheduler_params = {}

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get("T_max", 100),
                eta_min=scheduler_params.get("eta_min", 0.00001),
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_params.get("step_size", 30),
                gamma=scheduler_params.get("gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_params.get("factor", 0.5),
                patience=scheduler_params.get("patience", 10),
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        self.scheduler_type = scheduler_type

        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_motif_loss = 0.0
        num_batches = 0

        # Use modern torch.amp API (replaces deprecated torch.cuda.amp)
        scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            if use_amp and torch.cuda.is_available():
                # Use torch.amp instead of deprecated torch.cuda.amp
                with torch.amp.autocast('cuda'):
                    logits, rewiring_loss = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        return_rewiring_loss=True
                    )

                    # Compute task loss
                    task_loss = nn.functional.binary_cross_entropy_with_logits(
                        logits.view(-1),
                        batch.y.float().view(-1)
                    )

                    # Total loss with configurable rewiring weight
                    # NOTE: 0.01 is a hyperparameter that balances task loss vs. rewiring loss
                    loss = task_loss + 0.01 * rewiring_loss

                scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )

                scaler.step(self.optimizer)
                scaler.update()
            else:
                logits, rewiring_loss = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    return_rewiring_loss=True
                )

                # Compute task loss
                task_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    batch.y.float().view(-1)
                )

                # Total loss
                loss = task_loss + 0.01 * rewiring_loss

                loss.backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )

                self.optimizer.step()

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_motif_loss += rewiring_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        metrics = {
            'loss': total_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'motif_loss': total_motif_loss / num_batches,
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device)

            logits, rewiring_loss = self.model(
                batch.x,
                batch.edge_index,
                batch.batch,
                return_rewiring_loss=True
            )

            # Compute task loss
            task_loss = nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1),
                batch.y.float().view(-1)
            )

            # Total loss
            loss = task_loss + 0.01 * rewiring_loss

            total_loss += loss.item()
            num_batches += 1

        metrics = {
            'loss': total_loss / num_batches,
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        use_amp: bool = False,
        mlflow_logging: bool = False,
    ) -> Dict[str, list]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            use_amp: Whether to use automatic mixed precision.
            mlflow_logging: Whether to log to MLflow.

        Returns:
            Dictionary of training history.
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, use_amp=use_amp)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            if self.scheduler_type == "plateau":
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            if mlflow_logging:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'learning_rate': current_lr,
                    }, step=epoch)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

            # Save history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)

            # Early stopping and checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0

                # Save best model
                checkpoint_path = self.save_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break

        return {
            'train': self.train_history,
            'val': self.val_history,
        }

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
