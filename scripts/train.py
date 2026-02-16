#!/usr/bin/env python3
"""
Training script for Adaptive Spectral Rewiring on molecular graphs.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.data.loader import (
    load_molecular_dataset,
    create_dataloaders
)
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.training.trainer import Trainer


def setup_logging_custom(log_dir: str, experiment_name: str, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"{experiment_name}_train.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_config_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config_yaml(args.config)

    # Override config with command line arguments
    if args.name:
        config['experiment']['name'] = args.name
    if args.device:
        config['experiment']['device'] = args.device
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Set up logging
    logger = setup_logging_custom(
        config['experiment']['log_dir'],
        config['experiment']['name'],
        config['logging']['level']
    )
    logger.info("=" * 80)
    logger.info(f"Starting training: {config['experiment']['name']}")
    logger.info("=" * 80)

    # Set random seed
    set_seed(config['experiment']['seed'])
    logger.info(f"Random seed set to: {config['experiment']['seed']}")

    # Set device
    device = torch.device(
        config['experiment']['device'] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {config['data']['dataset']}")
    dataset, split_idx = load_molecular_dataset(
        dataset_name=config['data']['dataset'],
        root=config['data']['data_dir']
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        split_idx,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Get dataset info
    num_features = dataset.num_features
    num_classes = dataset.num_tasks
    logger.info(f"Dataset info - Features: {num_features}, Classes/Tasks: {num_classes}")

    # Create model
    logger.info("Creating model...")
    model = AdaptiveSpectralGNN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        rewiring_ratio=config['model']['rewiring'].get('edge_budget', 1.5) - 1.0,
        pooling=config['model']['pooling']
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {num_params:,} trainable parameters")

    # Create trainer
    logger.info(f"Optimizer: {config['training']['optimizer']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer_type=config['training']['optimizer'],
        scheduler_type=config['training']['scheduler'],
        gradient_clip=config['training']['grad_clip'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        save_dir=os.path.join(config['experiment']['checkpoint_dir'], config['experiment']['name'])
    )

    # Train model
    logger.info("Starting training...")
    num_epochs = config['training']['epochs']

    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            use_amp=False,
            mlflow_logging=config['logging'].get('use_wandb', False)
        )
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

    # Load best model for final evaluation
    best_model_path = Path(trainer.save_dir) / "best_model.pt"
    if best_model_path.exists():
        trainer.load_checkpoint(str(best_model_path))
        logger.info("Loaded best model for final evaluation")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)

    logger.info("Test Results:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Save final model
    checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / config['experiment']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'config': config
    }, final_checkpoint_path)

    logger.info(f"Final model saved to: {final_checkpoint_path}")

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Adaptive Spectral Rewiring model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training plots"
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
