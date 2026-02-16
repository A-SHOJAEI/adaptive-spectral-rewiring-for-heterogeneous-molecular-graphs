#!/usr/bin/env python3
"""
Evaluation script for Adaptive Spectral Rewiring on molecular graphs.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.data.loader import (
    load_molecular_dataset,
    create_dataloaders
)
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.evaluation.metrics import compute_metrics


def setup_logging(log_file=None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, device, logger):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    if config is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Load dataset to get dimensions
    dataset, _ = load_molecular_dataset(
        dataset_name=config['data']['dataset'],
        root=config['data']['data_dir']
    )

    # Create model
    model = AdaptiveSpectralGNN(
        num_features=dataset.num_features,
        num_classes=dataset.num_tasks,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        rewiring_ratio=config['model']['rewiring'].get('edge_budget', 1.5) - 1.0,
        pooling=config['model']['pooling']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    return model, config, dataset


def evaluate_model(model, data_loader, device, logger):
    """Evaluate model on a dataset."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    num_samples = 0

    logger.info("Running evaluation...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device)

            # Forward pass
            logits, _ = model(batch.x, batch.edge_index, batch.batch, return_rewiring_loss=False)

            # Compute loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1),
                batch.y.float().view(-1)
            )
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).flatten()

    avg_loss = total_loss / num_samples

    # Compute metrics
    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=all_probs,
        task_type="binary"
    )

    metrics['loss'] = avg_loss

    return metrics, all_preds, all_labels, all_probs


def save_predictions(predictions, labels, probabilities, output_path, logger):
    """Save predictions to CSV file."""
    df = pd.DataFrame({
        'prediction': predictions,
        'label': labels,
        'probability': probabilities,
        'correct': (predictions == labels).astype(int)
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to: {output_path}")


def main(args):
    """Main evaluation function."""
    # Set up logging
    logger = setup_logging(args.log_file)
    logger.info("=" * 80)
    logger.info("Starting evaluation")
    logger.info("=" * 80)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, config, dataset = load_model_from_checkpoint(args.checkpoint, device, logger)

    # Load dataset and create split
    dataset, split_idx = load_molecular_dataset(
        dataset_name=config['data']['dataset'],
        root=config['data']['data_dir']
    )

    # Create dataloader for the requested split
    _, _, test_loader = create_dataloaders(
        dataset,
        split_idx,
        batch_size=args.batch_size,
        num_workers=4
    )

    # Determine which loader to use
    if args.split == "test":
        data_loader = test_loader
    else:
        # For simplicity, use test loader
        logger.warning(f"Only test split is supported, using test split")
        data_loader = test_loader

    logger.info(f"Evaluating on {args.split} split")

    # Evaluate
    metrics, predictions, labels, probabilities = evaluate_model(
        model, data_loader, device, logger
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results:")
    logger.info("=" * 80)
    for metric_name, metric_value in sorted(metrics.items()):
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info("=" * 80)

    # Save predictions
    if args.save_predictions:
        output_path = Path(args.output_dir) / f"predictions_{args.split}.csv"
        save_predictions(predictions, labels, probabilities, output_path, logger)

    logger.info("\nEvaluation completed successfully!")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Adaptive Spectral Rewiring model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
