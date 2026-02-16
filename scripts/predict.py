#!/usr/bin/env python3
"""
Prediction script for Adaptive Spectral Rewiring on molecular graphs.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Batch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.data.loader import load_molecular_dataset
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.data.preprocessing import smiles_to_graph
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_model(checkpoint_path, device, logger):
    """Load model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Get dataset info
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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    return model, config


def load_smiles_from_file(file_path, logger):
    """Load SMILES strings from file."""
    logger.info(f"Loading SMILES from: {file_path}")

    file_path = Path(file_path)

    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        # Try common column names
        for col in ['smiles', 'SMILES', 'smile', 'SMILE']:
            if col in df.columns:
                smiles_list = df[col].tolist()
                break
        else:
            # Use first column
            smiles_list = df.iloc[:, 0].tolist()
    elif file_path.suffix == '.txt':
        with open(file_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(smiles_list)} SMILES strings")

    return smiles_list


def validate_smiles(smiles):
    """Validate SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def predict_batch(model, graphs, device, batch_size=32):
    """Predict for a batch of graphs."""
    predictions = []
    probabilities = []

    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i:i+batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)

        with torch.no_grad():
            logits, _ = model(batch.x, batch.edge_index, batch.batch, return_rewiring_loss=False)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

    return np.array(predictions).flatten(), np.array(probabilities).flatten()


def main(args):
    """Main prediction function."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting prediction")
    logger.info("=" * 80)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device, logger)

    # Load SMILES
    if args.smiles:
        smiles_list = [args.smiles]
    elif args.smiles_file:
        smiles_list = load_smiles_from_file(args.smiles_file, logger)
    else:
        raise ValueError("Either --smiles or --smiles-file must be provided")

    # Validate SMILES
    logger.info("Validating SMILES strings...")
    valid_smiles = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_list):
        if validate_smiles(smiles):
            valid_smiles.append(smiles)
            valid_indices.append(idx)
        else:
            logger.warning(f"Invalid SMILES at index {idx}: {smiles}")

    logger.info(f"Valid SMILES: {len(valid_smiles)} / {len(smiles_list)}")

    if len(valid_smiles) == 0:
        logger.error("No valid SMILES found")
        return

    # Convert SMILES to graphs
    logger.info("Converting SMILES to graphs...")
    graphs = []

    for smiles in tqdm(valid_smiles, desc="Processing"):
        try:
            graph = smiles_to_graph(smiles)
            graphs.append(graph)
        except Exception as e:
            logger.warning(f"Failed to convert SMILES: {smiles}, Error: {e}")

    if len(graphs) == 0:
        logger.error("No graphs could be created")
        return

    # Make predictions
    logger.info("Making predictions...")
    predictions, probabilities = predict_batch(model, graphs, device, args.batch_size)

    # Create results dataframe
    results = pd.DataFrame({
        'smiles': valid_smiles[:len(predictions)],
        'prediction': predictions,
        'probability': probabilities,
        'confidence': probabilities  # For binary, confidence is same as probability
    })

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Prediction Results:")
    logger.info("=" * 80)

    if args.verbose:
        for _, row in results.iterrows():
            logger.info(f"\nSMILES: {row['smiles']}")
            logger.info(f"  Prediction: {row['prediction']}")
            logger.info(f"  Probability: {row['probability']:.4f}")
    else:
        logger.info(f"Processed {len(results)} molecules")
        logger.info(f"Average probability: {results['probability'].mean():.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.csv':
            results.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            results.to_json(output_path, orient='records', indent=2)
        else:
            results.to_csv(output_path, index=False)

        logger.info(f"\nResults saved to: {output_path}")

    logger.info("\nPrediction completed successfully!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with Adaptive Spectral Rewiring model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--smiles",
        type=str,
        help="Single SMILES string to predict"
    )
    input_group.add_argument(
        "--smiles-file",
        type=str,
        help="File containing SMILES strings (CSV or TXT)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (CSV or JSON)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for prediction"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each molecule"
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.error(f"Prediction failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
