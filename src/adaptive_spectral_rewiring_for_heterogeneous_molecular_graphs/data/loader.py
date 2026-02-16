"""Data loading utilities for molecular datasets."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch_geometric.data
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree

# Monkey-patch torch.load to use weights_only=False for OGB compatibility with PyTorch 2.6
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that sets weights_only=False for OGB datasets."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ogb.graphproppred import PygGraphPropPredDataset


def load_molecular_dataset(
    dataset_name: str = "ogbg-molhiv",
    root: str = "./data",
    download: bool = True
) -> Tuple[PygGraphPropPredDataset, Dict]:
    """Load molecular dataset from OGB.

    Args:
        dataset_name: Name of the dataset (e.g., 'ogbg-molhiv', 'ogbg-molpcba').
        root: Root directory for dataset storage.
        download: Whether to download dataset if not present.

    Returns:
        Tuple of (dataset, split_idx) where split_idx contains train/val/test indices.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    try:
        dataset = PygGraphPropPredDataset(name=dataset_name, root=root)
        split_idx = dataset.get_idx_split()

        logging.info(f"Loaded {dataset_name} dataset:")
        logging.info(f"  Total graphs: {len(dataset)}")
        logging.info(f"  Train: {len(split_idx['train'])}")
        logging.info(f"  Val: {len(split_idx['valid'])}")
        logging.info(f"  Test: {len(split_idx['test'])}")
        logging.info(f"  Num features: {dataset.num_features}")
        logging.info(f"  Num tasks: {dataset.num_tasks}")

        return dataset, split_idx
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {e}")
        raise


def create_dataloaders(
    dataset: PygGraphPropPredDataset,
    split_idx: Dict,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch Geometric dataloaders for train/val/test splits.

    Args:
        dataset: PyG dataset object.
        split_idx: Dictionary with 'train', 'valid', 'test' indices.
        batch_size: Batch size for training.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        dataset[split_idx['train']],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset[split_idx['valid']],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset[split_idx['test']],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logging.info(f"Created dataloaders with batch_size={batch_size}")
    return train_loader, val_loader, test_loader


def compute_graph_statistics(dataset: PygGraphPropPredDataset) -> Dict[str, float]:
    """Compute statistics about the molecular graphs.

    Args:
        dataset: PyG dataset object.

    Returns:
        Dictionary containing graph statistics.
    """
    num_nodes_list = []
    num_edges_list = []
    avg_degree_list = []

    for data in dataset:
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]

        num_nodes_list.append(num_nodes)
        num_edges_list.append(num_edges)

        # Compute average degree
        deg = degree(data.edge_index[0], num_nodes=num_nodes)
        avg_degree_list.append(deg.mean().item())

    stats = {
        'avg_num_nodes': sum(num_nodes_list) / len(num_nodes_list),
        'avg_num_edges': sum(num_edges_list) / len(num_edges_list),
        'avg_degree': sum(avg_degree_list) / len(avg_degree_list),
        'max_num_nodes': max(num_nodes_list),
        'max_num_edges': max(num_edges_list)
    }

    logging.info("Graph statistics:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value:.2f}")

    return stats
