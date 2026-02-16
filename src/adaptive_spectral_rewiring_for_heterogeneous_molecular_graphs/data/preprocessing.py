"""Preprocessing utilities for molecular graphs."""

import logging
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
from scipy.linalg import eigh
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_graph(smiles: str) -> Data:
    """Convert SMILES string to PyG Data object.

    Args:
        smiles: SMILES representation of molecule.

    Returns:
        PyG Data object with node features and edge indices.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
        ]
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Add reverse edge

    if len(edge_indices) == 0:
        # Molecule with no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()

    data = Data(x=x, edge_index=edge_index)
    return data


def compute_spectral_gap(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Compute the spectral gap of a graph.

    The spectral gap is the difference between the second smallest and smallest
    eigenvalues of the graph Laplacian. Larger gaps indicate better connectivity.

    Args:
        edge_index: Edge indices of shape [2, num_edges].
        num_nodes: Number of nodes in the graph.

    Returns:
        Spectral gap value.
    """
    # Build adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj.numpy()

    # Compute degree matrix
    deg = np.sum(adj, axis=1)
    deg_matrix = np.diag(deg)

    # Compute Laplacian
    laplacian = deg_matrix - adj

    # Add small regularization for stability
    laplacian += 1e-8 * np.eye(num_nodes)

    # Compute eigenvalues
    try:
        eigenvalues = eigh(laplacian, eigvals_only=True)
        eigenvalues = np.sort(eigenvalues)

        # Spectral gap is difference between 2nd and 1st eigenvalue
        if len(eigenvalues) >= 2:
            gap = eigenvalues[1] - eigenvalues[0]
        else:
            gap = 0.0

        return float(gap)
    except Exception as e:
        logging.warning(f"Error computing spectral gap: {e}")
        return 0.0


def add_spectral_features(data: Data) -> Data:
    """Add spectral features to graph data.

    Args:
        data: PyG Data object.

    Returns:
        Data object with added spectral features.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Ensure undirected edges
    edge_index = to_undirected(edge_index)

    # Compute spectral gap
    spectral_gap = compute_spectral_gap(edge_index, num_nodes)
    data.spectral_gap = torch.tensor([spectral_gap], dtype=torch.float)

    return data


def identify_molecular_motifs(data: Data) -> List[List[int]]:
    """Identify important molecular motifs (rings, functional groups).

    Args:
        data: PyG Data object representing a molecular graph.

    Returns:
        List of node sets representing identified motifs.
    """
    # Convert to NetworkX for cycle detection
    G = to_networkx(data, to_undirected=True)

    motifs = []

    # Find all cycles (rings)
    try:
        cycles = nx.cycle_basis(G)
        motifs.extend(cycles)
    except Exception as e:
        logging.debug(f"Error finding cycles: {e}")

    # Find triangles (3-member rings)
    triangles = [list(clique) for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
    motifs.extend(triangles)

    return motifs


def compute_graph_statistics(data: Data) -> dict:
    """Compute basic graph statistics.

    Args:
        data: PyG Data object.

    Returns:
        Dictionary containing graph statistics.
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_features': data.x.size(1) if data.x is not None else 0,
        'avg_degree': num_edges / num_nodes if num_nodes > 0 else 0.0
    }
    return stats


def normalize_features(data: Data) -> Data:
    """Normalize node features.

    Args:
        data: PyG Data object.

    Returns:
        Data object with normalized features.
    """
    if data.x is not None:
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        data.x = (data.x - mean) / std
    return data


def augment_graph(data: Data, method: str = 'drop_edges', ratio: float = 0.1) -> Data:
    """Apply graph augmentation.

    Args:
        data: PyG Data object.
        method: Augmentation method ('drop_edges' or 'drop_nodes').
        ratio: Fraction to drop.

    Returns:
        Augmented Data object.
    """
    import copy
    aug_data = copy.deepcopy(data)

    if method == 'drop_edges':
        num_edges = aug_data.edge_index.size(1)
        num_drop = int(num_edges * ratio)
        keep_idx = torch.randperm(num_edges)[:-num_drop]
        aug_data.edge_index = aug_data.edge_index[:, keep_idx]
    elif method == 'drop_nodes':
        num_nodes = aug_data.num_nodes
        num_drop = max(1, int(num_nodes * ratio))
        keep_nodes = torch.randperm(num_nodes)[:-num_drop]
        # Filter edges that involve dropped nodes
        mask = torch.isin(aug_data.edge_index[0], keep_nodes) & torch.isin(aug_data.edge_index[1], keep_nodes)
        aug_data.edge_index = aug_data.edge_index[:, mask]
        aug_data.x = aug_data.x[keep_nodes]

    return aug_data


def preprocess_molecular_graphs(
    dataset: List[Data],
    add_spectral: bool = True,
    normalize_features: bool = True
) -> List[Data]:
    """Preprocess molecular graphs with spectral features.

    Args:
        dataset: List of PyG Data objects.
        add_spectral: Whether to add spectral features.
        normalize_features: Whether to normalize node features.

    Returns:
        List of preprocessed Data objects.
    """
    preprocessed = []

    for i, data in enumerate(dataset):
        try:
            # Add spectral features
            if add_spectral:
                data = add_spectral_features(data)

            # Normalize node features
            if normalize_features and data.x is not None:
                mean = data.x.mean(dim=0, keepdim=True)
                std = data.x.std(dim=0, keepdim=True)
                std = torch.where(std == 0, torch.ones_like(std), std)
                data.x = (data.x - mean) / std

            # Store motifs as graph-level attribute
            data.motifs = identify_molecular_motifs(data)

            preprocessed.append(data)

            if (i + 1) % 1000 == 0:
                logging.info(f"Preprocessed {i + 1}/{len(dataset)} graphs")

        except Exception as e:
            logging.warning(f"Error preprocessing graph {i}: {e}")
            preprocessed.append(data)

    return preprocessed
