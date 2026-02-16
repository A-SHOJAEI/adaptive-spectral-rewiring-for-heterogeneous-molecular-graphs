"""Core model implementation for adaptive spectral GNN."""

import logging
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    SAGEConv,
    RGCNConv,
    HeteroConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

from .components import SpectralRewiringLayer, HeterogeneousSpectralRewiringLayer


class AdaptiveSpectralGNN(nn.Module):
    """Graph Neural Network with adaptive spectral rewiring.

    This model combines graph convolutions with dynamic topology modification
    to address over-squashing in molecular graphs. The rewiring policy learns
    to add/remove edges to improve information flow while preserving molecular motifs.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.5,
        rewiring_layers: Optional[list] = None,
        rewiring_ratio: float = 0.1,
        pooling: str = "mean"
    ):
        """Initialize the Adaptive Spectral GNN.

        Args:
            num_features: Number of input node features.
            num_classes: Number of output classes (tasks).
            hidden_dim: Hidden dimension for GNN layers.
            num_layers: Number of GNN layers.
            dropout: Dropout rate.
            rewiring_layers: List of layer indices where rewiring is applied.
                            If None, applies to all layers.
            rewiring_ratio: Fraction of edges to rewire per layer.
            pooling: Graph pooling method ('mean' or 'add').
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # Determine which layers use rewiring
        if rewiring_layers is None:
            self.rewiring_layers = list(range(num_layers))
        else:
            self.rewiring_layers = rewiring_layers

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # GNN layers with adaptive rewiring
        self.convs = nn.ModuleList()
        self.rewiring_modules = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # GNN convolution
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

            # Batch normalization
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # Rewiring module for selected layers
            if i in self.rewiring_layers:
                self.rewiring_modules.append(
                    SpectralRewiringLayer(
                        hidden_dim=hidden_dim,
                        rewiring_ratio=rewiring_ratio,
                        preserve_motifs=True
                    )
                )
            else:
                self.rewiring_modules.append(None)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

        # Pooling function
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        logging.info(f"Initialized AdaptiveSpectralGNN with {num_layers} layers")
        logging.info(f"Rewiring applied at layers: {self.rewiring_layers}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_rewiring_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Node features [num_nodes, num_features].
            edge_index: Edge indices [2, num_edges].
            batch: Batch assignment for nodes [num_nodes].
            return_rewiring_loss: Whether to return rewiring auxiliary loss.

        Returns:
            Tuple of (output logits, rewiring_loss).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Convert input to float if needed
        if x.dtype != torch.float:
            x = x.float()

        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Accumulate rewiring losses
        total_rewiring_loss = torch.tensor(0.0, device=x.device)

        # Apply GNN layers with adaptive rewiring
        current_edge_index = edge_index

        for i in range(self.num_layers):
            # Apply rewiring if configured for this layer
            if self.rewiring_modules[i] is not None:
                current_edge_index, rewiring_loss = self.rewiring_modules[i](
                    h, current_edge_index, batch=batch
                )
                total_rewiring_loss = total_rewiring_loss + rewiring_loss

            # GNN convolution
            h = self.convs[i](h, current_edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Graph-level pooling
        graph_embedding = self.pool(h, batch)

        # Output prediction
        out = self.fc1(graph_embedding)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)

        if return_rewiring_loss:
            return out, total_rewiring_loss
        else:
            return out, torch.tensor(0.0, device=x.device)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters."""
        self.input_proj.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.batch_norms:
            bn.reset_parameters()

        for rewiring_module in self.rewiring_modules:
            if rewiring_module is not None:
                for module in rewiring_module.modules():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class BaselineGNN(nn.Module):
    """Baseline GNN without adaptive rewiring for comparison."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.5,
        pooling: str = "mean"
    ):
        """Initialize baseline GNN.

        Args:
            num_features: Number of input node features.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension.
            num_layers: Number of GNN layers.
            dropout: Dropout rate.
            pooling: Pooling method.
        """
        super().__init__()

        self.input_proj = nn.Linear(num_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout = dropout

        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment.

        Returns:
            Output logits.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for i in range(len(self.convs)):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_embedding = self.pool(h, batch)

        out = self.fc1(graph_embedding)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)

        return out
