"""Heterogeneous graph neural network with adaptive spectral rewiring."""

import logging
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    HeteroConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

from .components import HeterogeneousSpectralRewiringLayer


class HeterogeneousAdaptiveSpectralGNN(nn.Module):
    """Heterogeneous Graph Neural Network with adaptive spectral rewiring.

    Supports multiple node types and edge types with type-specific message passing
    and adaptive rewiring policies.
    """

    def __init__(
        self,
        num_features_dict: Dict[str, int],
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.5,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        rewiring_ratio: float = 0.1,
        pooling: str = "mean",
        use_rewiring: bool = True
    ):
        """Initialize the Heterogeneous Adaptive Spectral GNN.

        Args:
            num_features_dict: Dictionary mapping node types to their feature dimensions.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension for GNN layers.
            num_layers: Number of GNN layers.
            dropout: Dropout rate.
            node_types: List of node type names.
            edge_types: List of edge types as (src_type, edge_type, dst_type) tuples.
            rewiring_ratio: Fraction of edges to rewire.
            pooling: Graph pooling method.
            use_rewiring: Whether to use adaptive rewiring.
        """
        super().__init__()

        self.num_features_dict = num_features_dict
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_types = node_types or list(num_features_dict.keys())
        self.edge_types = edge_types or []
        self.use_rewiring = use_rewiring
        self.pooling = pooling

        # Input projections for each node type
        self.input_projs = nn.ModuleDict({
            node_type: nn.Linear(num_features, hidden_dim)
            for node_type, num_features in num_features_dict.items()
        })

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleDict()

        for i in range(num_layers):
            # Create HeteroConv with SAGEConv for each edge type (works better with hetero graphs)
            conv_dict = {}
            for src_type, edge_type, dst_type in self.edge_types:
                edge_key = (src_type, edge_type, dst_type)
                # SAGEConv doesn't add self-loops, works better with heterogeneous graphs
                conv_dict[edge_key] = SAGEConv(hidden_dim, hidden_dim)

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

            # Batch normalization for each node type
            bn_dict = {}
            for node_type in self.node_types:
                bn_dict[node_type] = nn.BatchNorm1d(hidden_dim)
            self.batch_norms[f"layer_{i}"] = nn.ModuleDict(bn_dict)

        # Heterogeneous rewiring if enabled
        if use_rewiring:
            self.rewiring_module = HeterogeneousSpectralRewiringLayer(
                hidden_dim=hidden_dim,
                node_types=self.node_types,
                edge_types=self.edge_types,
                rewiring_ratio=rewiring_ratio,
                type_specific_rewiring=True
            )
        else:
            self.rewiring_module = None

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

        # Pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        logging.info(f"Initialized HeterogeneousAdaptiveSpectralGNN with {num_layers} layers")
        logging.info(f"Node types: {self.node_types}")
        logging.info(f"Edge types: {len(self.edge_types)}")

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_rewiring_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through heterogeneous model.

        Args:
            x_dict: Dictionary mapping node types to node features.
            edge_index_dict: Dictionary mapping edge types to edge indices.
            batch_dict: Dictionary mapping node types to batch assignments.
            return_rewiring_loss: Whether to return rewiring loss.

        Returns:
            Tuple of (output logits, rewiring_loss).
        """
        try:
            # Initialize batch assignments if not provided
            if batch_dict is None:
                batch_dict = {
                    node_type: torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                    for node_type, x in x_dict.items()
                }

            # Input projection for each node type
            h_dict = {}
            for node_type, x in x_dict.items():
                if x.dtype != torch.float:
                    x = x.float()
                h = self.input_projs[node_type](x)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h_dict[node_type] = h

            # Accumulate rewiring loss
            total_rewiring_loss = torch.tensor(0.0, device=next(iter(x_dict.values())).device)

            # Apply GNN layers with optional rewiring
            current_edge_index_dict = edge_index_dict

            for i in range(self.num_layers):
                # Apply rewiring if enabled
                if self.rewiring_module is not None and self.training:
                    current_edge_index_dict, rewiring_loss = self.rewiring_module(
                        h_dict, current_edge_index_dict
                    )
                    total_rewiring_loss = total_rewiring_loss + rewiring_loss

                # Heterogeneous convolution
                h_dict = self.convs[i](h_dict, current_edge_index_dict)

                # Batch normalization and activation for each node type
                for node_type in h_dict.keys():
                    if node_type in self.batch_norms[f"layer_{i}"]:
                        h_dict[node_type] = self.batch_norms[f"layer_{i}"][node_type](h_dict[node_type])
                        h_dict[node_type] = F.relu(h_dict[node_type])
                        h_dict[node_type] = F.dropout(h_dict[node_type], p=self.dropout, training=self.training)

            # Graph-level pooling (aggregate all node types)
            graph_embeddings = []
            for node_type, h in h_dict.items():
                if node_type in batch_dict:
                    pooled = self.pool(h, batch_dict[node_type])
                    graph_embeddings.append(pooled)

            # Combine embeddings from all node types
            if len(graph_embeddings) > 0:
                graph_embedding = torch.stack(graph_embeddings).mean(dim=0)
            else:
                # Fallback
                device = next(iter(x_dict.values())).device
                graph_embedding = torch.zeros(1, self.hidden_dim, device=device)

            # Output prediction
            out = self.fc1(graph_embedding)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            out = self.fc2(out)

            if return_rewiring_loss:
                return out, total_rewiring_loss
            else:
                return out, torch.tensor(0.0, device=graph_embedding.device)

        except Exception as e:
            logging.error(f"Error in heterogeneous forward pass: {e}")
            raise
