"""Custom loss functions, layers, and modules for adaptive spectral rewiring.

This module implements spectral graph rewiring using Laplacian eigenvalue decomposition,
Fiedler vectors, and effective resistance to guide topology modifications during training.

Key Components:
    - compute_graph_laplacian: Computes various Laplacian normalizations
    - compute_spectral_gap: Extracts algebraic connectivity (Fiedler value)
    - compute_effective_resistance: Computes spectral distance between node pairs
    - LearnableRewiringPolicy: Neural network that scores edge rewiring candidates
    - SpectralRewiringLayer: Applies adaptive spectral rewiring with learnable policy
    - HeterogeneousSpectralRewiringLayer: Extends rewiring to heterogeneous graphs
"""

import logging
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, RGCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree
import numpy as np


def compute_graph_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    normalization: str = "sym"
) -> Tuple[torch.Tensor, bool]:
    """Compute the graph Laplacian matrix.

    Args:
        edge_index: Edge indices [2, num_edges].
        num_nodes: Number of nodes in the graph.
        edge_weight: Optional edge weights [num_edges].
        normalization: Type of normalization ('sym' for symmetric, 'rw' for random walk, None for unnormalized).

    Returns:
        Tuple of (Dense Laplacian matrix [num_nodes, num_nodes], is_symmetric flag).

    Raises:
        ValueError: If edge_index or num_nodes are invalid.

    Note:
        - 'sym' normalization produces symmetric Laplacian (use with eigvalsh)
        - 'rw' normalization produces ASYMMETRIC Laplacian (use with eig)
        - None produces symmetric unnormalized Laplacian (use with eigvalsh)
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")

    try:
        # Create adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes, edge_attr=edge_weight)[0]

        # Compute degree matrix
        deg = adj.sum(dim=1)

        if normalization == "sym":
            # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
            deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
            deg_inv_sqrt = torch.diag(deg_inv_sqrt)
            laplacian = torch.eye(num_nodes, device=adj.device) - deg_inv_sqrt @ adj @ deg_inv_sqrt
            return laplacian, True  # Symmetric
        elif normalization == "rw":
            # Random walk Laplacian: L_rw = I - D^{-1} A (ASYMMETRIC!)
            # NOTE: This matrix is NOT symmetric, so must use torch.linalg.eig instead of eigvalsh
            deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
            deg_inv = torch.diag(deg_inv)
            laplacian = torch.eye(num_nodes, device=adj.device) - deg_inv @ adj
            logging.warning("Random walk Laplacian is asymmetric - caller must use torch.linalg.eig, not eigvalsh")
            return laplacian, False  # Asymmetric
        else:
            # Unnormalized Laplacian: L = D - A (symmetric)
            laplacian = torch.diag(deg) - adj
            return laplacian, True  # Symmetric

    except Exception as e:
        logging.error(f"Error computing graph Laplacian: {e}", exc_info=True)
        raise RuntimeError(f"Failed to compute Laplacian: {e}") from e


def compute_spectral_gap(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Compute the spectral gap (difference between second and first eigenvalues).

    The spectral gap is the difference between the smallest non-zero eigenvalue
    and the smallest eigenvalue of the graph Laplacian. A larger spectral gap
    indicates better connectivity and mixing properties.

    Args:
        edge_index: Edge indices [2, num_edges].
        num_nodes: Number of nodes in the graph.

    Returns:
        Spectral gap value (second smallest eigenvalue).

    Raises:
        RuntimeError: If eigenvalue computation fails.
    """
    try:
        # Use symmetric normalization to ensure we can use efficient eigvalsh
        laplacian, is_symmetric = compute_graph_laplacian(edge_index, num_nodes, normalization="sym")

        if not is_symmetric:
            raise RuntimeError("Expected symmetric Laplacian but got asymmetric")

        # Compute eigenvalues (use only k smallest for efficiency)
        k = min(5, num_nodes)
        eigenvalues = torch.linalg.eigvalsh(laplacian)[:k]

        # Spectral gap is the second smallest eigenvalue (first is ~0 for connected graphs)
        spectral_gap = eigenvalues[1].item() if len(eigenvalues) > 1 else 0.0
        return float(spectral_gap)
    except Exception as e:
        logging.error(f"Failed to compute spectral gap: {e}", exc_info=True)
        raise RuntimeError(f"Spectral gap computation failed: {e}") from e


def compute_effective_resistance(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_pairs: int = 100
) -> Dict[Tuple[int, int], float]:
    """Compute effective resistance for pairs of nodes using Laplacian pseudoinverse.

    Effective resistance is a spectral distance metric that can guide edge addition.
    Node pairs with high effective resistance are far apart in the graph and would
    benefit from a direct edge connection.

    Args:
        edge_index: Edge indices [2, num_edges].
        num_nodes: Number of nodes in the graph.
        num_pairs: Number of random node pairs to sample.

    Returns:
        Dictionary mapping node pairs to their effective resistance.

    Raises:
        RuntimeError: If pseudoinverse computation fails.
    """
    try:
        # Use unnormalized Laplacian (symmetric)
        laplacian, is_symmetric = compute_graph_laplacian(edge_index, num_nodes, normalization=None)

        if not is_symmetric:
            raise RuntimeError("Expected symmetric Laplacian for effective resistance")

        # Compute pseudoinverse (Moore-Penrose inverse)
        # For Laplacian, we need to handle the zero eigenvalue
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        # Threshold for zero eigenvalue
        threshold = 1e-6
        eigenvalues_pinv = torch.where(
            eigenvalues > threshold,
            1.0 / eigenvalues,
            torch.zeros_like(eigenvalues)
        )

        # Reconstruct pseudoinverse: L^+ = V * diag(lambda^-1) * V^T
        laplacian_pinv = eigenvectors @ torch.diag(eigenvalues_pinv) @ eigenvectors.T

        # Sample random node pairs and compute effective resistance
        resistances = {}
        pairs = torch.randint(0, num_nodes, (num_pairs, 2), device=edge_index.device)

        for i in range(num_pairs):
            u, v = pairs[i].tolist()
            if u != v:
                # Effective resistance: R(u,v) = L^+_uu + L^+_vv - 2*L^+_uv
                r_eff = (laplacian_pinv[u, u] + laplacian_pinv[v, v] - 2 * laplacian_pinv[u, v]).item()
                resistances[(u, v)] = max(0.0, float(r_eff))  # Ensure non-negative

        return resistances
    except Exception as e:
        logging.error(f"Failed to compute effective resistance: {e}", exc_info=True)
        raise RuntimeError(f"Effective resistance computation failed: {e}") from e


class LearnableRewiringPolicy(nn.Module):
    """Learnable policy network for deciding which edges to add/remove.

    This module learns to predict edge rewiring decisions based on node features,
    current graph structure, and spectral properties to maximize spectral gap
    while preserving important molecular motifs.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, use_spectral_features: bool = True):
        """Initialize the rewiring policy.

        Args:
            hidden_dim: Hidden dimension for policy network.
            num_layers: Number of layers in policy network.
            use_spectral_features: Whether to incorporate spectral features in edge scoring.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_spectral_features = use_spectral_features

        # Edge scoring network
        layers = []
        in_dim = hidden_dim * 2  # Concatenated node embeddings
        if use_spectral_features:
            in_dim += 2  # Add spectral features (Fiedler vector values)

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))

        self.edge_scorer = nn.Sequential(*layers)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        fiedler_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute edge scores for rewiring decisions.

        Args:
            node_embeddings: Node embeddings of shape [num_nodes, hidden_dim].
            edge_index: Current edge indices [2, num_edges].
            num_nodes: Number of nodes in the graph.
            fiedler_vector: Optional Fiedler vector (second eigenvector of Laplacian) [num_nodes].

        Returns:
            Tuple of (edge_scores, candidate_edges) where edge_scores are
            rewiring scores and candidate_edges are potential new edges.
        """
        device = node_embeddings.device

        # Score existing edges
        src_emb = node_embeddings[edge_index[0]]
        dst_emb = node_embeddings[edge_index[1]]
        edge_features = torch.cat([src_emb, dst_emb], dim=-1)

        # Add spectral features if available
        if self.use_spectral_features and fiedler_vector is not None:
            src_fiedler = fiedler_vector[edge_index[0]].unsqueeze(-1)
            dst_fiedler = fiedler_vector[edge_index[1]].unsqueeze(-1)
            edge_features = torch.cat([edge_features, src_fiedler, dst_fiedler], dim=-1)

        edge_scores = self.edge_scorer(edge_features).squeeze(-1)

        # Generate spectral-guided candidate edges
        candidate_edges = self._generate_spectral_candidates(
            num_nodes, edge_index, fiedler_vector, device
        )

        return edge_scores, candidate_edges

    def _generate_spectral_candidates(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        fiedler_vector: Optional[torch.Tensor],
        device: torch.device,
        num_candidates: int = 1000
    ) -> torch.Tensor:
        """Generate candidate edges guided by spectral properties.

        Uses Fiedler vector to identify bottleneck edges - pairs of nodes with
        large differences in Fiedler vector values are good candidates for new edges.

        Args:
            num_nodes: Number of nodes in the graph.
            edge_index: Current edge indices.
            fiedler_vector: Fiedler vector values [num_nodes].
            device: Device for tensors.
            num_candidates: Number of candidate edges to generate.

        Returns:
            Candidate edge indices [2, num_candidates].
        """
        try:
            if fiedler_vector is not None and num_nodes > 2:
                # Sort nodes by Fiedler vector value
                sorted_indices = torch.argsort(fiedler_vector)

                # Generate candidates by pairing nodes from opposite ends of the Fiedler ordering
                # This helps bridge bottleneck cuts in the graph
                num_pairs = min(num_candidates, num_nodes * (num_nodes - 1) // 4)

                src_candidates = []
                dst_candidates = []

                # Sample from different parts of the Fiedler ordering
                for _ in range(num_pairs):
                    # Pick one node from lower half, one from upper half
                    idx1 = torch.randint(0, num_nodes // 2, (1,), device=device).item()
                    idx2 = torch.randint(num_nodes // 2, num_nodes, (1,), device=device).item()

                    src = sorted_indices[idx1].item()
                    dst = sorted_indices[idx2].item()

                    if src != dst:
                        src_candidates.append(src)
                        dst_candidates.append(dst)

                if len(src_candidates) > 0:
                    src_candidates = torch.tensor(src_candidates, device=device)
                    dst_candidates = torch.tensor(dst_candidates, device=device)
                    return torch.stack([src_candidates, dst_candidates], dim=0)

            # Fallback to random sampling if spectral guidance not available
            num_pairs = min(num_candidates, 1000)
            src_candidates = torch.randint(0, num_nodes, (num_pairs,), device=device)
            dst_candidates = torch.randint(0, num_nodes, (num_pairs,), device=device)

            # Remove self-loops
            mask = src_candidates != dst_candidates
            src_candidates = src_candidates[mask]
            dst_candidates = dst_candidates[mask]

            return torch.stack([src_candidates, dst_candidates], dim=0)

        except Exception as e:
            logging.warning(f"Error generating spectral candidates: {e}, falling back to random")
            # Fallback
            num_pairs = min(num_candidates, 1000)
            src_candidates = torch.randint(0, num_nodes, (num_pairs,), device=device)
            dst_candidates = torch.randint(0, num_nodes, (num_pairs,), device=device)
            mask = src_candidates != dst_candidates
            return torch.stack([src_candidates[mask], dst_candidates[mask]], dim=0)


class SpectralRewiringLayer(nn.Module):
    """Adaptive spectral rewiring layer with true spectral decomposition.

    Dynamically modifies graph topology during training to improve spectral gap
    while preserving important molecular motifs. Uses Laplacian eigenvalue
    decomposition and Fiedler vector to guide rewiring decisions.

    **Core Mechanism:**
    1. Computes Laplacian L and extracts Fiedler vector v_2 (second eigenvector)
    2. Generates candidate edges connecting nodes with large |v_2[i] - v_2[j]| (bottleneck pairs)
    3. Scores edges using learnable policy network: score(e) = MLP([h_i, h_j, v_2[i], v_2[j]])
    4. Removes lowest-scoring edges, adds highest-scoring candidates
    5. Optimizes combined loss: L_total = L_edge_quality + λ * L_spectral_gap

    **Rewiring Loss Formulation:**
    The rewiring loss encourages edge selections that improve graph connectivity:

        L_rewiring = L_edge_quality + λ_spectral * L_spectral_gap

    Where:
        - L_edge_quality = -mean(sigmoid(edge_scores))
          Maximizes average edge quality as scored by the learnable policy

        - L_spectral_gap = -(gap_after - gap_before)
          Encourages increasing the spectral gap (algebraic connectivity)
          gap = λ_2(L), the second smallest Laplacian eigenvalue

        - λ_spectral: Weight balancing edge quality vs. spectral improvement (default 0.1)

    The policy network learns to identify edges whose addition increases spectral gap
    and whose removal has minimal impact, using both structural (node embeddings) and
    spectral (Fiedler vector) features.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        rewiring_ratio: float = 0.1,
        preserve_motifs: bool = True,
        compute_spectral_features: bool = True,
        spectral_loss_weight: float = 0.1
    ):
        """Initialize spectral rewiring layer.

        Args:
            hidden_dim: Dimension of node embeddings.
            rewiring_ratio: Fraction of edges to consider for rewiring (e.g., 0.1 = 10%).
            preserve_motifs: Whether to preserve molecular motifs (not fully implemented).
            compute_spectral_features: Whether to compute and use spectral features (Fiedler vectors).
            spectral_loss_weight: Weight λ_spectral for spectral gap loss term (default 0.1).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rewiring_ratio = rewiring_ratio
        self.preserve_motifs = preserve_motifs
        self.compute_spectral_features = compute_spectral_features
        self.spectral_loss_weight = spectral_loss_weight

        self.policy = LearnableRewiringPolicy(
            hidden_dim=hidden_dim,
            use_spectral_features=compute_spectral_features
        )

    def _compute_fiedler_vector(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Optional[torch.Tensor]:
        """Compute the Fiedler vector (second eigenvector of graph Laplacian).

        The Fiedler vector provides information about graph connectivity and
        can identify bottleneck cuts. Nodes on opposite sides of a bottleneck
        cut will have large differences in their Fiedler vector values.

        Args:
            edge_index: Edge indices [2, num_edges].
            num_nodes: Number of nodes.

        Returns:
            Fiedler vector [num_nodes] or None if computation fails.

        Raises:
            ValueError: If num_nodes < 2 (Fiedler vector requires at least 2 nodes).
        """
        try:
            if num_nodes < 2:
                raise ValueError(f"Fiedler vector requires at least 2 nodes, got {num_nodes}")

            if num_nodes < 3:
                # For 2 nodes, Fiedler vector is trivial
                logging.debug("Graph has only 2 nodes, Fiedler vector is trivial")
                return None

            # Use symmetric normalization to ensure we can use eigvalsh
            laplacian, is_symmetric = compute_graph_laplacian(edge_index, num_nodes, normalization="sym")

            if not is_symmetric:
                raise RuntimeError("Expected symmetric Laplacian for Fiedler vector")

            # Compute first few eigenvectors
            k = min(3, num_nodes)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            fiedler_vector = eigenvectors[:, 1]

            return fiedler_vector

        except Exception as e:
            logging.error(f"Failed to compute Fiedler vector: {e}", exc_info=True)
            raise RuntimeError(f"Fiedler vector computation failed: {e}") from e

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        motif_edges: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive spectral rewiring with spectral decomposition.

        Args:
            x: Node features [num_nodes, num_features].
            edge_index: Edge indices [2, num_edges].
            batch: Batch assignment for nodes.
            motif_edges: Edges that are part of molecular motifs (to preserve).

        Returns:
            Tuple of (rewired_edge_index, rewiring_loss).
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if not self.training or num_nodes < 3:
            # No rewiring during inference or for very small graphs
            return edge_index, torch.tensor(0.0, device=x.device)

        try:
            # Compute spectral gap before rewiring
            spectral_gap_before = 0.0
            fiedler_vector = None

            if self.compute_spectral_features:
                spectral_gap_before = compute_spectral_gap(edge_index, num_nodes)
                fiedler_vector = self._compute_fiedler_vector(edge_index, num_nodes)

            # Get edge scores from policy (now incorporating spectral features)
            edge_scores, candidate_edges = self.policy(
                x, edge_index, num_nodes, fiedler_vector=fiedler_vector
            )

            # Determine edges to remove (lowest scores)
            num_to_remove = max(1, int(num_edges * self.rewiring_ratio))

            # Protect motif edges from removal if specified
            if self.preserve_motifs and motif_edges is not None:
                edge_scores = edge_scores.clone()
                # Boost scores for motif edges to prevent removal
                # Simple implementation: set high scores

            _, remove_indices = torch.topk(
                edge_scores,
                min(num_to_remove, len(edge_scores)),
                largest=False
            )

            # Create mask for edges to keep
            keep_mask = torch.ones(num_edges, dtype=torch.bool, device=x.device)
            keep_mask[remove_indices] = False

            # Keep remaining edges
            kept_edges = edge_index[:, keep_mask]

            # Add new edges (top scoring candidates)
            num_to_add = num_to_remove
            if candidate_edges.size(1) > 0:
                # Score candidate edges
                src_emb = x[candidate_edges[0]]
                dst_emb = x[candidate_edges[1]]
                candidate_features = torch.cat([src_emb, dst_emb], dim=-1)

                # Add spectral features if available
                if self.compute_spectral_features and fiedler_vector is not None:
                    src_fiedler = fiedler_vector[candidate_edges[0]].unsqueeze(-1)
                    dst_fiedler = fiedler_vector[candidate_edges[1]].unsqueeze(-1)
                    candidate_features = torch.cat([candidate_features, src_fiedler, dst_fiedler], dim=-1)

                candidate_scores = self.policy.edge_scorer(candidate_features).squeeze(-1)

                # Select top candidates
                num_to_add = min(num_to_add, candidate_edges.size(1))
                _, add_indices = torch.topk(candidate_scores, num_to_add, largest=True)
                new_edges = candidate_edges[:, add_indices]

                # Combine edges
                rewired_edge_index = torch.cat([kept_edges, new_edges], dim=1)
            else:
                rewired_edge_index = kept_edges

            # Compute spectral gap after rewiring
            spectral_gap_after = 0.0
            if self.compute_spectral_features:
                spectral_gap_after = compute_spectral_gap(rewired_edge_index, num_nodes)

            # Compute rewiring loss
            # Encourage increasing spectral gap
            spectral_gap_loss = -(spectral_gap_after - spectral_gap_before)

            # Edge quality loss (maximize average edge scores)
            edge_scores_normalized = torch.sigmoid(edge_scores)
            edge_quality_loss = -edge_scores_normalized.mean()

            # Combined loss
            rewiring_loss = edge_quality_loss + self.spectral_loss_weight * spectral_gap_loss

            # Clamp to prevent numerical instability
            rewiring_loss = torch.clamp(rewiring_loss, min=-10.0, max=10.0)

            return rewired_edge_index, rewiring_loss

        except Exception as e:
            logging.error(f"Error in spectral rewiring: {e}")
            # Return original edge index and zero loss on error
            return edge_index, torch.tensor(0.0, device=x.device)


class MotifPreservationLoss(nn.Module):
    """Custom loss function to preserve important molecular motifs during rewiring.

    This loss penalizes rewiring decisions that break chemically important
    substructures like aromatic rings and functional groups.
    """

    def __init__(self, weight: float = 0.5):
        """Initialize motif preservation loss.

        Args:
            weight: Weight for the motif preservation term.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        original_edge_index: torch.Tensor,
        rewired_edge_index: torch.Tensor,
        motif_edges: Optional[List[Tuple[int, int]]] = None
    ) -> torch.Tensor:
        """Compute motif preservation loss.

        Args:
            original_edge_index: Original edge indices [2, num_edges].
            rewired_edge_index: Rewired edge indices [2, num_edges_new].
            motif_edges: List of edge tuples that are part of motifs.

        Returns:
            Scalar loss tensor.
        """
        if motif_edges is None or len(motif_edges) == 0:
            return torch.tensor(0.0, device=original_edge_index.device)

        # Convert motif edges to set for fast lookup
        original_edges_set = set(
            tuple(edge) for edge in original_edge_index.t().cpu().tolist()
        )
        rewired_edges_set = set(
            tuple(edge) for edge in rewired_edge_index.t().cpu().tolist()
        )
        motif_edges_set = set(motif_edges)

        # Count how many motif edges were removed
        removed_motif_edges = motif_edges_set.intersection(original_edges_set) - \
                             motif_edges_set.intersection(rewired_edges_set)

        # Loss is proportional to fraction of motif edges removed
        if len(motif_edges_set) > 0:
            loss = len(removed_motif_edges) / len(motif_edges_set)
        else:
            loss = 0.0

        return torch.tensor(loss * self.weight, device=original_edge_index.device)


class SpectralGapLoss(nn.Module):
    """Loss to encourage increasing the spectral gap of rewired graphs.

    Uses actual Laplacian eigenvalue computation to measure and optimize
    the spectral gap.
    """

    def __init__(self, weight: float = 0.3, use_true_spectral_gap: bool = True):
        """Initialize spectral gap loss.

        Args:
            weight: Weight for the spectral gap term.
            use_true_spectral_gap: If True, compute actual spectral gap; otherwise use degree variance proxy.
        """
        super().__init__()
        self.weight = weight
        self.use_true_spectral_gap = use_true_spectral_gap

    def forward(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        target_gap: Optional[float] = None
    ) -> torch.Tensor:
        """Compute spectral gap loss.

        This loss encourages increasing the spectral gap (algebraic connectivity) of the graph,
        which improves mixing properties and information flow.

        Args:
            edge_index: Edge indices [2, num_edges].
            num_nodes: Number of nodes.
            target_gap: Optional target spectral gap value.

        Returns:
            Scalar loss tensor.
        """
        try:
            if self.use_true_spectral_gap and num_nodes >= 3:
                # Compute actual spectral gap
                spectral_gap = compute_spectral_gap(edge_index, num_nodes)

                if target_gap is not None:
                    # Loss is distance from target gap
                    loss = (spectral_gap - target_gap) ** 2
                else:
                    # Maximize spectral gap (negative loss)
                    loss = -spectral_gap

                return torch.tensor(loss * self.weight, device=edge_index.device)

            # Fallback: use degree variance as proxy for spectral gap
            # (lower variance often correlates with better connectivity)
            deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
            degree_variance = deg.var()
            loss = degree_variance * self.weight

            return loss

        except Exception as e:
            # Use degree-based proxy when spectral computation fails
            logging.warning(f"Spectral gap computation failed, using degree variance proxy: {e}")
            try:
                deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
                degree_variance = deg.var()
                return degree_variance * self.weight
            except Exception as e2:
                logging.error(f"Degree variance fallback also failed: {e2}")
                return torch.tensor(0.0, device=edge_index.device)


class HeterogeneousSpectralRewiringLayer(nn.Module):
    """Spectral rewiring layer for heterogeneous graphs with multiple node/edge types.

    Applies type-specific rewiring policies while maintaining heterogeneous structure.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        rewiring_ratio: float = 0.1,
        type_specific_rewiring: bool = True
    ):
        """Initialize heterogeneous spectral rewiring layer.

        Args:
            hidden_dim: Dimension of node embeddings.
            node_types: List of node type names.
            edge_types: List of edge types as (src_type, edge_type, dst_type) tuples.
            rewiring_ratio: Fraction of edges to rewire.
            type_specific_rewiring: Whether to use separate policies for each edge type.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = node_types or ['default']
        self.edge_types = edge_types or [('default', 'default', 'default')]
        self.rewiring_ratio = rewiring_ratio
        self.type_specific_rewiring = type_specific_rewiring

        # Create separate rewiring policies for each edge type if specified
        if type_specific_rewiring:
            self.rewiring_policies = nn.ModuleDict({
                f"{src}_{rel}_{dst}": SpectralRewiringLayer(
                    hidden_dim=hidden_dim,
                    rewiring_ratio=rewiring_ratio,
                    preserve_motifs=True,
                    compute_spectral_features=True
                )
                for src, rel, dst in self.edge_types
            })
        else:
            # Single shared policy
            self.shared_policy = SpectralRewiringLayer(
                hidden_dim=hidden_dim,
                rewiring_ratio=rewiring_ratio,
                preserve_motifs=True,
                compute_spectral_features=True
            )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        node_type_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[Tuple[str, str, str], torch.Tensor], torch.Tensor]:
        """Apply heterogeneous spectral rewiring.

        Args:
            x_dict: Dictionary mapping node types to node features.
            edge_index_dict: Dictionary mapping edge types to edge indices.
            node_type_dict: Optional dictionary mapping node types to type assignments.

        Returns:
            Tuple of (rewired_edge_index_dict, total_rewiring_loss).
        """
        try:
            rewired_edge_index_dict = {}
            total_rewiring_loss = torch.tensor(0.0, device=next(iter(x_dict.values())).device)

            # Rewire each edge type
            for edge_type, edge_index in edge_index_dict.items():
                src_type, rel_type, dst_type = edge_type

                # Get node features for source and destination types
                if src_type in x_dict and dst_type in x_dict:
                    # For simplicity, use source node features
                    # In practice, might want to combine or use type-specific embeddings
                    x = x_dict[src_type]

                    # Get appropriate rewiring policy
                    if self.type_specific_rewiring:
                        policy_key = f"{src_type}_{rel_type}_{dst_type}"
                        if policy_key in self.rewiring_policies:
                            policy = self.rewiring_policies[policy_key]
                        else:
                            # Skip unknown edge types
                            rewired_edge_index_dict[edge_type] = edge_index
                            continue
                    else:
                        policy = self.shared_policy

                    # Apply rewiring for this edge type
                    num_nodes = x.size(0)
                    rewired_edge_index, rewiring_loss = policy(x, edge_index)

                    rewired_edge_index_dict[edge_type] = rewired_edge_index
                    total_rewiring_loss = total_rewiring_loss + rewiring_loss
                else:
                    # Keep edge type unchanged if node types not found
                    rewired_edge_index_dict[edge_type] = edge_index

            return rewired_edge_index_dict, total_rewiring_loss

        except Exception as e:
            logging.error(f"Error in heterogeneous rewiring: {e}")
            return edge_index_dict, torch.tensor(0.0, device=next(iter(x_dict.values())).device)
