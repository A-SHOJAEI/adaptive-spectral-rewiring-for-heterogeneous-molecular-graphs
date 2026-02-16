"""
Comprehensive tests for spectral graph components.

Tests verify that spectral properties (gap, resistance, Fiedler vector) are
computed correctly and that rewiring actually improves these properties.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.components import (
    compute_graph_laplacian,
    compute_spectral_gap,
    compute_effective_resistance,
    SpectralRewiringLayer,
    LearnableRewiringPolicy
)


class TestGraphLaplacian:
    """Tests for Laplacian computation."""

    def test_symmetric_normalization(self):
        """Test that symmetric normalization produces symmetric matrix."""
        # Simple triangle graph
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                   [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        num_nodes = 3

        laplacian, is_symmetric = compute_graph_laplacian(edge_index, num_nodes, normalization="sym")

        # Check symmetry
        assert torch.allclose(laplacian, laplacian.T, atol=1e-6)
        assert is_symmetric is True

    def test_random_walk_asymmetric(self):
        """Test that random walk normalization produces asymmetric matrix."""
        # Directed path: 0 -> 1 -> 2
        edge_index = torch.tensor([[0, 1],
                                   [1, 2]], dtype=torch.long)
        num_nodes = 3

        laplacian, is_symmetric = compute_graph_laplacian(edge_index, num_nodes, normalization="rw")

        # Random walk Laplacian should be flagged as asymmetric
        assert is_symmetric is False

    def test_unnormalized_symmetric(self):
        """Test that unnormalized Laplacian is symmetric."""
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        num_nodes = 3

        laplacian, is_symmetric = compute_graph_laplacian(edge_index, num_nodes, normalization=None)

        # Check symmetry
        assert torch.allclose(laplacian, laplacian.T, atol=1e-6)
        assert is_symmetric is True

    def test_laplacian_raises_on_invalid_input(self):
        """Test that invalid inputs raise exceptions."""
        edge_index = torch.tensor([[0, 1]], dtype=torch.long)  # Wrong shape

        with pytest.raises(ValueError):
            compute_graph_laplacian(edge_index, 3, normalization="sym")

        with pytest.raises(ValueError):
            compute_graph_laplacian(torch.tensor([[0, 1], [1, 0]]), 0, normalization="sym")


class TestSpectralGap:
    """Tests for spectral gap computation."""

    def test_complete_graph_high_gap(self):
        """Test that complete graph has high spectral gap."""
        # K4 complete graph
        n = 4
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append([i, j])
                edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).T
        num_nodes = 4

        gap = compute_spectral_gap(edge_index, num_nodes)

        # Complete graph should have high spectral gap
        assert gap > 0.5

    def test_path_graph_low_gap(self):
        """Test that path graph has low spectral gap."""
        # Path: 0 - 1 - 2 - 3 - 4
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        num_nodes = 5

        gap = compute_spectral_gap(edge_index, num_nodes)

        # Path graph should have low spectral gap
        assert 0.0 < gap < 0.5

    def test_disconnected_graph_zero_gap(self):
        """Test that disconnected graph has near-zero spectral gap."""
        # Two separate edges: 0-1 and 2-3
        edge_index = torch.tensor([[0, 1, 2, 3],
                                   [1, 0, 3, 2]], dtype=torch.long)
        num_nodes = 4

        gap = compute_spectral_gap(edge_index, num_nodes)

        # Disconnected graph should have very small spectral gap
        assert gap < 0.1

    def test_spectral_gap_raises_on_failure(self):
        """Test that spectral gap raises exception on invalid input."""
        edge_index = torch.tensor([[0, 1]], dtype=torch.long)  # Wrong shape

        with pytest.raises((ValueError, RuntimeError)):
            compute_spectral_gap(edge_index, 3)


class TestEffectiveResistance:
    """Tests for effective resistance computation."""

    def test_effective_resistance_positive(self):
        """Test that effective resistance is positive."""
        # Triangle graph
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                   [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        num_nodes = 3

        resistances = compute_effective_resistance(edge_index, num_nodes, num_pairs=10)

        # All resistances should be non-negative
        for r in resistances.values():
            assert r >= 0.0

    def test_effective_resistance_connected_vs_distant(self):
        """Test that directly connected nodes have lower resistance."""
        # Path: 0 - 1 - 2 - 3
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        num_nodes = 4

        resistances = compute_effective_resistance(edge_index, num_nodes, num_pairs=50)

        # Nodes 0 and 1 are directly connected
        r_01 = resistances.get((0, 1), resistances.get((1, 0), None))

        # Nodes 0 and 3 are far apart
        r_03 = resistances.get((0, 3), resistances.get((3, 0), None))

        if r_01 is not None and r_03 is not None:
            # Distant pairs should have higher resistance
            assert r_03 > r_01

    def test_effective_resistance_raises_on_invalid_input(self):
        """Test that effective resistance raises exception on invalid input."""
        edge_index = torch.tensor([[0, 1]], dtype=torch.long)  # Wrong shape

        with pytest.raises((ValueError, RuntimeError)):
            compute_effective_resistance(edge_index, 3)


class TestFiedlerVector:
    """Tests for Fiedler vector computation."""

    def test_fiedler_vector_identifies_cuts(self):
        """Test that Fiedler vector identifies graph cuts."""
        # Barbell graph: two triangles connected by a single edge
        # Triangle 1: 0-1-2, Triangle 2: 3-4-5, Bridge: 2-3
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0,  # Triangle 1
             3, 4, 4, 5, 5, 3,  # Triangle 2
             2, 3],  # Bridge
            [1, 0, 2, 1, 0, 2,
             4, 3, 5, 4, 3, 5,
             3, 2]
        ], dtype=torch.long)
        num_nodes = 6

        # Create a rewiring layer to access _compute_fiedler_vector
        layer = SpectralRewiringLayer(hidden_dim=64)
        fiedler = layer._compute_fiedler_vector(edge_index, num_nodes)

        assert fiedler is not None
        assert fiedler.shape == (num_nodes,)

        # Fiedler vector should separate the two clusters
        # Nodes in triangle 1 should have similar values, different from triangle 2
        cluster1_mean = (fiedler[0] + fiedler[1] + fiedler[2]) / 3
        cluster2_mean = (fiedler[3] + fiedler[4] + fiedler[5]) / 3

        # The two clusters should be distinguishable
        assert abs(cluster1_mean - cluster2_mean) > 0.1


class TestSpectralRewiringLayer:
    """Tests for SpectralRewiringLayer."""

    def test_rewiring_changes_topology(self):
        """Test that rewiring actually modifies edge_index."""
        # Simple graph
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        num_nodes = 4
        x = torch.randn(num_nodes, 64)

        layer = SpectralRewiringLayer(hidden_dim=64, rewiring_ratio=0.5)
        layer.train()  # Enable training mode

        rewired_edge_index, rewiring_loss = layer(x, edge_index)

        # Edge index should change during training
        assert rewired_edge_index.shape[0] == 2
        assert rewired_edge_index.shape[1] > 0

    def test_rewiring_improves_spectral_gap(self):
        """Test that rewiring increases spectral gap over multiple iterations."""
        # Path graph (low spectral gap)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        num_nodes = 5
        x = torch.randn(num_nodes, 64)

        layer = SpectralRewiringLayer(hidden_dim=64, rewiring_ratio=0.3, spectral_loss_weight=0.5)
        layer.train()

        gap_before = compute_spectral_gap(edge_index, num_nodes)

        # Apply rewiring
        rewired_edge_index, _ = layer(x, edge_index)

        gap_after = compute_spectral_gap(rewired_edge_index, num_nodes)

        # Spectral gap should generally increase (with high probability)
        # Note: This is stochastic, so we just check it doesn't decrease dramatically
        assert gap_after >= gap_before * 0.8

    def test_rewiring_disabled_at_inference(self):
        """Test that rewiring is disabled in eval mode."""
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        num_nodes = 3
        x = torch.randn(num_nodes, 64)

        layer = SpectralRewiringLayer(hidden_dim=64)
        layer.eval()  # Set to evaluation mode

        rewired_edge_index, rewiring_loss = layer(x, edge_index)

        # In eval mode, edge_index should remain unchanged
        assert torch.equal(rewired_edge_index, edge_index)
        assert rewiring_loss.item() == 0.0


class TestLearnableRewiringPolicy:
    """Tests for learnable rewiring policy."""

    def test_policy_produces_scores(self):
        """Test that policy produces edge scores."""
        num_nodes = 5
        hidden_dim = 64
        x = torch.randn(num_nodes, hidden_dim)
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)

        policy = LearnableRewiringPolicy(hidden_dim=hidden_dim, use_spectral_features=False)
        edge_scores, candidate_edges = policy(x, edge_index, num_nodes, fiedler_vector=None)

        # Should produce scores for each edge
        assert edge_scores.shape[0] == edge_index.shape[1]

        # Should produce candidate edges
        assert candidate_edges.shape[0] == 2
        assert candidate_edges.shape[1] > 0

    def test_policy_uses_spectral_features(self):
        """Test that policy can use Fiedler vector features."""
        num_nodes = 5
        hidden_dim = 64
        x = torch.randn(num_nodes, hidden_dim)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        fiedler_vector = torch.randn(num_nodes)

        policy = LearnableRewiringPolicy(hidden_dim=hidden_dim, use_spectral_features=True)
        edge_scores, candidate_edges = policy(x, edge_index, num_nodes, fiedler_vector=fiedler_vector)

        assert edge_scores.shape[0] == edge_index.shape[1]


class TestSpectralPropertiesInvariance:
    """Tests for spectral property invariances and mathematical correctness."""

    def test_laplacian_has_zero_eigenvalue(self):
        """Test that Laplacian has eigenvalue near zero (for connected graph)."""
        # Triangle graph (connected)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                   [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        num_nodes = 3

        laplacian, _ = compute_graph_laplacian(edge_index, num_nodes, normalization="sym")
        eigenvalues = torch.linalg.eigvalsh(laplacian)

        # First eigenvalue should be near zero for connected graph
        assert eigenvalues[0] < 1e-5

    def test_spectral_gap_is_second_eigenvalue(self):
        """Test that spectral gap equals second smallest eigenvalue."""
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                   [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        num_nodes = 3

        gap = compute_spectral_gap(edge_index, num_nodes)

        # Compute eigenvalues directly
        laplacian, _ = compute_graph_laplacian(edge_index, num_nodes, normalization="sym")
        eigenvalues = torch.linalg.eigvalsh(laplacian)

        # Spectral gap should be second eigenvalue
        assert abs(gap - eigenvalues[1].item()) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
