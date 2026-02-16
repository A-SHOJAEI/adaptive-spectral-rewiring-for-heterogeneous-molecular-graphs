"""
Tests for model components.
"""

import pytest
import torch
import torch.nn as nn

from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.components import SpectralRewiringLayer


class TestSpectralConv:
    """Tests for SpectralConv layer."""
    
    def test_spectral_conv_initialization(self):
        """Test SpectralConv initialization."""
        layer = SpectralConv(in_channels=32, out_channels=64)
        
        assert layer.in_channels == 32
        assert layer.out_channels == 64
    
    def test_spectral_conv_forward(self, simple_graph, device):
        """Test SpectralConv forward pass."""
        layer = SpectralConv(in_channels=16, out_channels=32).to(device)
        graph = simple_graph.to(device)
        
        out = layer(graph.x, graph.edge_index)
        
        assert out.shape == (graph.num_nodes, 32)
    
    def test_spectral_conv_with_edge_weight(self, simple_graph, device):
        """Test SpectralConv with edge weights."""
        layer = SpectralConv(in_channels=16, out_channels=32).to(device)
        graph = simple_graph.to(device)
        edge_weight = torch.rand(graph.edge_index.size(1)).to(device)
        
        out = layer(graph.x, graph.edge_index, edge_weight)
        
        assert out.shape == (graph.num_nodes, 32)


class TestAdaptiveRewiring:
    """Tests for AdaptiveRewiring module."""
    
    def test_rewiring_initialization(self, config_dict):
        """Test AdaptiveRewiring initialization."""
        rewiring = AdaptiveRewiring(
            hidden_dim=64,
            config=config_dict['model']['rewiring']
        )
        
        assert rewiring.k_hops == 2
        assert rewiring.edge_budget == 1.5
    
    def test_rewiring_forward(self, simple_graph, device, config_dict):
        """Test rewiring forward pass."""
        rewiring = AdaptiveRewiring(
            hidden_dim=16,
            config=config_dict['model']['rewiring']
        ).to(device)
        
        graph = simple_graph.to(device)
        
        new_edge_index, edge_scores = rewiring(graph.x, graph.edge_index)
        
        assert new_edge_index.shape[0] == 2
        assert new_edge_index.shape[1] > 0
    
    def test_rewiring_preserves_original(self, simple_graph, device, config_dict):
        """Test that rewiring preserves original edges when configured."""
        config_dict['model']['rewiring']['preserve_original'] = True
        
        rewiring = AdaptiveRewiring(
            hidden_dim=16,
            config=config_dict['model']['rewiring']
        ).to(device)
        
        graph = simple_graph.to(device)
        original_edges = graph.edge_index.size(1)
        
        new_edge_index, _ = rewiring(graph.x, graph.edge_index)
        
        # New graph should have at least as many edges as original
        assert new_edge_index.shape[1] >= original_edges
    
    def test_rewiring_edge_budget(self, simple_graph, device, config_dict):
        """Test edge budget constraint."""
        config_dict['model']['rewiring']['edge_budget'] = 1.5
        
        rewiring = AdaptiveRewiring(
            hidden_dim=16,
            config=config_dict['model']['rewiring']
        ).to(device)
        
        graph = simple_graph.to(device)
        original_edges = graph.edge_index.size(1)
        
        new_edge_index, _ = rewiring(graph.x, graph.edge_index)
        
        # Check edge budget is respected
        assert new_edge_index.shape[1] <= original_edges * 1.5 * 1.1  # Allow 10% tolerance


class TestGraphPooling:
    """Tests for GraphPooling module."""
    
    @pytest.mark.parametrize("pooling_type", ["mean", "max", "sum", "attention"])
    def test_pooling_types(self, simple_graph, device, pooling_type):
        """Test different pooling types."""
        pooling = GraphPooling(hidden_dim=16, pooling_type=pooling_type).to(device)
        graph = simple_graph.to(device)
        
        # Create batch tensor
        batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
        
        out = pooling(graph.x, batch)
        
        assert out.shape == (1, 16)  # 1 graph, 16 features
    
    def test_pooling_multiple_graphs(self, graph_batch, device):
        """Test pooling with multiple graphs."""
        pooling = GraphPooling(hidden_dim=16, pooling_type='mean').to(device)
        graph_batch = graph_batch.to(device)
        
        out = pooling(graph_batch.x, graph_batch.batch)
        
        assert out.shape[0] == graph_batch.num_graphs
        assert out.shape[1] == 16


class TestAdaptiveSpectralGNN:
    """Tests for AdaptiveSpectralGNN model."""
    
    def test_model_initialization(self, config_dict):
        """Test model initialization."""
        model = AdaptiveSpectralGNN(
            in_channels=32,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        )
        
        assert model.in_channels == 32
        assert model.hidden_channels == 64
        assert model.out_channels == 2
        assert model.num_layers == 3
    
    def test_model_forward(self, simple_graph, device, config_dict):
        """Test model forward pass."""
        model = AdaptiveSpectralGNN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        graph = simple_graph.to(device)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
        
        out = model(graph)
        
        assert out.shape == (1, 2)  # 1 graph, 2 classes
    
    def test_model_forward_batch(self, graph_batch, device, config_dict):
        """Test model forward pass with batch."""
        model = AdaptiveSpectralGNN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        graph_batch = graph_batch.to(device)
        
        out = model(graph_batch)
        
        assert out.shape == (graph_batch.num_graphs, 2)
    
    def test_model_with_rewiring_disabled(self, simple_graph, device, config_dict):
        """Test model with rewiring disabled."""
        config_dict['model']['rewiring']['enabled'] = False
        
        model = AdaptiveSpectralGNN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        graph = simple_graph.to(device)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
        
        out = model(graph)
        
        assert out.shape == (1, 2)
    
    def test_model_dropout(self, simple_graph, device, config_dict):
        """Test model with dropout."""
        model = AdaptiveSpectralGNN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.5,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        graph = simple_graph.to(device)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
        
        # Training mode
        model.train()
        out1 = model(graph)
        
        # Evaluation mode
        model.eval()
        out2 = model(graph)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(out1, out2)
    
    def test_model_parameter_count(self, config_dict):
        """Test model parameter count."""
        model = AdaptiveSpectralGNN(
            in_channels=32,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        )
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert num_params > 0
    
    @pytest.mark.parametrize("gnn_type", ["gcn", "gat", "gin", "sage"])
    def test_different_gnn_types(self, simple_graph, device, config_dict, gnn_type):
        """Test different GNN backbone types."""
        model = AdaptiveSpectralGNN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            gnn_type=gnn_type,
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        graph = simple_graph.to(device)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
        
        out = model(graph)
        
        assert out.shape == (1, 2)


class TestModelComponents:
    """Tests for miscellaneous model components."""
    
    def test_batch_norm(self, device):
        """Test batch normalization."""
        bn = nn.BatchNorm1d(32).to(device)
        x = torch.randn(10, 32).to(device)
        
        out = bn(x)
        
        assert out.shape == x.shape
    
    def test_activation_functions(self, device):
        """Test various activation functions."""
        x = torch.randn(10, 32).to(device)
        
        relu_out = torch.relu(x)
        assert relu_out.shape == x.shape
        
        gelu_out = torch.nn.functional.gelu(x)
        assert gelu_out.shape == x.shape
        
        elu_out = torch.nn.functional.elu(x)
        assert elu_out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
