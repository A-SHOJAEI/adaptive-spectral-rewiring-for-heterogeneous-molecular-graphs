"""
Tests for data loading and preprocessing modules.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.data.preprocessing import (
    smiles_to_graph,
    compute_graph_statistics,
    normalize_features,
    augment_graph
)
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.data.loader import load_molecular_dataset


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""
    
    def test_smiles_to_graph_basic(self, sample_smiles):
        """Test basic SMILES to graph conversion."""
        smiles = sample_smiles[0]
        graph = smiles_to_graph(smiles)
        
        assert isinstance(graph, Data)
        assert graph.x is not None
        assert graph.edge_index is not None
        assert graph.num_nodes > 0
        assert graph.edge_index.size(1) > 0
    
    def test_smiles_to_graph_all_samples(self, sample_smiles):
        """Test SMILES conversion for all samples."""
        for smiles in sample_smiles:
            graph = smiles_to_graph(smiles)
            assert graph.num_nodes > 0
            assert graph.edge_index.size(1) > 0
    
    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        invalid_smiles = "invalid_smiles_123"
        with pytest.raises(Exception):
            smiles_to_graph(invalid_smiles)
    
    def test_compute_graph_statistics(self, simple_graph):
        """Test graph statistics computation."""
        stats = compute_graph_statistics(simple_graph)
        
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'avg_degree' in stats
        assert stats['num_nodes'] == 5
        assert stats['num_edges'] == 8
    
    def test_normalize_features(self, simple_graph):
        """Test feature normalization."""
        normalized_graph = normalize_features(simple_graph)
        
        assert normalized_graph.x.shape == simple_graph.x.shape
        # Check if features are normalized (mean ~0, std ~1)
        assert torch.allclose(normalized_graph.x.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(normalized_graph.x.std(), torch.tensor(1.0), atol=1e-1)
    
    def test_augment_graph(self, simple_graph):
        """Test graph augmentation."""
        augmented = augment_graph(simple_graph, method='drop_nodes', ratio=0.2)
        
        assert isinstance(augmented, Data)
        # Augmented graph should have fewer nodes
        assert augmented.num_nodes <= simple_graph.num_nodes


class TestMolecularGraphDataset:
    """Tests for MolecularGraphDataset class."""
    
    @pytest.mark.slow
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = MolecularGraphDataset(
            name='ogbg-molhiv',
            root='test_data'
        )
        
        assert len(dataset) > 0
        assert dataset.num_features > 0
        assert dataset.num_classes > 0
    
    @pytest.mark.slow
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = MolecularGraphDataset(
            name='ogbg-molhiv',
            root='test_data'
        )
        
        graph = dataset[0]
        assert isinstance(graph, Data)
        assert graph.x is not None
        assert graph.edge_index is not None
        assert graph.y is not None
    
    def test_mock_dataset(self, mock_dataset):
        """Test with mock dataset."""
        assert len(mock_dataset) == 20
        assert mock_dataset.num_features == 32
        assert mock_dataset.num_classes == 2
        
        # Test item retrieval
        graph = mock_dataset[0]
        assert isinstance(graph, Data)
        assert graph.num_nodes == 10


class TestDataLoading:
    """Tests for data loading utilities."""
    
    def test_batch_creation(self, simple_graph):
        """Test batch creation from graphs."""
        graphs = [simple_graph.clone() for _ in range(4)]
        batch = Batch.from_data_list(graphs)
        
        assert batch.num_graphs == 4
        assert batch.x.size(0) == 5 * 4  # 5 nodes per graph, 4 graphs
    
    def test_batch_with_different_sizes(self):
        """Test batching graphs of different sizes."""
        graphs = []
        for i in range(3):
            num_nodes = 5 + i * 2
            x = torch.randn(num_nodes, 16)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            graphs.append(Data(x=x, edge_index=edge_index))
        
        batch = Batch.from_data_list(graphs)
        assert batch.num_graphs == 3
    
    def test_dataloader_iteration(self, mock_dataset):
        """Test DataLoader iteration."""
        from torch.utils.data import DataLoader
        
        loader = DataLoader(mock_dataset, batch_size=4, shuffle=False)
        
        batch_count = 0
        for batch in loader:
            assert isinstance(batch, Data) or isinstance(batch, Batch)
            batch_count += 1
        
        assert batch_count == 5  # 20 samples / 4 batch_size


class TestFeatureExtraction:
    """Tests for feature extraction."""
    
    def test_node_features_shape(self, molecular_graph):
        """Test node features have correct shape."""
        assert molecular_graph.x.dim() == 2
        assert molecular_graph.x.size(0) == molecular_graph.num_nodes
    
    def test_edge_features_shape(self, molecular_graph):
        """Test edge features have correct shape."""
        if molecular_graph.edge_attr is not None:
            assert molecular_graph.edge_attr.dim() == 2
            assert molecular_graph.edge_attr.size(0) == molecular_graph.edge_index.size(1)
    
    def test_type_features(self, molecular_graph):
        """Test node and edge type features."""
        assert hasattr(molecular_graph, 'node_type')
        assert hasattr(molecular_graph, 'edge_type')
        assert molecular_graph.node_type.size(0) == molecular_graph.num_nodes
        assert molecular_graph.edge_type.size(0) == molecular_graph.edge_index.size(1)


class TestDataTransforms:
    """Tests for data transformations."""
    
    def test_add_self_loops(self, simple_graph):
        """Test adding self-loops to graph."""
        from torch_geometric.utils import add_self_loops, remove_self_loops
        
        # Remove existing self-loops first
        edge_index, _ = remove_self_loops(simple_graph.edge_index)
        
        # Add self-loops
        edge_index_with_loops, _ = add_self_loops(edge_index)
        
        assert edge_index_with_loops.size(1) > edge_index.size(1)
    
    def test_to_undirected(self, simple_graph):
        """Test converting to undirected graph."""
        from torch_geometric.utils import to_undirected
        
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        undirected = to_undirected(edge_index)
        
        # Undirected should have both directions
        assert undirected.size(1) >= edge_index.size(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
