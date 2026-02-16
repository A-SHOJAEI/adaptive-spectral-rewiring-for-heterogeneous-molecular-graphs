"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch


@pytest.fixture
def device():
    """Fixture for device selection."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Fixture for random seed."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def simple_graph():
    """Fixture for a simple graph."""
    x = torch.randn(5, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    y = torch.tensor([0])
    
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def molecular_graph():
    """Fixture for a molecular graph with node and edge types."""
    x = torch.randn(10, 32)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8]
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), 8)
    node_type = torch.randint(0, 4, (10,))
    edge_type = torch.randint(0, 3, (edge_index.size(1),))
    y = torch.tensor([1])
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        edge_type=edge_type,
        y=y
    )


@pytest.fixture
def graph_batch(simple_graph):
    """Fixture for a batch of graphs."""
    graphs = []
    for i in range(4):
        graph = simple_graph.clone()
        graph.y = torch.tensor([i % 2])
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


@pytest.fixture
def config_dict():
    """Fixture for configuration dictionary."""
    return {
        'experiment': {
            'name': 'test_experiment',
            'seed': 42,
            'device': 'cpu',
            'log_dir': 'test_logs',
            'checkpoint_dir': 'test_checkpoints'
        },
        'data': {
            'dataset': 'ogbg-molhiv',
            'data_dir': 'test_data',
            'batch_size': 4,
            'num_workers': 0,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        'model': {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'pooling': 'mean',
            'use_batch_norm': True,
            'activation': 'relu',
            'gnn': {
                'type': 'gcn',
                'aggr': 'add',
                'normalize': True
            },
            'rewiring': {
                'enabled': True,
                'method': 'adaptive_spectral',
                'k_hops': 2,
                'top_k_edges': 5,
                'spectral_threshold': 0.5,
                'update_frequency': 1,
                'preserve_original': True,
                'edge_budget': 1.5,
                'use_laplacian': True,
                'use_adjacency': True,
                'num_eigenvalues': 16,
                'adaptive_threshold': True,
                'threshold_learning_rate': 0.001,
                'node_type_aware': True,
                'edge_type_aware': True
            }
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'warmup_epochs': 2,
            'grad_clip': 1.0,
            'early_stopping': {
                'enabled': True,
                'patience': 5,
                'min_delta': 0.0001
            },
            'loss': {
                'type': 'cross_entropy',
                'label_smoothing': 0.0
            },
            'regularization': {
                'spectral_loss_weight': 0.01,
                'rewiring_loss_weight': 0.001
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'auc_roc', 'f1_score'],
            'eval_frequency': 1
        },
        'logging': {
            'level': 'INFO',
            'use_wandb': False,
            'log_frequency': 5
        }
    }


@pytest.fixture
def mock_dataset():
    """Fixture for a mock dataset."""
    class MockDataset:
        def __init__(self):
            self.num_features = 32
            self.num_classes = 2
            self.data = []
            
            # Generate 20 sample graphs
            for i in range(20):
                x = torch.randn(10, 32)
                edge_index = torch.tensor([
                    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8]
                ], dtype=torch.long)
                y = torch.tensor([i % 2])
                self.data.append(Data(x=x, edge_index=edge_index, y=y))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return MockDataset()


@pytest.fixture
def sample_smiles():
    """Fixture for sample SMILES strings."""
    return [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CCN(CC)CC",  # Triethylamine
        "CC(C)C",  # Isobutane
    ]


@pytest.fixture
def mock_checkpoint(tmp_path, config_dict):
    """Fixture for a mock checkpoint file."""
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    
    checkpoint = {
        'epoch': 10,
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'test_metrics': {'accuracy': 0.85, 'auc_roc': 0.90},
        'config': config_dict
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup fixture that runs after each test."""
    yield
    # Cleanup code can be added here if needed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
