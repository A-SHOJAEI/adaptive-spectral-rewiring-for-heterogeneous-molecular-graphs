"""
Tests for training components.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.training.trainer import Trainer
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.components import SpectralGapLoss, MotifPreservationLoss
from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.evaluation.metrics import compute_metrics


class TestTrainer:
    """Tests for Trainer class."""
    
    def test_trainer_initialization(self, device, config_dict):
        """Test Trainer initialization."""
        from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN
        
        model = AdaptiveSpectralGNN(
            in_channels=32,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        config = Config(config_dict)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            config=config,
            logger=None
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device == device
    
    def test_training_step(self, device, config_dict, graph_batch):
        """Test single training step."""
        from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN
        
        model = AdaptiveSpectralGNN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        config = Config(config_dict)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            config=config,
            logger=None
        )
        
        graph_batch = graph_batch.to(device)
        loss = trainer.training_step(graph_batch)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_validation_step(self, device, config_dict, graph_batch):
        """Test single validation step."""
        from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN
        
        model = AdaptiveSpectralGNN(
            in_channels=16,
            hidden_channels=32,
            out_channels=2,
            num_layers=3,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        config = Config(config_dict)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            config=config,
            logger=None
        )
        
        graph_batch = graph_batch.to(device)
        loss, metrics = trainer.validation_step(graph_batch)
        
        assert isinstance(loss, float)
        assert isinstance(metrics, dict)
        assert loss > 0
    
    def test_full_training_loop(self, device, config_dict, mock_dataset):
        """Test full training loop with mock data."""
        from src.adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models.model import AdaptiveSpectralGNN
        
        # Reduce epochs for testing
        config_dict['training']['epochs'] = 2
        
        model = AdaptiveSpectralGNN(
            in_channels=32,
            hidden_channels=32,
            out_channels=2,
            num_layers=2,
            dropout=0.1,
            pooling='mean',
            rewiring_config=config_dict['model']['rewiring']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        config = Config(config_dict)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            config=config,
            logger=None
        )
        
        # Create dataloaders
        train_loader = DataLoader(mock_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(mock_dataset, batch_size=4, shuffle=False)
        
        history = trainer.train(train_loader, val_loader)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 2


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_cross_entropy_loss(self, device):
        """Test cross-entropy loss."""
        criterion = nn.CrossEntropyLoss()
        
        logits = torch.randn(10, 2).to(device)
        labels = torch.randint(0, 2, (10,)).to(device)
        
        loss = criterion(logits, labels)
        
        assert loss.item() > 0
        assert loss.requires_grad
    
    def test_spectral_loss(self, simple_graph, device):
        """Test spectral loss."""
        spectral_loss = SpectralLoss().to(device)
        
        graph = simple_graph.to(device)
        loss = spectral_loss(graph.x, graph.edge_index)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_rewiring_smoothness(self, simple_graph, device):
        """Test rewiring smoothness loss."""
        smoothness_loss = RewiringSmoothness().to(device)
        
        graph = simple_graph.to(device)
        edge_scores = torch.rand(graph.edge_index.size(1)).to(device)
        
        loss = smoothness_loss(graph.x, graph.edge_index, edge_scores)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestOptimizers:
    """Tests for optimizers."""
    
    @pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "sgd"])
    def test_optimizer_creation(self, optimizer_name):
        """Test creation of different optimizers."""
        model = nn.Linear(10, 2)
        
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        assert optimizer is not None
    
    def test_optimizer_step(self, device):
        """Test optimizer step."""
        model = nn.Linear(10, 2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Get initial parameter
        initial_param = model.weight.clone()
        
        # Forward and backward
        x = torch.randn(5, 10).to(device)
        y = torch.randint(0, 2, (5,)).to(device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check parameters changed
        assert not torch.equal(initial_param, model.weight)


class TestSchedulers:
    """Tests for learning rate schedulers."""
    
    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler
        for _ in range(5):
            scheduler.step()
        
        # LR should have changed
        assert optimizer.param_groups[0]['lr'] != initial_lr
    
    def test_step_scheduler(self):
        """Test step LR scheduler."""
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step 3 times
        for _ in range(3):
            scheduler.step()
        
        # LR should have decreased
        assert optimizer.param_groups[0]['lr'] < initial_lr


class TestMetrics:
    """Tests for metrics calculation."""
    
    def test_accuracy(self):
        """Test accuracy metric."""
        calc = MetricsCalculator()
        
        predictions = torch.tensor([0, 1, 1, 0, 1])
        labels = torch.tensor([0, 1, 0, 0, 1])
        
        accuracy = calc.compute_accuracy(predictions, labels)
        
        assert 0 <= accuracy <= 1
        assert accuracy == 0.8  # 4 out of 5 correct
    
    def test_metrics_calculator(self):
        """Test MetricsCalculator."""
        calc = MetricsCalculator()
        
        predictions = torch.tensor([0, 1, 1, 0, 1]).numpy()
        labels = torch.tensor([0, 1, 0, 0, 1]).numpy()
        probabilities = torch.softmax(torch.randn(5, 2), dim=1).numpy()
        
        metrics = calc.compute_metrics(
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            metric_names=['accuracy', 'f1_score', 'precision', 'recall']
        )
        
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics


class TestEarlyStopping:
    """Tests for early stopping."""
    
    def test_early_stopping_basic(self):
        """Test basic early stopping."""
        from src.training.trainer import EarlyStopping
        
        early_stopping = EarlyStopping(patience=3, min_delta=0.001, mode='max')
        
        # Improving metrics
        assert not early_stopping.should_stop(0.5)
        assert not early_stopping.should_stop(0.6)
        assert not early_stopping.should_stop(0.7)
        
        # Plateauing metrics
        assert not early_stopping.should_stop(0.7)
        assert not early_stopping.should_stop(0.7)
        assert not early_stopping.should_stop(0.7)
        assert early_stopping.should_stop(0.7)
    
    def test_early_stopping_mode_min(self):
        """Test early stopping with minimization."""
        from src.training.trainer import EarlyStopping
        
        early_stopping = EarlyStopping(patience=2, min_delta=0.001, mode='min')
        
        # Improving (decreasing) metrics
        assert not early_stopping.should_stop(1.0)
        assert not early_stopping.should_stop(0.8)
        assert not early_stopping.should_stop(0.6)
        
        # Not improving
        assert not early_stopping.should_stop(0.7)
        assert not early_stopping.should_stop(0.7)
        assert early_stopping.should_stop(0.7)


class TestGradientClipping:
    """Tests for gradient clipping."""
    
    def test_gradient_clipping(self, device):
        """Test gradient clipping."""
        model = nn.Linear(10, 2).to(device)
        
        # Create large gradients
        x = torch.randn(5, 10).to(device)
        y = torch.randint(0, 2, (5,)).to(device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y) * 100  # Scale up loss
        loss.backward()
        
        # Get gradient norm before clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        # Clip gradients
        max_norm = 1.0
        grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Check clipping worked
        if grad_norm_before > max_norm:
            assert grad_norm_after <= max_norm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
