# Adaptive Spectral Rewiring for Heterogeneous Molecular Graphs

Graph neural network with adaptive spectral rewiring for molecular property prediction. Uses Laplacian eigenvalue decomposition and Fiedler vector analysis to dynamically optimize graph topology during training.

## Key Features

- **Spectral rewiring** with Laplacian eigenvalue decomposition and spectral gap optimization
- **Fiedler vector-guided** candidate edge generation to bridge bottleneck cuts
- **Learnable rewiring policy** that scores edges using structural and spectral features
- **Heterogeneous graph support** with type-specific message passing
- **Effective resistance** computation for identifying graph bottlenecks

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Requirements:
- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0

## Quick Start

Train a model:

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate on test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

Make predictions:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --smiles "CCO"
```

## Architecture

### Spectral Rewiring

The core innovation is adaptive spectral rewiring that:

1. **Computes graph Laplacian**: L = D - A (or normalized variants)
2. **Extracts Fiedler vector**: Second eigenvector of L identifying bottleneck cuts
3. **Generates candidate edges**: Pairs nodes with large Fiedler value differences
4. **Learns edge scoring policy**: MLP incorporating both structural (node embeddings) and spectral (Fiedler values) features
5. **Rewires graph topology**: Removes lowest-scoring edges, adds highest-scoring candidates
6. **Optimizes spectral gap**: Maximizes algebraic connectivity (second smallest eigenvalue)

### Rewiring Loss Formulation

```
L_rewiring = L_edge_quality + λ_spectral * L_spectral_gap

where:
  L_edge_quality = -mean(sigmoid(edge_scores))
  L_spectral_gap = -(gap_after - gap_before)
  gap = λ_2(L), the second smallest Laplacian eigenvalue
```

The learnable policy network identifies edges whose addition increases spectral gap and whose removal has minimal impact on graph connectivity.

### Heterogeneous Graph Support

Handles multiple node and edge types:

- Type-specific input projections
- HeteroConv layers with per-edge-type message passing
- Separate rewiring policies for each edge type
- Type-aware pooling and aggregation

## Configuration

Key settings in `configs/default.yaml`:

```yaml
model:
  hidden_dim: 256
  num_layers: 5
  rewiring:
    enabled: true
    edge_budget: 1.5
    spectral_threshold: 0.5

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping:
    patience: 20
```

## Implementation Details

### Spectral Components

- `compute_graph_laplacian()`: Computes symmetric/asymmetric normalized Laplacian
- `compute_spectral_gap()`: Extracts second smallest eigenvalue (algebraic connectivity)
- `compute_effective_resistance()`: Uses Laplacian pseudoinverse for spectral distance
- `LearnableRewiringPolicy`: MLP edge scorer with spectral features
- `SpectralRewiringLayer`: Applies rewiring with spectral guidance

### Heterogeneous Components

- `HeterogeneousAdaptiveSpectralGNN`: Main heterogeneous model
- `HeterogeneousSpectralRewiringLayer`: Type-specific rewiring policies
- `HeteroConv`: Heterogeneous graph convolution wrapper

## Training

The training loop:

1. Forward pass through GNN layers
2. Apply spectral rewiring at configured layers
3. Compute task loss (binary cross-entropy for classification)
4. Compute spectral gap loss (encourage larger gap)
5. Compute rewiring loss (edge quality + spectral improvement)
6. Backpropagate combined loss: `L_total = L_task + 0.01 * L_rewiring`
7. Update model and rewiring policy parameters

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
├── configs/            # YAML configuration files
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model implementations
│   │   ├── model.py           # Main GNN model
│   │   ├── hetero_model.py    # Heterogeneous GNN
│   │   └── components.py      # Spectral rewiring layers
│   ├── training/      # Training loop and utilities
│   └── evaluation/    # Metrics and analysis
├── scripts/           # Training and evaluation scripts
├── tests/            # Test suite
└── data/             # Dataset storage
```

## Datasets

Supported molecular datasets:

- **OGBG-MOLHIV**: Binary classification for HIV activity prediction
- **OGBG-MOLPCBA**: Multi-task bioassay prediction
- Custom SMILES datasets via preprocessing utilities

## Performance

Spectral decomposition is optimized using:
- `torch.linalg.eigh` for efficient symmetric eigenvalue computation
- Fiedler vector caching during rewiring
- Limited candidate edge sampling to prevent memory issues
- Rewiring only active during training (disabled at inference)

## Technical Notes

### Laplacian Normalization

- **Symmetric** (`sym`): L_sym = I - D^{-1/2} A D^{-1/2} (default, produces symmetric matrix)
- **Random Walk** (`rw`): L_rw = I - D^{-1} A (asymmetric, requires torch.linalg.eig)
- **Unnormalized** (`None`): L = D - A (symmetric)

For spectral gap and Fiedler vector computation, symmetric normalization is used to enable efficient `torch.linalg.eigvalsh`.

### Error Handling

All spectral computation functions raise exceptions on failure rather than silently returning fallback values. This ensures errors are caught early during development.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei
