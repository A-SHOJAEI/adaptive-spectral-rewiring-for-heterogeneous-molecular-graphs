# Quick Start Guide

## Installation

```bash
# Clone or navigate to the project directory
cd adaptive-spectral-rewiring-for-heterogeneous-molecular-graphs

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Basic Usage

### 1. Train a Model

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train with custom settings
python scripts/train.py \
    --config configs/default.yaml \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### 2. Evaluate a Trained Model

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint models/best_model.pt \
    --config configs/default.yaml

# Evaluate with specific split
python scripts/evaluate.py \
    --checkpoint models/best_model.pt \
    --split test
```

### 3. Make Predictions

```bash
# Predict from SMILES string
python scripts/predict.py \
    --checkpoint models/best_model.pt \
    --smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

# Predict from CSV file
python scripts/predict.py \
    --checkpoint models/best_model.pt \
    --input molecules.csv \
    --output predictions.csv
```

### 4. Run Ablation Studies

```bash
# Baseline (no rewiring)
python scripts/train.py \
    --config configs/ablation.yaml \
    --variant baseline

# Full rewiring
python scripts/train.py \
    --config configs/ablation.yaml \
    --variant full_rewiring

# Different k values
python scripts/train.py \
    --config configs/ablation.yaml \
    --variant k_rewire_3
```

### 5. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Project Structure Overview

```
adaptive-spectral-rewiring-for-heterogeneous-molecular-graphs/
├── configs/           # YAML configurations
├── scripts/           # Executable scripts (train, eval, predict)
├── src/              # Source code
│   └── adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/
│       ├── data/     # Data loading & preprocessing
│       ├── models/   # Model implementations
│       ├── training/ # Training loop
│       ├── evaluation/ # Metrics & analysis
│       └── utils/    # Utilities
├── tests/            # Unit tests
├── models/           # Saved model checkpoints
└── results/          # Evaluation results
```

## Key Configuration Parameters

Edit `configs/default.yaml`:

```yaml
model:
  hidden_dim: 64          # Hidden layer size
  num_layers: 3           # Number of GNN layers
  k_rewire: 5             # Edges to consider for rewiring
  
training:
  learning_rate: 0.001    # Initial learning rate
  batch_size: 32          # Training batch size
  num_epochs: 100         # Maximum epochs
  early_stopping_patience: 20

data:
  dataset_name: "BBBP"    # MoleculeNet dataset
  train_ratio: 0.8        # Training split
  val_ratio: 0.1          # Validation split
```

## Common Tasks

### Check Model Architecture

```python
from adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models import AdaptiveSpectralGNN

model = AdaptiveSpectralGNN(
    input_dim=9,
    hidden_dim=64,
    output_dim=1,
    num_layers=3,
    k_rewire=5
)
print(model)
```

### Load and Inspect Dataset

```python
from adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.data import get_moleculenet_dataset

dataset = get_moleculenet_dataset(name="BBBP", root="./data")
print(f"Dataset size: {len(dataset)}")
print(f"First graph: {dataset[0]}")
```

### Custom Training Loop

```python
from adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.training import Trainer
from adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.utils import set_seed

set_seed(42)

trainer = Trainer(
    model=model,
    device=device,
    learning_rate=0.001,
    scheduler_type="cosine"
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100
)
```

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 16  # or smaller
```

### Slow Training

Enable mixed precision:
```bash
python scripts/train.py --config configs/default.yaml --use-amp
```

### Poor Performance

Try different configurations:
- Increase `hidden_dim` (64 → 128)
- Increase `num_layers` (3 → 4)
- Adjust `learning_rate` (0.001 → 0.0005)
- Change `k_rewire` (5 → 10)

## Next Steps

1. Review the full documentation in `README.md`
2. Explore ablation study results
3. Customize the model for your specific task
4. Extend with additional molecular descriptors
5. Implement your own rewiring strategies

## Support

For issues or questions, please refer to:
- README.md for detailed documentation
- Comments in source code for implementation details
- Test files for usage examples
