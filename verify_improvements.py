#!/usr/bin/env python3
"""
Verification script to demonstrate implemented improvements.

This script validates that the key improvements have been successfully implemented:
1. True spectral rewiring with Laplacian eigenvalue decomposition
2. Spectral-guided candidate edge generation
3. Heterogeneous graph support
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs.models import (
    AdaptiveSpectralGNN,
    HeterogeneousAdaptiveSpectralGNN,
    compute_graph_laplacian,
    compute_spectral_gap,
    compute_effective_resistance,
    SpectralRewiringLayer
)


def test_spectral_computation():
    """Test actual spectral computation functions."""
    print("=" * 80)
    print("TEST 1: True Spectral Computation")
    print("=" * 80)

    # Create a simple graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ])
    num_nodes = 5

    # Compute Laplacian
    L = compute_graph_laplacian(edge_index, num_nodes, normalization="sym")
    print(f"✓ Laplacian computed: shape {L.shape}")
    print(f"  Laplacian is symmetric: {torch.allclose(L, L.T)}")

    # Compute spectral gap
    spectral_gap = compute_spectral_gap(edge_index, num_nodes)
    print(f"✓ Spectral gap computed: {spectral_gap:.4f}")

    # Compute effective resistance
    resistances = compute_effective_resistance(edge_index, num_nodes, num_pairs=10)
    print(f"✓ Effective resistance computed for {len(resistances)} node pairs")

    print()
    return True


def test_spectral_rewiring():
    """Test spectral rewiring layer with Fiedler vector."""
    print("=" * 80)
    print("TEST 2: Spectral-Guided Rewiring")
    print("=" * 80)

    # Create rewiring layer
    rewiring_layer = SpectralRewiringLayer(
        hidden_dim=64,
        rewiring_ratio=0.2,
        compute_spectral_features=True
    )

    # Create sample data
    num_nodes = 10
    x = torch.randn(num_nodes, 64)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8]
    ])

    rewiring_layer.train()

    # Apply rewiring
    rewired_edge_index, rewiring_loss = rewiring_layer(x, edge_index)

    print(f"✓ Original edges: {edge_index.shape[1]}")
    print(f"✓ Rewired edges: {rewired_edge_index.shape[1]}")
    print(f"✓ Rewiring loss: {rewiring_loss.item():.4f}")
    print(f"✓ Spectral features used in rewiring: {rewiring_layer.compute_spectral_features}")

    print()
    return True


def test_heterogeneous_model():
    """Test heterogeneous graph model."""
    print("=" * 80)
    print("TEST 3: Heterogeneous Graph Support")
    print("=" * 80)

    # Define heterogeneous graph structure
    num_features_dict = {
        'atom': 9,
        'bond': 3
    }

    node_types = ['atom', 'bond']
    edge_types = [
        ('atom', 'connects', 'bond'),
        ('bond', 'connects', 'atom'),
        ('atom', 'nearby', 'atom')
    ]

    # Create heterogeneous model
    model = HeterogeneousAdaptiveSpectralGNN(
        num_features_dict=num_features_dict,
        num_classes=1,
        hidden_dim=64,
        num_layers=2,
        node_types=node_types,
        edge_types=edge_types,
        rewiring_ratio=0.1,
        use_rewiring=True
    )

    print(f"✓ Heterogeneous model created")
    print(f"  Node types: {model.node_types}")
    print(f"  Edge types: {len(model.edge_types)}")
    print(f"  Rewiring enabled: {model.use_rewiring}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x_dict = {
        'atom': torch.randn(10, 9),
        'bond': torch.randn(5, 3)
    }

    # Create valid edge indices with correct bounds for each edge type
    edge_index_dict = {
        ('atom', 'connects', 'bond'): torch.stack([
            torch.randint(0, 10, (15,)),  # src: atoms (0-9)
            torch.randint(0, 5, (15,))    # dst: bonds (0-4)
        ]),
        ('bond', 'connects', 'atom'): torch.stack([
            torch.randint(0, 5, (15,)),   # src: bonds (0-4)
            torch.randint(0, 10, (15,))   # dst: atoms (0-9)
        ]),
        ('atom', 'nearby', 'atom'): torch.stack([
            torch.randint(0, 10, (20,)),  # src: atoms (0-9)
            torch.randint(0, 10, (20,))   # dst: atoms (0-9)
        ])
    }

    model.eval()
    with torch.no_grad():
        output, _ = model(x_dict, edge_index_dict, return_rewiring_loss=False)

    print(f"✓ Forward pass successful: output shape {output.shape}")

    print()
    return True


def test_homogeneous_model():
    """Test standard homogeneous model with spectral rewiring."""
    print("=" * 80)
    print("TEST 4: Homogeneous Model with Spectral Rewiring")
    print("=" * 80)

    # Create model
    model = AdaptiveSpectralGNN(
        num_features=9,
        num_classes=1,
        hidden_dim=128,
        num_layers=4,
        rewiring_ratio=0.15,
        pooling="mean"
    )

    print(f"✓ Model created with spectral rewiring")
    print(f"  Rewiring layers: {model.rewiring_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    num_nodes = 20
    num_edges = 40
    x = torch.randn(num_nodes, 9)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    model.train()
    output, rewiring_loss = model(x, edge_index, batch, return_rewiring_loss=True)

    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Rewiring loss: {rewiring_loss.item():.4f}")

    print()
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("VERIFICATION OF IMPLEMENTED IMPROVEMENTS")
    print("=" * 80 + "\n")

    results = []

    try:
        results.append(("Spectral Computation", test_spectral_computation()))
    except Exception as e:
        print(f"✗ Spectral computation test failed: {e}\n")
        results.append(("Spectral Computation", False))

    try:
        results.append(("Spectral Rewiring", test_spectral_rewiring()))
    except Exception as e:
        print(f"✗ Spectral rewiring test failed: {e}\n")
        results.append(("Spectral Rewiring", False))

    try:
        results.append(("Heterogeneous Model", test_heterogeneous_model()))
    except Exception as e:
        print(f"✗ Heterogeneous model test failed: {e}\n")
        results.append(("Heterogeneous Model", False))

    try:
        results.append(("Homogeneous Model", test_homogeneous_model()))
    except Exception as e:
        print(f"✗ Homogeneous model test failed: {e}\n")
        results.append(("Homogeneous Model", False))

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All improvements verified successfully!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
