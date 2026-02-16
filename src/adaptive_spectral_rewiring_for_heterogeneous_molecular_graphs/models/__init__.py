"""Model implementations and custom components."""

from .model import AdaptiveSpectralGNN, BaselineGNN
from .hetero_model import HeterogeneousAdaptiveSpectralGNN
from .components import (
    SpectralRewiringLayer,
    HeterogeneousSpectralRewiringLayer,
    LearnableRewiringPolicy,
    MotifPreservationLoss,
    SpectralGapLoss,
    compute_graph_laplacian,
    compute_spectral_gap,
    compute_effective_resistance
)

# Aliases for convenience
RewiringPolicy = LearnableRewiringPolicy
AdaptiveSpectralRewiringGNN = AdaptiveSpectralGNN

__all__ = [
    "AdaptiveSpectralGNN",
    "BaselineGNN",
    "HeterogeneousAdaptiveSpectralGNN",
    "AdaptiveSpectralRewiringGNN",
    "SpectralRewiringLayer",
    "HeterogeneousSpectralRewiringLayer",
    "MotifPreservationLoss",
    "SpectralGapLoss",
    "LearnableRewiringPolicy",
    "RewiringPolicy",
    "compute_graph_laplacian",
    "compute_spectral_gap",
    "compute_effective_resistance"
]
