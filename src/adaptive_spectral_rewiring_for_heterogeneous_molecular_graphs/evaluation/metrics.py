"""Evaluation metrics for molecular property prediction."""

import logging
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: str = "binary",
) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC metrics).
        task_type: Type of task ('binary' or 'multiclass').

    Returns:
        Dictionary of metric values.
    """
    metrics = {}

    try:
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        if task_type == "binary":
            metrics['precision'] = precision_score(
                y_true, y_pred, zero_division=0
            )
            metrics['recall'] = recall_score(
                y_true, y_pred, zero_division=0
            )
            metrics['f1'] = f1_score(
                y_true, y_pred, zero_division=0
            )

            # AUC metrics
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                except ValueError as e:
                    logger.warning(f"Could not compute AUC metrics: {e}")
                    metrics['roc_auc'] = 0.0
                    metrics['pr_auc'] = 0.0
        else:
            # Multiclass
            metrics['precision_macro'] = precision_score(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics['recall_macro'] = recall_score(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics['f1_macro'] = f1_score(
                y_true, y_pred, average='macro', zero_division=0
            )

            if y_prob is not None:
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                except ValueError as e:
                    logger.warning(f"Could not compute multiclass AUC: {e}")
                    metrics['roc_auc_ovr'] = 0.0

    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        metrics['accuracy'] = 0.0

    return metrics


def compute_spectral_metrics(
    spectral_gaps_before: list,
    spectral_gaps_after: list,
) -> Dict[str, float]:
    """Compute metrics related to spectral rewiring.

    Args:
        spectral_gaps_before: Spectral gaps before rewiring.
        spectral_gaps_after: Spectral gaps after rewiring.

    Returns:
        Dictionary of spectral metrics.
    """
    metrics = {}

    if len(spectral_gaps_before) > 0 and len(spectral_gaps_after) > 0:
        gaps_before = np.array(spectral_gaps_before)
        gaps_after = np.array(spectral_gaps_after)

        # Mean spectral gap improvement
        mean_before = gaps_before.mean()
        mean_after = gaps_after.mean()

        if mean_before > 0:
            improvement = (mean_after - mean_before) / mean_before
        else:
            improvement = 0.0

        metrics['spectral_gap_before'] = float(mean_before)
        metrics['spectral_gap_after'] = float(mean_after)
        metrics['spectral_gap_improvement'] = float(improvement)
        metrics['spectral_gap_std_before'] = float(gaps_before.std())
        metrics['spectral_gap_std_after'] = float(gaps_after.std())
    else:
        metrics['spectral_gap_improvement'] = 0.0

    return metrics


def compute_rewiring_efficiency(
    num_edges_original: int,
    num_edges_rewired: int,
    performance_gain: float,
) -> float:
    """Compute rewiring efficiency metric.

    Efficiency measures the performance gain per edge added.

    Args:
        num_edges_original: Original number of edges.
        num_edges_rewired: Number of edges after rewiring.
        performance_gain: Improvement in task performance.

    Returns:
        Rewiring efficiency score.
    """
    if num_edges_rewired <= num_edges_original:
        return 0.0

    edge_increase_ratio = (num_edges_rewired - num_edges_original) / num_edges_original

    if edge_increase_ratio > 0:
        efficiency = performance_gain / edge_increase_ratio
    else:
        efficiency = 0.0

    return float(efficiency)
