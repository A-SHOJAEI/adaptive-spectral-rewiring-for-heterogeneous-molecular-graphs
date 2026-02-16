"""Results analysis and visualization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzer for model results and visualizations."""

    def __init__(self, results_dir: str = "./results"):
        """Initialize analyzer.

        Args:
            results_dir: Directory to save results.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = "metrics.json",
    ) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics.
            filename: Output filename.
        """
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_path}")

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        filename: str = "training_history.png",
    ) -> None:
        """Plot training and validation loss curves.

        Args:
            train_losses: List of training losses.
            val_losses: List of validation losses.
            filename: Output filename.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training history plot to {output_path}")

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        filename: str = "metrics_comparison.png",
    ) -> None:
        """Plot comparison of metrics across different models.

        Args:
            metrics_dict: Dictionary mapping model names to metrics.
            filename: Output filename.
        """
        df = pd.DataFrame(metrics_dict).T

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot classification metrics
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        available_metrics = [m for m in classification_metrics if m in df.columns]

        if available_metrics:
            df[available_metrics].plot(kind='bar', ax=axes[0])
            axes[0].set_title('Classification Metrics', fontsize=14)
            axes[0].set_ylabel('Score', fontsize=12)
            axes[0].set_xlabel('Model', fontsize=12)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

        # Plot spectral metrics
        spectral_metrics = ['spectral_gap_improvement', 'rewiring_efficiency']
        available_spectral = [m for m in spectral_metrics if m in df.columns]

        if available_spectral:
            df[available_spectral].plot(kind='bar', ax=axes[1])
            axes[1].set_title('Spectral Rewiring Metrics', fontsize=14)
            axes[1].set_ylabel('Score', fontsize=12)
            axes[1].set_xlabel('Model', fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metrics comparison to {output_path}")

    def plot_spectral_gap_distribution(
        self,
        gaps_before: List[float],
        gaps_after: List[float],
        filename: str = "spectral_gap_distribution.png",
    ) -> None:
        """Plot distribution of spectral gaps before and after rewiring.

        Args:
            gaps_before: Spectral gaps before rewiring.
            gaps_after: Spectral gaps after rewiring.
            filename: Output filename.
        """
        plt.figure(figsize=(10, 6))

        plt.hist(
            gaps_before,
            bins=30,
            alpha=0.5,
            label='Before Rewiring',
            color='blue',
            edgecolor='black',
        )
        plt.hist(
            gaps_after,
            bins=30,
            alpha=0.5,
            label='After Rewiring',
            color='red',
            edgecolor='black',
        )

        plt.xlabel('Spectral Gap', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Spectral Gap Distribution', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved spectral gap distribution to {output_path}")

    def generate_report(
        self,
        metrics: Dict[str, float],
        config: Dict,
    ) -> str:
        """Generate a text report of results.

        Args:
            metrics: Dictionary of metrics.
            config: Configuration dictionary.

        Returns:
            Report string.
        """
        report_lines = [
            "=" * 60,
            "EVALUATION REPORT",
            "=" * 60,
            "",
            "Configuration:",
            "-" * 60,
        ]

        for key, value in config.items():
            report_lines.append(f"  {key}: {value}")

        report_lines.extend([
            "",
            "Results:",
            "-" * 60,
        ])

        for key, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.4f}")
            else:
                report_lines.append(f"  {key}: {value}")

        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        # Save to file
        report_path = self.results_dir / "report.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Generated report at {report_path}")

        return report
