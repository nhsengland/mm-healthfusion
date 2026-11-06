"""Share binary classification metrics between training and evaluation."""

from __future__ import annotations

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryMatthewsCorrCoef,
    BinaryRecall,
    BinarySpecificity,
)


class BinaryBalancedAccuracy(Metric):
    """Average the fractions of positive and negative labels predicted correctly."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("true_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Count correct and incorrect predictions using a probability threshold of 0.5."""
        y_pred = (preds.reshape(-1) >= 0.5).to(dtype=torch.int64)
        y_true = target.reshape(-1).to(dtype=torch.int64)
        self.true_positives += torch.sum((y_pred == 1) & (y_true == 1)).to(
            dtype=self.true_positives.dtype,
        )
        self.true_negatives += torch.sum((y_pred == 0) & (y_true == 0)).to(
            dtype=self.true_negatives.dtype,
        )
        self.false_positives += torch.sum((y_pred == 1) & (y_true == 0)).to(
            dtype=self.false_positives.dtype,
        )
        self.false_negatives += torch.sum((y_pred == 0) & (y_true == 1)).to(
            dtype=self.false_negatives.dtype,
        )

    def compute(self) -> torch.Tensor:
        """Average recall and specificity, using zero for a class with no samples."""
        recall_denominator = self.true_positives + self.false_negatives
        specificity_denominator = self.true_negatives + self.false_positives
        recall = self.true_positives / recall_denominator.clamp_min(1.0)
        specificity = self.true_negatives / specificity_denominator.clamp_min(1.0)
        return (recall + specificity) / 2.0


def build_binary_metrics() -> MetricCollection:
    """Create AUROC, recall, specificity, MCC and balanced accuracy metrics."""
    return MetricCollection(
        {
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "mcc": BinaryMatthewsCorrCoef(),
            "ba": BinaryBalancedAccuracy(),
        }
    )
