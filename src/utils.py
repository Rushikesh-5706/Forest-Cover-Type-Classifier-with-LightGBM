"""Utility functions for Forest Cover Type Classifier."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, accuracy_score)


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, macro and micro F1, precision, recall.
    """
    return {
        "Accuracy":   accuracy_score(y_true, y_pred),
        "Macro F1":   f1_score(y_true, y_pred, average="macro"),
        "Micro F1":   f1_score(y_true, y_pred, average="micro"),
        "Macro Prec": precision_score(y_true, y_pred, average="macro"),
        "Macro Rec":  recall_score(y_true, y_pred, average="macro"),
    }


def print_section_complete(section_number: int, section_name: str) -> None:
    """Print a section completion message."""
    print(f"✓ Section {section_number} complete — {section_name}")
