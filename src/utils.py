"""Utility functions for Forest Cover Type Classifier."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Micro F1": f1_score(y_true, y_pred, average="micro"),
        "Macro Prec": precision_score(y_true, y_pred, average="macro"),
        "Macro Rec": recall_score(y_true, y_pred, average="macro"),
    }

def print_section_complete(section_number, section_name):
    print(f"Section {section_number} complete — {section_name}")
