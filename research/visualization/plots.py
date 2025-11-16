"""
Plotting utilities for model evaluation.

Provides PR curves, ROC curves, and other diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from pathlib import Path
from typing import Optional


def plot_pr_curve(y_true: np.ndarray, scores: np.ndarray, 
                  path: str, title: str = "Precision-Recall Curve") -> None:
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    scores : np.ndarray
        Predicted scores or probabilities
    path : str
        Path to save plot
    title : str
        Plot title
    """
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                   path: str, title: str = "ROC Curve") -> None:
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Predicted scores or probabilities
    path : str
        Path to save plot
    title : str
        Plot title
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

