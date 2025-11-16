"""
Diagnostic utilities for model evaluation.

Provides confusion matrix reporting and per-class metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from pathlib import Path
from typing import List, Optional


def confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str], output_path: str) -> None:
    """
    Generate confusion matrix report with CSV and normalized heatmap.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : List[str]
        Names of classes
    output_path : str
        Base path for output files (will create .csv and .png)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Save as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_path.with_suffix('.csv'))
    
    # Create normalized confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'})
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     class_names: List[str], output_path: str) -> pd.DataFrame:
    """
    Compute per-class precision, recall, and F1 score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : List[str]
        Names of classes
    output_path : str
        Path to save CSV file
        
    Returns:
    --------
    pd.DataFrame : Per-class metrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
    recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': np.bincount(y_true, minlength=len(class_names))
    })
    
    # Save to CSV
    metrics_df.to_csv(output_path, index=False)
    
    return metrics_df

