"""
Standardized metric computation for model evaluation.

Provides consistent metric functions with proper validation and error handling.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional, Tuple


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    float : Macro-averaged F1 score
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """
    Compute macro-averaged ROC-AUC score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (integer-encoded)
    y_prob : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
        
    Returns:
    --------
    float or None : Macro-averaged ROC-AUC, or None if computation fails
        
    Raises:
    -------
    ValueError: If y_prob is None or invalid shape
    """
    if y_prob is None:
        raise ValueError("y_prob cannot be None for AUC computation")
    
    if len(y_prob.shape) != 2:
        raise ValueError(f"y_prob must be 2D array (n_samples, n_classes), got shape {y_prob.shape}")
    
    n_classes = y_prob.shape[1]
    n_samples = y_prob.shape[0]
    
    if len(y_true) != n_samples:
        raise ValueError(f"y_true length ({len(y_true)}) must match y_prob n_samples ({n_samples})")
    
    # Validate probabilities
    validate_probabilities(y_prob, n_classes)
    
    # Determine if binary or multi-class
    unique_labels = np.unique(y_true)
    n_unique = len(unique_labels)
    
    if n_unique == 2:
        # Binary classification: use positive class probabilities
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class probabilities but binary labels - use one-vs-rest
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    elif n_unique > 2:
        # Multi-class: use one-vs-rest macro averaging
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError as e:
            # Some classes might not be present in y_true
            print(f"Warning: AUC computation failed: {e}")
            return None
    else:
        raise ValueError(f"Invalid number of unique labels: {n_unique}")
    
    return auc


def validate_probabilities(y_prob: np.ndarray, K: int) -> bool:
    """
    Validate that probability array is properly formatted.
    
    Parameters:
    -----------
    y_prob : np.ndarray
        Predicted probabilities
    K : int
        Expected number of classes
        
    Returns:
    --------
    bool : True if validation passes
        
    Raises:
    -------
    ValueError: If probabilities are invalid (NaN, wrong shape, don't sum to ~1.0)
    """
    if y_prob is None:
        raise ValueError("y_prob cannot be None")
    
    # Check shape
    if len(y_prob.shape) != 2:
        raise ValueError(f"y_prob must be 2D array (n_samples, n_classes), got shape {y_prob.shape}")
    
    if y_prob.shape[1] != K:
        raise ValueError(f"y_prob must have {K} columns (one per class), got {y_prob.shape[1]}")
    
    # Check for NaN
    if np.any(np.isnan(y_prob)):
        raise ValueError("y_prob contains NaN values")
    
    # Check that rows sum to approximately 1.0
    row_sums = np.sum(y_prob, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"y_prob rows must sum to ~1.0, but found sums in range [{row_sums.min():.6f}, {row_sums.max():.6f}]")
    
    # Check that probabilities are non-negative
    if np.any(y_prob < 0):
        raise ValueError("y_prob contains negative values")
    
    # Check that probabilities are <= 1.0
    if np.any(y_prob > 1.0 + 1e-6):
        raise ValueError(f"y_prob contains values > 1.0, max = {y_prob.max():.6f}")
    
    return True

