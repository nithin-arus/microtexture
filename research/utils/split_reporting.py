"""
Split reporting utilities for class imbalance analysis.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional


def split_stats(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
               class_names: List[str]) -> Dict:
    """
    Compute class distribution statistics for train/val/test splits.
    
    Parameters:
    -----------
    y_train, y_val, y_test : np.ndarray
        Encoded labels for each split
    class_names : List[str]
        List of class names
        
    Returns:
    --------
    dict : Statistics with counts and proportions for each split
    """
    stats = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for split_name, y_split in [('train', y_train), ('val', y_val), ('test', y_test)]:
        total = len(y_split)
        counts = np.bincount(y_split, minlength=len(class_names))
        proportions = counts / total if total > 0 else np.zeros(len(class_names))
        
        stats[split_name] = {
            'total': int(total),
            'counts': {class_names[i]: int(counts[i]) for i in range(len(class_names))},
            'proportions': {class_names[i]: float(proportions[i]) for i in range(len(class_names))}
        }
    
    return stats

