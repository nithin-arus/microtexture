"""
Determinism utilities for reproducible research.

Sets global random seeds across all libraries (Python, NumPy, scikit-learn, PyTorch, TensorFlow).
"""

import random
import numpy as np
import warnings

def set_global_seeds(seed):
    """
    Set global random seeds for reproducibility across all libraries.
    
    Parameters:
    -----------
    seed : int
        Random seed value to use for all libraries
        
    Sets seeds for:
    - Python random module
    - NumPy random
    - scikit-learn random state (via environment variable)
    - PyTorch (if available)
    - TensorFlow/Keras (if available)
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # scikit-learn uses numpy.random, but set environment for extra safety
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for CUDA
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # Older PyTorch versions
            torch.set_deterministic(True)
    except ImportError:
        pass  # PyTorch not available
    
    # TensorFlow/Keras (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Set deterministic operations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        # Disable GPU memory growth for reproducibility
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass  # GPU configuration may fail
    except ImportError:
        pass  # TensorFlow not available
    
    # Suppress warnings about determinism
    warnings.filterwarnings('ignore', category=UserWarning, message='.*deterministic.*')
    
    return seed

