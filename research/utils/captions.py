"""
Caption generation utilities for plots and tables.

Generates descriptive captions for visualizations and results tables.
"""

from typing import Dict, List, Optional


def build_caption(config: Dict, metrics_def: Optional[Dict] = None) -> str:
    """
    Build a descriptive caption for plots/tables.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
            - feature_sets: List of feature set names (e.g., ['handcrafted', 'deep'])
            - models: List of model names
            - splits: Split strategy (e.g., 'sample_aware')
            - n_seeds: Number of seeds used
            - fusion_strategy: Fusion strategy if applicable (e.g., 'concatenate')
            - test_size: Test set proportion
            - val_size: Validation set proportion
    metrics_def : dict, optional
        Metric definitions with keys:
            - metrics: List of metric names
            - averaging: Averaging scheme (e.g., 'macro', 'weighted')
            
    Returns:
    --------
    str : Formatted caption string
    """
    caption_parts = []
    
    # Feature sets
    if 'feature_sets' in config:
        feature_sets = config['feature_sets']
        if len(feature_sets) == 1:
            caption_parts.append(f"Features: {feature_sets[0]}")
        else:
            caption_parts.append(f"Features: {' + '.join(feature_sets)}")
            if 'fusion_strategy' in config:
                caption_parts.append(f"Fusion: {config['fusion_strategy']}")
    
    # Models
    if 'models' in config:
        models = config['models']
        if len(models) <= 3:
            caption_parts.append(f"Models: {', '.join(models)}")
        else:
            caption_parts.append(f"Models: {', '.join(models[:3])} + {len(models)-3} more")
    
    # Splits
    if 'splits' in config:
        caption_parts.append(f"Splits: {config['splits']}")
        if 'test_size' in config and 'val_size' in config:
            caption_parts.append(f"({config['test_size']:.0%} test, {config['val_size']:.0%} val)")
    
    # Seeds
    if 'n_seeds' in config:
        caption_parts.append(f"Seeds: {config['n_seeds']}")
    
    # Metrics
    if metrics_def:
        if 'metrics' in metrics_def:
            metrics = metrics_def['metrics']
            caption_parts.append(f"Metrics: {', '.join(metrics)}")
        if 'averaging' in metrics_def:
            caption_parts.append(f"Averaging: {metrics_def['averaging']}")
    
    caption = " | ".join(caption_parts)
    return caption

