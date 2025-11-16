"""
Statistical tests for comparing model configurations.

Provides paired t-tests, Wilcoxon signed-rank tests, and confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test between two arrays.
    
    Parameters:
    -----------
    a, b : np.ndarray
        Arrays of values to compare (must have same length)
        
    Returns:
    --------
    tuple : (t_statistic, p_value)
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length: {len(a)} != {len(b)}")
    
    t_stat, p_value = stats.ttest_rel(a, b)
    return t_stat, p_value


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test between two arrays.
    
    Parameters:
    -----------
    a, b : np.ndarray
        Arrays of values to compare (must have same length)
        
    Returns:
    --------
    tuple : (statistic, p_value)
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length: {len(a)} != {len(b)}")
    
    statistic, p_value = stats.wilcoxon(a, b)
    return statistic, p_value


def ci95(values: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute 95% confidence interval for array of values.
    
    Parameters:
    -----------
    values : np.ndarray
        Array of values
        
    Returns:
    --------
    tuple : (mean, std, ci_lower, ci_upper)
        - mean: Sample mean
        - std: Sample standard deviation (ddof=1)
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
    """
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    n = len(values)
    
    # 95% CI using t-distribution
    t_critical = stats.t.ppf(0.975, df=n-1)  # Two-tailed, 95% CI
    margin = t_critical * std / np.sqrt(n)
    
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return mean, std, ci_lower, ci_upper


def compare_configurations(config1_results: Dict, config2_results: Dict, 
                          metric: str = 'macro_f1') -> pd.DataFrame:
    """
    Compare two configurations across multiple seeds using statistical tests.
    
    Parameters:
    -----------
    config1_results : dict
        Results from first configuration, structure: {seed: {model_name: {metric: value}}}
    config2_results : dict
        Results from second configuration, same structure
    metric : str
        Metric to compare (e.g., 'macro_f1', 'accuracy', 'macro_auc')
        
    Returns:
    --------
    pd.DataFrame : Comparison results with columns:
        - model: Model name
        - metric: Metric name
        - config1_mean: Mean for config1
        - config2_mean: Mean for config2
        - delta: Difference (config2 - config1)
        - p_value_ttest: p-value from paired t-test
        - p_value_wilcoxon: p-value from Wilcoxon test
        - ci_lower: Lower bound of 95% CI for delta
        - ci_upper: Upper bound of 95% CI for delta
    """
    # Get common seeds
    seeds1 = set(config1_results.keys())
    seeds2 = set(config2_results.keys())
    common_seeds = sorted(seeds1 & seeds2)
    
    if len(common_seeds) < 2:
        raise ValueError(f"Need at least 2 common seeds for comparison. Found: {common_seeds}")
    
    # Get common models
    models1 = set()
    models2 = set()
    for seed in common_seeds:
        if seed in config1_results:
            models1.update(config1_results[seed].keys())
        if seed in config2_results:
            models2.update(config2_results[seed].keys())
    common_models = sorted(models1 & models2)
    
    if len(common_models) == 0:
        raise ValueError("No common models found between configurations")
    
    # Compare each model
    comparison_data = []
    
    for model_name in common_models:
        # Extract metric values for each seed
        config1_values = []
        config2_values = []
        
        for seed in common_seeds:
            if (seed in config1_results and model_name in config1_results[seed] and
                seed in config2_results and model_name in config2_results[seed]):
                
                val1 = config1_results[seed][model_name].get(metric)
                val2 = config2_results[seed][model_name].get(metric)
                
                if val1 is not None and val2 is not None:
                    config1_values.append(val1)
                    config2_values.append(val2)
        
        if len(config1_values) < 2:
            continue  # Skip if not enough values
        
        config1_values = np.array(config1_values)
        config2_values = np.array(config2_values)
        
        # Compute statistics
        config1_mean, config1_std, _, _ = ci95(config1_values)
        config2_mean, config2_std, _, _ = ci95(config2_values)
        delta = config2_mean - config1_mean
        
        # Compute deltas for CI
        deltas = config2_values - config1_values
        _, _, ci_lower, ci_upper = ci95(deltas)
        
        # Statistical tests
        try:
            t_stat, p_value_ttest = paired_ttest(config1_values, config2_values)
        except:
            p_value_ttest = np.nan
        
        try:
            w_stat, p_value_wilcoxon = wilcoxon_signed_rank(config1_values, config2_values)
        except:
            p_value_wilcoxon = np.nan
        
        comparison_data.append({
            'model': model_name,
            'metric': metric,
            'config1_mean': config1_mean,
            'config1_std': config1_std,
            'config2_mean': config2_mean,
            'config2_std': config2_std,
            'delta': delta,
            'p_value_ttest': p_value_ttest,
            'p_value_wilcoxon': p_value_wilcoxon,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_seeds': len(config1_values)
        })
    
    return pd.DataFrame(comparison_data)

