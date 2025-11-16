"""
Feature importance and redundancy analysis.

Provides permutation importance, SHAP values (optional), and correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP importance will be disabled.")


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray, 
                                   n_repeats: int = 10, 
                                   random_state: int = 42,
                                   feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute permutation importance for a trained model.
    
    Parameters:
    -----------
    model : sklearn-like model
        Trained model with predict or predict_proba method
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    n_repeats : int
        Number of times to permute each feature
    random_state : int
        Random seed
    feature_names : List[str], optional
        Names of features
        
    Returns:
    --------
    pd.DataFrame : Importance scores with columns: feature, importance_mean, importance_std
    """
    # Compute permutation importance
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    
    # Create DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df


def compute_shap_importance(model, X: np.ndarray, feature_names: Optional[List[str]] = None,
                           output_dir: Optional[str] = None, 
                           sample_size: Optional[int] = 1000) -> Optional[pd.DataFrame]:
    """
    Compute SHAP values for feature importance (tree models only).
    
    Parameters:
    -----------
    model : sklearn-like model
        Trained tree model (RandomForest, XGBoost, LightGBM, etc.)
    X : np.ndarray
        Feature matrix
    feature_names : List[str], optional
        Names of features
    output_dir : str, optional
        Directory to save SHAP plots
    sample_size : int, optional
        Number of samples to use for SHAP computation (for speed)
        
    Returns:
    --------
    pd.DataFrame or None : SHAP importance scores, or None if SHAP unavailable or model not supported
    """
    if not SHAP_AVAILABLE:
        return None
    
    # Sample data if too large
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    try:
        # Try TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # Multi-class: average absolute SHAP values across classes
            shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            # Binary or single output
            shap_abs = np.abs(shap_values)
        
        # Compute mean importance per feature
        importance = np.mean(shap_abs, axis=0)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': importance
        })
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        
        # Save plot if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Bar plot of mean |SHAP values|
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(importance_df))
            top_features = importance_df.head(top_n)
            plt.barh(range(top_n), top_features['shap_importance'].values[::-1])
            plt.yticks(range(top_n), top_features['feature'].values[::-1])
            plt.xlabel('Mean |SHAP Value|', fontsize=12)
            plt.title('Top Feature Importance (SHAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'shap_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save CSV
            importance_df.to_csv(output_path / 'shap_importance.csv', index=False)
        
        return importance_df
        
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return None


def analyze_feature_correlations(X: np.ndarray, feature_names: Optional[List[str]] = None,
                                threshold: float = 0.95,
                                output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze feature correlations and identify highly correlated pairs.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    feature_names : List[str], optional
        Names of features
    threshold : float
        Correlation threshold for identifying redundant features
    output_path : str, optional
        Path to save correlation analysis CSV
        
    Returns:
    --------
    pd.DataFrame : Highly correlated feature pairs
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Compute correlation matrix
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()
    
    # Find highly correlated pairs
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value >= threshold:
                highly_correlated.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    corr_df = pd.DataFrame(highly_correlated)
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    # Highlight removed features
    removed_features = ['variance', 'haralick_asm']
    if corr_df is not None and len(corr_df) > 0:
        # Check if any removed features appear in correlations
        for feat in removed_features:
            if feat in feature_names:
                print(f"Note: Removed feature '{feat}' would have high correlation with other features")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corr_df.to_csv(output_path, index=False)
        
        # Also save full correlation matrix
        corr_matrix.to_csv(output_path.parent / 'feature_correlations_full.csv')
    
    return corr_df

