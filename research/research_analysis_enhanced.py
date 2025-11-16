"""
Enhanced research_analysis.py with multi-seed, deep features, and fusion support.

This file contains the enhanced methods that can be integrated into research_analysis.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import hashlib
import json

def prepare_features_with_fusion(self, feature_mode='handcrafted', fusion_strategy='concatenate'):
    """
    Prepare features based on mode: handcrafted, deep-only, or hybrid.
    
    Parameters:
    -----------
    feature_mode : str
        'handcrafted', 'deep_only', or 'hybrid'
    fusion_strategy : str
        'concatenate', 'weighted', or 'attention' (for hybrid mode)
        
    Returns:
    --------
    tuple : (X, feature_names)
    """
    if feature_mode == 'handcrafted':
        # Use existing handcrafted features
        return self.X, self.feature_names
    
    elif feature_mode == 'deep_only':
        # Load deep features
        if self.deep_features_path and Path(self.deep_features_path).exists():
            deep_df = pd.read_csv(self.deep_features_path)
            print(f"Loaded deep features: {deep_df.shape}")
            
            # Merge with main dataframe on filename
            if 'filename' in self.features_df.columns and 'filename' in deep_df.columns:
                merged = self.features_df.merge(deep_df, on='filename', how='inner')
                print(f"Merged data: {merged.shape}")
                
                # Extract deep feature columns
                deep_feat_cols = [col for col in deep_df.columns 
                                 if col.startswith('feat_') and col not in ['filename', 'label']]
                X_deep = merged[deep_feat_cols].values
                
                # Scale deep features
                from sklearn.preprocessing import StandardScaler
                scaler_deep = StandardScaler()
                X_deep_scaled = scaler_deep.fit_transform(X_deep)
                
                return X_deep_scaled, deep_feat_cols
            else:
                raise ValueError("Cannot merge: filename column missing")
        else:
            raise ValueError(f"Deep features file not found: {self.deep_features_path}")
    
    elif feature_mode == 'hybrid':
        # Combine handcrafted and deep features
        X_handcrafted = self.X
        X_deep = None
        
        # Load deep features
        if self.deep_features_path and Path(self.deep_features_path).exists():
            deep_df = pd.read_csv(self.deep_features_path)
            
            # Merge
            if 'filename' in self.features_df.columns and 'filename' in deep_df.columns:
                merged = self.features_df.merge(deep_df, on='filename', how='inner')
                deep_feat_cols = [col for col in deep_df.columns 
                                 if col.startswith('feat_') and col not in ['filename', 'label']]
                X_deep = merged[deep_feat_cols].values
                
                # Scale both
                from sklearn.preprocessing import StandardScaler
                scaler_hand = StandardScaler()
                scaler_deep = StandardScaler()
                X_handcrafted_scaled = scaler_hand.fit_transform(X_handcrafted)
                X_deep_scaled = scaler_deep.fit_transform(X_deep)
                
                # Fuse features
                if fusion_strategy == 'concatenate':
                    X_fused = np.hstack([X_handcrafted_scaled, X_deep_scaled])
                    feature_names_fused = list(self.feature_names) + deep_feat_cols
                elif fusion_strategy == 'weighted':
                    # Simple weighted combination (can be enhanced)
                    X_fused = np.hstack([X_handcrafted_scaled * 0.5, X_deep_scaled * 0.5])
                    feature_names_fused = list(self.feature_names) + deep_feat_cols
                else:
                    # Default to concatenate
                    X_fused = np.hstack([X_handcrafted_scaled, X_deep_scaled])
                    feature_names_fused = list(self.feature_names) + deep_feat_cols
                
                return X_fused, feature_names_fused
            else:
                raise ValueError("Cannot merge for hybrid mode")
        else:
            raise ValueError(f"Deep features file not found for hybrid mode: {self.deep_features_path}")
    
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

def run_supervised_analysis_multiseed(self, test_size=0.2, val_size=0.1, cv_folds=5,
                                     n_seeds=5, seeds=None, feature_mode='handcrafted',
                                     fusion_strategy='concatenate', save_splits_dir=None,
                                     use_saved_splits=False, stratify_column='label'):
    """
    Run supervised analysis across multiple seeds with support for different feature modes.
    
    Parameters:
    -----------
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of data for validation
    cv_folds : int
        Number of cross-validation folds
    n_seeds : int
        Number of seeds to use
    seeds : List[int], optional
        List of seeds to use
    feature_mode : str
        'handcrafted', 'deep_only', or 'hybrid'
    fusion_strategy : str
        'concatenate', 'weighted', or 'attention' (for hybrid mode)
    save_splits_dir : str, optional
        Directory to save/load splits
    use_saved_splits : bool
        If True, try to load saved splits
    stratify_column : str
        Column name for stratification
        
    Returns:
    --------
    dict : Results with 'aggregated' and 'per_seed' keys
    """
    if self.X is None or self.y_encoded is None:
        raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
    
    # Prepare features based on mode
    X, feature_names = prepare_features_with_fusion(self, feature_mode, fusion_strategy)
    
    # Default seeds
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011][:n_seeds]
    
    print(f"Running supervised analysis with {feature_mode} features across {len(seeds)} seeds...")
    
    # Create config hash
    config_str = json.dumps({
        'test_size': test_size,
        'val_size': val_size,
        'feature_mode': feature_mode,
        'fusion_strategy': fusion_strategy,
        'stratify_column': stratify_column
    }, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    # Create split key
    class_list = self.class_names if self.class_names else sorted(np.unique(self.y_encoded).tolist())
    data_hash = hashlib.sha256(str(sorted(self.features_df['filename'].values if 'filename' in self.features_df.columns else range(len(self.features_df)))).encode()).hexdigest()[:16]
    split_key = self.sample_splitter.make_split_key(seeds[0], 'sample_aware', class_list, data_hash)
    
    per_seed_results = {}
    all_metrics = {model_name: [] for model_name in ['svm_linear', 'random_forest', 'mlp_classifier', 'xgboost', 'lightgbm']}
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"Seed {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'='*80}")
        
        # Set global seed
        set_global_seeds(seed)
        
        # Try to load splits if requested
        train_idx = None
        val_idx = None
        test_idx = None
        
        if use_saved_splits and save_splits_dir:
            split_dir = Path(save_splits_dir) / f"seed_{seed}"
            loaded = self.sample_splitter.load_split_indices(str(split_dir), split_key)
            if loaded:
                train_idx, val_idx, test_idx, metadata = loaded
                print(f"Loaded splits from {split_dir}")
        
        # Create splits if not loaded
        if train_idx is None:
            # Use sample-aware splitting
            train_idx, val_idx, test_idx, leak_summary = self.sample_splitter.split_by_samples(
                self.features_df, test_size=test_size, val_size=val_size,
                stratify_column=stratify_column, seed=seed
            )
            
            # Save splits if requested
            if save_splits_dir:
                split_dir = Path(save_splits_dir) / f"seed_{seed}"
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Compute label counts
                y_train_labels = self.y_encoded[train_idx]
                y_val_labels = self.y_encoded[val_idx]
                y_test_labels = self.y_encoded[test_idx]
                label_counts = {
                    'train': {cls: int(np.sum(y_train_labels == i)) for i, cls in enumerate(class_list)},
                    'val': {cls: int(np.sum(y_val_labels == i)) for i, cls in enumerate(class_list)},
                    'test': {cls: int(np.sum(y_test_labels == i)) for i, cls in enumerate(class_list)}
                }
                
                self.sample_splitter.save_split_indices(
                    train_idx, val_idx, test_idx,
                    str(split_dir), config_hash, seed, split_key, leak_summary, label_counts
                )
                print(f"Saved splits to {split_dir}")
        
        # Get splits for this seed
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = self.y_encoded[train_idx], self.y_encoded[val_idx], self.y_encoded[test_idx]
        
        # Scale features (if not already scaled in prepare_features_with_fusion)
        if feature_mode == 'handcrafted':
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # Already scaled in prepare_features_with_fusion
            X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
        
        # Train models with multi-seed support
        model_suite = SupervisedModelSuite(random_state=seed)
        seed_results = model_suite.train_all_models(
            X_train_scaled, X_test_scaled, y_train, y_test, feature_names, cv_folds, encoder=self.label_encoder
        )
        
        per_seed_results[seed] = seed_results
        
        # Collect metrics
        for model_name, results in seed_results.items():
            if model_name in all_metrics:
                all_metrics[model_name].append({
                    'accuracy': results['test_accuracy'],
                    'macro_f1': results.get('macro_f1', results.get('f1_score', 0)),
                    'macro_auc': results.get('macro_auc', results.get('test_auc'))
                })
    
    # Aggregate results
    aggregated_results = {}
    for model_name, seed_metrics in all_metrics.items():
        if not seed_metrics:
            continue
        
        accuracies = [m['accuracy'] for m in seed_metrics if m['accuracy'] is not None]
        macro_f1s = [m['macro_f1'] for m in seed_metrics if m['macro_f1'] is not None]
        macro_aucs = [m['macro_auc'] for m in seed_metrics if m['macro_auc'] is not None]
        
        if accuracies:
            acc_mean, acc_std, acc_ci_lower, acc_ci_upper = ci95(np.array(accuracies))
        else:
            acc_mean = acc_std = acc_ci_lower = acc_ci_upper = None
        
        if macro_f1s:
            f1_mean, f1_std, f1_ci_lower, f1_ci_upper = ci95(np.array(macro_f1s))
        else:
            f1_mean = f1_std = f1_ci_lower = f1_ci_upper = None
        
        auc_mean = auc_std = auc_ci_lower = auc_ci_upper = None
        if macro_aucs and all(a is not None for a in macro_aucs):
            auc_mean, auc_std, auc_ci_lower, auc_ci_upper = ci95(np.array(macro_aucs))
        
        aggregated_results[model_name] = {
            'accuracy': {'mean': acc_mean, 'std': acc_std, 'ci_lower': acc_ci_lower, 'ci_upper': acc_ci_upper},
            'macro_f1': {'mean': f1_mean, 'std': f1_std, 'ci_lower': f1_ci_lower, 'ci_upper': f1_ci_upper},
            'macro_auc': {'mean': auc_mean, 'std': auc_std, 'ci_lower': auc_ci_lower, 'ci_upper': auc_ci_upper} if auc_mean else None
        }
    
    # Save aggregated results
    results_summary = {
        'aggregated': aggregated_results,
        'per_seed': per_seed_results,
        'config': {
            'feature_mode': feature_mode,
            'fusion_strategy': fusion_strategy,
            'seeds': seeds,
            'config_hash': config_hash
        }
    }
    
    # Serialize results
    serialize_results(results_summary, self.config, str(self.output_dir / "results"))
    
    return results_summary

