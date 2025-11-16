#!/usr/bin/env python3
"""
Test script for new components.

Tests all newly implemented components on the actual dataset.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Testing New Components")
print("=" * 80)

# Test 1: Schema Migration
print("\n[Test 1] Schema Migration")
print("-" * 80)
try:
    from research.utils.schema_migration import check_feature_schema, auto_fix_schema
    from analysis.feature_extractor import get_feature_schema
    
    # Load data
    df = pd.read_csv('data/features.csv')
    print(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Check schema
    expected_schema = get_feature_schema()
    missing, extra, legacy = check_feature_schema(df.columns.tolist(), expected_schema)
    
    print(f"Missing features: {len(missing)}")
    print(f"Extra features: {len(extra)}")
    print(f"Legacy features: {legacy}")
    
    # Auto-fix schema
    df_fixed = auto_fix_schema(df.copy(), expected_schema, log_migrations=True)
    print(f"Fixed data: {df_fixed.shape[0]} samples, {df_fixed.shape[1]} columns")
    print("✅ Schema migration test passed")
except Exception as e:
    print(f"❌ Schema migration test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Sample-Aware Splitting
print("\n[Test 2] Sample-Aware Splitting")
print("-" * 80)
try:
    from research.utils.sample_aware_splitting import SampleAwareSplitter
    from research.utils.determinism import set_global_seeds
    
    # Use fixed dataframe
    df_test = df_fixed.copy() if 'df_fixed' in locals() else df.copy()
    
    # Set seed
    set_global_seeds(42)
    
    # Create splitter
    splitter = SampleAwareSplitter(random_state=42)
    
    # Split by samples
    train_idx, val_idx, test_idx, leak_summary = splitter.split_by_samples(
        df_test,
        test_size=0.2,
        val_size=0.1,
        stratify_column='label',
        seed=42
    )
    
    print(f"Train: {len(train_idx)} rows")
    print(f"Val: {len(val_idx)} rows")
    print(f"Test: {len(test_idx)} rows")
    print(f"Leak check: {'✓ PASSED' if leak_summary['is_valid'] else '✗ FAILED'}")
    print(f"Train samples: {leak_summary['n_train_samples']}")
    print(f"Val samples: {leak_summary['n_val_samples']}")
    print(f"Test samples: {leak_summary['n_test_samples']}")
    
    # Test saving and loading
    split_dir = 'test_output/splits'
    config_hash = 'test_hash_123'
    split_key = splitter.make_split_key(42, 'sample_aware', 
                                       sorted(df_test['label'].unique().tolist()),
                                       'test_hash')
    
    saved_path = splitter.save_split_indices(
        train_idx, val_idx, test_idx,
        split_dir, config_hash, 42, split_key, leak_summary
    )
    print(f"Saved splits to: {saved_path}")
    
    # Test loading
    loaded = splitter.load_split_indices(split_dir, split_key)
    if loaded:
        train_loaded, val_loaded, test_loaded, metadata = loaded
        print(f"Loaded splits: train={len(train_loaded)}, val={len(val_loaded)}, test={len(test_loaded)}")
        assert len(train_loaded) == len(train_idx), "Train split mismatch"
        assert len(val_loaded) == len(val_idx), "Val split mismatch"
        assert len(test_loaded) == len(test_idx), "Test split mismatch"
        print("✅ Split persistence test passed")
    else:
        print("⚠️ Could not load splits (may be expected)")
    
    print("✅ Sample-aware splitting test passed")
except Exception as e:
    print(f"❌ Sample-aware splitting test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Label Encoding
print("\n[Test 3] Label Encoding")
print("-" * 80)
try:
    from research.utils.data_manifest import encode_labels
    
    # Create labels from dataframe
    labels = df_test['label'].values if 'df_test' in locals() else df['label'].values
    
    # Encode labels
    y_encoded, encoder, class_names = encode_labels(labels)
    
    print(f"Encoded labels: {len(y_encoded)} samples")
    print(f"Class names: {class_names}")
    print(f"Unique encoded values: {np.unique(y_encoded)}")
    
    assert len(y_encoded) == len(labels), "Label length mismatch"
    assert len(class_names) == len(np.unique(labels)), "Class count mismatch"
    
    print("✅ Label encoding test passed")
except Exception as e:
    print(f"❌ Label encoding test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Metrics
print("\n[Test 4] Standardized Metrics")
print("-" * 80)
try:
    from research.evaluation.metrics import compute_macro_f1, compute_macro_auc, validate_probabilities
    from sklearn.ensemble import RandomForestClassifier
    
    # Prepare data
    df_test = df_fixed.copy() if 'df_fixed' in locals() else df.copy()
    feature_cols = [col for col in df_test.columns if col not in ['filename', 'path', 'label', 
                                                                  'fractal_overlay_path', 'fractal_equation']]
    
    # Remove legacy columns if present
    for col in ['variance', 'haralick_asm', 'fractal_goodness_of_fit']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    X = df_test[feature_cols].values
    y = df_test['label'].values
    
    # Encode labels
    if 'encoder' not in locals():
        y_encoded, encoder, class_names = encode_labels(y)
    else:
        y_encoded = encoder.transform(y)
        class_names = encoder.classes_.tolist()
    
    # Split data
    if 'train_idx' not in locals():
        from research.utils.sample_aware_splitting import SampleAwareSplitter
        splitter = SampleAwareSplitter(random_state=42)
        train_idx, val_idx, test_idx, _ = splitter.split_by_samples(
            df_test, test_size=0.2, val_size=0.1, stratify_column='label', seed=42
        )
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    # Train a simple model
    print("Training RandomForest for metrics test...")
    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Test metrics
    macro_f1 = compute_macro_f1(y_test, y_pred)
    print(f"Macro F1: {macro_f1:.4f}")
    
    # Validate probabilities
    validate_probabilities(y_proba, len(class_names))
    print("Probability validation passed")
    
    # Test AUC
    macro_auc = compute_macro_auc(y_test, y_proba)
    print(f"Macro AUC: {macro_auc:.4f if macro_auc else 'N/A'}")
    
    print("✅ Metrics test passed")
except Exception as e:
    print(f"❌ Metrics test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Statistical Tests
print("\n[Test 5] Statistical Tests")
print("-" * 80)
try:
    from research.evaluation.stat_tests import ci95, paired_ttest
    
    # Create sample data
    a = np.random.randn(10) + 1.0
    b = np.random.randn(10) + 1.1
    
    # Test CI95
    mean, std, ci_lower, ci_upper = ci95(a)
    print(f"CI95 for sample a: mean={mean:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Test paired t-test
    t_stat, p_value = paired_ttest(a, b)
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    print("✅ Statistical tests passed")
except Exception as e:
    print(f"❌ Statistical tests failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Feature Importance
print("\n[Test 6] Feature Importance")
print("-" * 80)
try:
    from research.evaluation.feature_importance import compute_permutation_importance, analyze_feature_correlations
    
    # Use model from previous test
    if 'model' in locals() and 'X_test' in locals() and 'y_test' in locals():
        # Compute permutation importance
        importance_df = compute_permutation_importance(
            model, X_test, y_test, n_repeats=3, random_state=42,
            feature_names=feature_cols[:10]  # Limit to first 10 for speed
        )
        print(f"Computed importance for {len(importance_df)} features")
        print(f"Top 5 features:\n{importance_df.head()}")
        
        # Test correlation analysis
        corr_df = analyze_feature_correlations(
            X_test[:, :10], feature_names=feature_cols[:10], threshold=0.9
        )
        print(f"Found {len(corr_df)} highly correlated pairs")
        
        print("✅ Feature importance test passed")
    else:
        print("⚠️ Skipping feature importance (requires trained model)")
except Exception as e:
    print(f"❌ Feature importance test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Split Reporting
print("\n[Test 7] Split Reporting")
print("-" * 80)
try:
    from research.utils.split_reporting import split_stats
    
    # Use splits from previous test
    if 'train_idx' in locals() and 'val_idx' in locals() and 'test_idx' in locals():
        y_train_split = y_encoded[train_idx]
        y_val_split = y_encoded[val_idx]
        y_test_split = y_encoded[test_idx]
        
        stats = split_stats(y_train_split, y_val_split, y_test_split, class_names)
        
        print("Split statistics:")
        for split_name in ['train', 'val', 'test']:
            print(f"  {split_name}: {stats[split_name]['total']} samples")
            print(f"    Classes: {list(stats[split_name]['counts'].keys())[:5]}...")
        
        print("✅ Split reporting test passed")
    else:
        print("⚠️ Skipping split reporting (requires splits)")
except Exception as e:
    print(f"❌ Split reporting test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Multi-Seed Training
print("\n[Test 8] Multi-Seed Training")
print("-" * 80)
try:
    from research.models.supervised_models import SupervisedModelSuite
    from research.utils.determinism import set_global_seeds
    
    # Use data from previous tests
    if 'X_train' in locals() and 'X_test' in locals() and 'y_train' in locals() and 'y_test' in locals():
        # Test single seed first
        set_global_seeds(42)
        model_suite = SupervisedModelSuite(random_state=42)
        
        # Train just one model for speed
        print("Training RandomForest (single seed)...")
        results = model_suite.train_single_model(
            'random_forest', X_train, X_test, y_train, y_test, cv_folds=3, encoder=encoder
        )
        
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Macro AUC: {results.get('macro_auc', 'N/A')}")
        
        print("✅ Multi-seed training test passed (single seed)")
    else:
        print("⚠️ Skipping multi-seed training (requires data)")
except Exception as e:
    print(f"❌ Multi-seed training test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)

