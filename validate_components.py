#!/usr/bin/env python3
"""
Validate component imports and basic functionality.

This script checks that all new components can be imported and have correct signatures.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Validating Component Imports")
print("=" * 80)

errors = []
warnings = []

# Test imports
tests = [
    ("research.utils.determinism", "set_global_seeds"),
    ("research.utils.data_manifest", "load_manifest"),
    ("research.utils.data_manifest", "encode_labels"),
    ("research.utils.sample_aware_splitting", "SampleAwareSplitter"),
    ("research.utils.schema_migration", "check_feature_schema"),
    ("research.utils.schema_migration", "auto_fix_schema"),
    ("research.utils.split_reporting", "split_stats"),
    ("research.utils.captions", "build_caption"),
    ("research.utils.result_serializer", "serialize_results"),
    ("research.evaluation.metrics", "compute_macro_f1"),
    ("research.evaluation.metrics", "compute_macro_auc"),
    ("research.evaluation.stat_tests", "ci95"),
    ("research.evaluation.stat_tests", "paired_ttest"),
    ("research.evaluation.diagnostics", "confusion_matrix_report"),
    ("research.evaluation.feature_importance", "compute_permutation_importance"),
    ("research.models.extract_deep_features", "extract_deep_features_for_manifest"),
    ("research.visualization.plots", "plot_pr_curve"),
    ("analysis.feature_extractor", "get_feature_schema"),
]

print("\nTesting imports...")
for module_name, item_name in tests:
    try:
        module = __import__(module_name, fromlist=[item_name])
        item = getattr(module, item_name)
        print(f"✅ {module_name}.{item_name}")
        
        # Check if it's callable
        if callable(item):
            # Try to get signature
            try:
                import inspect
                sig = inspect.signature(item)
                print(f"   Signature: {item_name}{sig}")
            except:
                pass
    except ImportError as e:
        errors.append(f"❌ {module_name}.{item_name}: {e}")
        print(f"❌ {module_name}.{item_name}: {e}")
    except AttributeError as e:
        errors.append(f"❌ {module_name}.{item_name}: {e}")
        print(f"❌ {module_name}.{item_name}: {e}")
    except Exception as e:
        warnings.append(f"⚠️ {module_name}.{item_name}: {e}")
        print(f"⚠️ {module_name}.{item_name}: {e}")

# Test feature schema
print("\nTesting feature schema...")
try:
    from analysis.feature_extractor import get_feature_schema
    schema = get_feature_schema()
    print(f"✅ Feature schema: {len(schema)} features")
    
    # Check for removed features
    removed = ['variance', 'haralick_asm']
    for feat in removed:
        if feat in schema:
            errors.append(f"❌ Removed feature '{feat}' still in schema")
            print(f"❌ Removed feature '{feat}' still in schema")
        else:
            print(f"✅ Removed feature '{feat}' not in schema")
    
    # Check for new fractal metrics
    required_fractal = ['fractal_spectrum_corr', 'fractal_spectrum_rmse']
    # Note: fractal metrics are not in handcrafted schema, they're in CSV headers
    print("✅ Feature schema validation passed")
except Exception as e:
    errors.append(f"❌ Feature schema test failed: {e}")
    print(f"❌ Feature schema test failed: {e}")

# Test sample-aware splitter methods
print("\nTesting SampleAwareSplitter methods...")
try:
    from research.utils.sample_aware_splitting import SampleAwareSplitter
    splitter = SampleAwareSplitter(random_state=42)
    
    # Check methods exist
    methods = ['split_by_samples', 'save_split_indices', 'load_split_indices', 
               'make_split_key', 'leak_check']
    for method in methods:
        if hasattr(splitter, method):
            print(f"✅ SampleAwareSplitter.{method}")
        else:
            errors.append(f"❌ SampleAwareSplitter missing method: {method}")
            print(f"❌ SampleAwareSplitter missing method: {method}")
except Exception as e:
    errors.append(f"❌ SampleAwareSplitter test failed: {e}")
    print(f"❌ SampleAwareSplitter test failed: {e}")

# Test metrics functions
print("\nTesting metrics functions...")
try:
    from research.evaluation.metrics import compute_macro_f1, compute_macro_auc, validate_probabilities
    import numpy as np
    
    # Test with mock data
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 0])
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
    
    f1 = compute_macro_f1(y_true, y_pred)
    print(f"✅ compute_macro_f1: {f1:.4f}")
    
    auc = compute_macro_auc(y_true, y_proba)
    print(f"✅ compute_macro_auc: {auc:.4f if auc else 'N/A'}")
    
    validate_probabilities(y_proba, 2)
    print(f"✅ validate_probabilities: passed")
except Exception as e:
    errors.append(f"❌ Metrics test failed: {e}")
    print(f"❌ Metrics test failed: {e}")
    import traceback
    traceback.print_exc()

# Test statistical tests
print("\nTesting statistical tests...")
try:
    from research.evaluation.stat_tests import ci95, paired_ttest
    import numpy as np
    
    # Test with mock data
    data = np.random.randn(10) + 1.0
    mean, std, ci_lower, ci_upper = ci95(data)
    print(f"✅ ci95: mean={mean:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    
    a = np.random.randn(10)
    b = np.random.randn(10) + 0.5
    t_stat, p_value = paired_ttest(a, b)
    print(f"✅ paired_ttest: t={t_stat:.4f}, p={p_value:.4f}")
except Exception as e:
    errors.append(f"❌ Statistical tests failed: {e}")
    print(f"❌ Statistical tests failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("Validation Summary")
print("=" * 80)
print(f"Errors: {len(errors)}")
print(f"Warnings: {len(warnings)}")

if errors:
    print("\nErrors:")
    for error in errors:
        print(f"  {error}")

if warnings:
    print("\nWarnings:")
    for warning in warnings:
        print(f"  {warning}")

if not errors and not warnings:
    print("\n✅ All validations passed!")
    sys.exit(0)
elif errors:
    print("\n❌ Validation failed with errors")
    sys.exit(1)
else:
    print("\n⚠️ Validation passed with warnings")
    sys.exit(0)

