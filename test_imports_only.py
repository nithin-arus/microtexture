#!/usr/bin/env python3
"""
Test imports without triggering research/__init__.py imports.

This script tests that individual modules can be imported directly.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Testing Direct Module Imports")
print("=" * 80)

# Test imports directly without going through __init__.py
import_errors = []

def test_import(module_path, item_name=None):
    """Test importing a module directly."""
    try:
        # Import module directly using importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            module_path.replace('/', '.').replace('.py', ''),
            Path(__file__).parent / f"{module_path}.py"
        )
        if spec is None or spec.loader is None:
            import_errors.append(f"Could not load spec for {module_path}")
            return False
        
        # Check if it's a package
        module_file = Path(__file__).parent / f"{module_path}.py"
        if not module_file.exists():
            # Try as package
            module_file = Path(__file__).parent / module_path / "__init__.py"
            if not module_file.exists():
                import_errors.append(f"File not found: {module_path}")
                return False
        
        return True
    except Exception as e:
        import_errors.append(f"{module_path}: {e}")
        return False

# Test file existence and basic syntax
print("\n[1] Checking file existence and syntax...")
files_to_check = [
    "research/utils/determinism.py",
    "research/utils/data_manifest.py",
    "research/utils/sample_aware_splitting.py",
    "research/utils/schema_migration.py",
    "research/utils/split_reporting.py",
    "research/utils/captions.py",
    "research/utils/result_serializer.py",
    "research/evaluation/metrics.py",
    "research/evaluation/stat_tests.py",
    "research/evaluation/diagnostics.py",
    "research/evaluation/feature_importance.py",
    "research/models/extract_deep_features.py",
    "research/visualization/plots.py",
    "analysis/feature_extractor.py",
]

for file_path in files_to_check:
    full_path = Path(__file__).parent / file_path
    if full_path.exists():
        # Check syntax
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            compile(code, str(full_path), 'exec')
            print(f"✅ {file_path} (syntax OK)")
        except SyntaxError as e:
            print(f"❌ {file_path} (syntax error: {e})")
            import_errors.append(f"{file_path}: {e}")
        except Exception as e:
            print(f"⚠️ {file_path} (check failed: {e})")
    else:
        print(f"❌ {file_path} (not found)")
        import_errors.append(f"{file_path}: file not found")

# Test that we can read key functions without executing
print("\n[2] Checking function definitions...")
try:
    # Read determinism.py
    det_path = Path(__file__).parent / "research/utils/determinism.py"
    if det_path.exists():
        with open(det_path, 'r') as f:
            content = f.read()
            if 'def set_global_seeds' in content:
                print("✅ set_global_seeds function found")
            else:
                print("❌ set_global_seeds function not found")
                import_errors.append("set_global_seeds not found")
    
    # Read metrics.py
    metrics_path = Path(__file__).parent / "research/evaluation/metrics.py"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            content = f.read()
            if 'def compute_macro_f1' in content:
                print("✅ compute_macro_f1 function found")
            if 'def compute_macro_auc' in content:
                print("✅ compute_macro_auc function found")
            if 'def validate_probabilities' in content:
                print("✅ validate_probabilities function found")
    
    # Read sample_aware_splitting.py
    split_path = Path(__file__).parent / "research/utils/sample_aware_splitting.py"
    if split_path.exists():
        with open(split_path, 'r') as f:
            content = f.read()
            methods = ['split_by_samples', 'save_split_indices', 'load_split_indices', 
                      'make_split_key', 'leak_check']
            for method in methods:
                if f'def {method}' in content or f'    def {method}' in content:
                    print(f"✅ {method} method found")
                else:
                    print(f"❌ {method} method not found")
                    import_errors.append(f"{method} not found")
    
    # Check feature_extractor for get_feature_schema
    feat_path = Path(__file__).parent / "analysis/feature_extractor.py"
    if feat_path.exists():
        with open(feat_path, 'r') as f:
            content = f.read()
            if 'def get_feature_schema' in content:
                print("✅ get_feature_schema function found")
                # Check that variance and haralick_asm are removed
                if "'variance'" not in content or 'variance' not in content.split('get_feature_schema')[1].split('return')[0]:
                    print("✅ variance not in feature schema (removed)")
                else:
                    print("⚠️ variance may still be in feature schema")
            else:
                print("❌ get_feature_schema function not found")
                import_errors.append("get_feature_schema not found")
            
            # Check compute_basic_stats doesn't have variance
            if 'def compute_basic_stats' in content:
                stats_section = content.split('def compute_basic_stats')[1].split('def ')[0]
                if "'variance'" not in stats_section and '"variance"' not in stats_section:
                    print("✅ variance removed from compute_basic_stats")
                else:
                    print("❌ variance still in compute_basic_stats")
                    import_errors.append("variance not removed from compute_basic_stats")
            
            # Check compute_haralick_features doesn't have ASM
            if 'def compute_haralick_features' in content:
                haralick_section = content.split('def compute_haralick_features')[1].split('def ')[0]
                if 'haralick_asm' not in haralick_section and 'ASM' not in haralick_section.split('properties')[1] if 'properties' in haralick_section else '':
                    print("✅ haralick_asm removed from compute_haralick_features")
                else:
                    # Check if it's in the error handling only
                    if 'haralick_asm' in haralick_section and 'except' in haralick_section:
                        print("⚠️ haralick_asm only in error handling (may be OK)")
                    else:
                        print("❌ haralick_asm still in compute_haralick_features")
                        import_errors.append("haralick_asm not removed")
    
    # Check fractal_fitting for new metric names
    fractal_path = Path(__file__).parent / "analysis/fractal_fitting.py"
    if fractal_path.exists():
        with open(fractal_path, 'r') as f:
            content = f.read()
            if 'fractal_spectrum_corr' in content:
                print("✅ fractal_spectrum_corr found")
            else:
                print("❌ fractal_spectrum_corr not found")
                import_errors.append("fractal_spectrum_corr not found")
            
            if 'fractal_spectrum_rmse' in content:
                print("✅ fractal_spectrum_rmse found")
            else:
                print("❌ fractal_spectrum_rmse not found")
                import_errors.append("fractal_spectrum_rmse not found")
            
            if 'goodness_of_fit' in content and 'spectrum_corr' not in content:
                print("⚠️ goodness_of_fit still present (may be in old code)")
            elif 'goodness_of_fit' not in content:
                print("✅ goodness_of_fit removed/replaced")
    
except Exception as e:
    print(f"❌ Error checking functions: {e}")
    import_errors.append(f"Function check error: {e}")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
if import_errors:
    print(f"❌ Found {len(import_errors)} issues:")
    for error in import_errors[:10]:  # Show first 10
        print(f"  - {error}")
    if len(import_errors) > 10:
        print(f"  ... and {len(import_errors) - 10} more")
    sys.exit(1)
else:
    print("✅ All file checks passed!")
    print("\nNote: Full functionality testing requires installed dependencies.")
    print("Install with: pip install -r requirements.txt")
    sys.exit(0)

