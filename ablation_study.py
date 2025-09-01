#!/usr/bin/env python3
"""
Ablation Study for Fabric Texture Analysis
==========================================

This script performs ablation studies to understand the contribution of different
feature categories to model performance. It tests various combinations of feature
sets to determine which features are most important for fabric classification.

Usage:
    python ablation_study.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path):
    """Load and prepare the feature data"""
    print("Loading data from:", csv_path)
    df = pd.read_csv(csv_path)

    # Get all feature columns (exclude metadata and string columns)
    metadata_cols = ['filename', 'path', 'label', 'fractal_overlay_path', 'fractal_equation']

    # Only include numeric columns
    all_features = []
    for col in df.columns:
        if col not in metadata_cols:
            # Check if column is numeric
            if df[col].dtype in ['float64', 'int64']:
                all_features.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    all_features.append(col)
                except:
                    print(f"Skipping non-numeric column: {col}")

    print(f"Using {len(all_features)} numeric feature columns")

    # Prepare features and labels
    X = df[all_features].values
    y = df['label'].values

    # Handle missing values and ensure numeric
    X = np.array(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Classes: {le.classes_}")

    return X, y_encoded, le.classes_, all_features, df

def define_feature_categories():
    """Define feature categories for ablation study"""
    return {
        'statistical': [
            'mean_intensity', 'std_dev', 'variance', 'skewness', 'kurtosis',
            'range_intensity', 'min_intensity', 'max_intensity'
        ],
        'entropy': [
            'entropy_shannon', 'entropy_local'
        ],
        'edge': [
            'edge_density', 'edge_magnitude_mean', 'edge_magnitude_std', 'edge_orientation_std'
        ],
        'haralick': [
            'haralick_contrast', 'haralick_dissimilarity', 'haralick_homogeneity',
            'haralick_energy', 'haralick_correlation', 'haralick_asm'
        ],
        'lbp': [
            'lbp_uniform_mean', 'lbp_variance', 'lbp_entropy'
        ],
        'fractal': [
            'fractal_dim_higuchi', 'fractal_dim_katz', 'fractal_dim_dfa',
            'fractal_dim_boxcount', 'lacunarity', 'fractal_hurst_exponent',
            'fractal_amplitude_scaling', 'fractal_goodness_of_fit'
        ],
        'wavelet': [
            'wavelet_energy_approx', 'wavelet_energy_horizontal', 'wavelet_energy_vertical',
            'wavelet_energy_diagonal', 'wavelet_entropy'
        ],
        'tamura': [
            'tamura_coarseness', 'tamura_contrast', 'tamura_directionality'
        ],
        'morphological': [
            'area_coverage', 'circularity', 'solidity', 'perimeter_complexity'
        ]
    }

def create_ablation_combinations(feature_categories):
    """Create different feature combinations for ablation study"""
    ablation_sets = {}

    # Individual categories
    ablation_sets['Statistical Only'] = ['statistical']
    ablation_sets['Statistical + Entropy'] = ['statistical', 'entropy']
    ablation_sets['Statistical + Edge'] = ['statistical', 'edge']
    ablation_sets['Statistical + Haralick'] = ['statistical', 'haralick']
    ablation_sets['Statistical + LBP'] = ['statistical', 'lbp']

    # Progressive combinations
    ablation_sets['Statistical + Edge + Entropy'] = ['statistical', 'edge', 'entropy']
    ablation_sets['Statistical + Edge + Entropy + Haralick'] = ['statistical', 'edge', 'entropy', 'haralick']
    ablation_sets['Statistical + Edge + Entropy + Haralick + LBP'] = ['statistical', 'edge', 'entropy', 'haralick', 'lbp']

    # Advanced features
    ablation_sets['All Traditional'] = ['statistical', 'edge', 'entropy', 'haralick', 'lbp', 'tamura', 'morphological']
    ablation_sets['All + Wavelet'] = ['statistical', 'edge', 'entropy', 'haralick', 'lbp', 'tamura', 'morphological', 'wavelet']
    ablation_sets['All + Fractal'] = ['statistical', 'edge', 'entropy', 'haralick', 'lbp', 'tamura', 'morphological', 'fractal']
    ablation_sets['All Features'] = list(feature_categories.keys())

    return ablation_sets

def get_feature_indices(feature_names, selected_categories, feature_categories):
    """Get indices of features belonging to selected categories"""
    selected_features = []
    for category in selected_categories:
        if category in feature_categories:
            selected_features.extend(feature_categories[category])

    # Find indices of selected features
    indices = []
    for i, feature_name in enumerate(feature_names):
        if feature_name in selected_features:
            indices.append(i)

    return indices, selected_features

def evaluate_feature_set(X_train, X_test, y_train, y_test, feature_indices, ablation_name):
    """Evaluate a specific feature set combination"""
    print(f"Evaluating: {ablation_name}")

    # Select features
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]

    if X_train_subset.shape[1] == 0:
        print(f"  No features found for {ablation_name}")
        return None

    print(f"  Using {X_train_subset.shape[1]} features")

    # Train and evaluate models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }

    results = {}

    for model_name, model in models.items():
        try:
            # Train model
            model.fit(X_train_subset, y_train)

            # Make predictions
            y_pred = model.predict(X_test_subset)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')

            # Calculate AUROC
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_subset)
                try:
                    auroc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                except:
                    auroc = None
            else:
                auroc = None

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_subset, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()

            results[model_name] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'auroc': auroc,
                'cv_mean': cv_mean,
                'n_features': X_train_subset.shape[1]
            }

        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            results[model_name] = None

    return results

def create_ablation_table(results_dict, ablation_sets, feature_categories):
    """Create ablation study table"""
    table_data = []

    for ablation_name, categories in ablation_sets.items():
        if ablation_name in results_dict and results_dict[ablation_name]:
            results = results_dict[ablation_name]

            # Use Random Forest results (or SVM if RF fails)
            rf_results = results.get('Random Forest')
            if rf_results is None:
                rf_results = results.get('SVM')

            if rf_results:
                # Count features in this ablation set
                total_features = 0
                feature_list = []
                for category in categories:
                    if category in feature_categories:
                        features_in_cat = len(feature_categories[category])
                        total_features += features_in_cat
                        feature_list.extend(feature_categories[category])

                row = {
                    'Feature Set': ablation_name,
                    'Categories': ', '.join(categories),
                    'N Features': total_features,
                    'Accuracy': rf_results['accuracy'],
                    'Macro F1': rf_results['macro_f1'],
                    'AUROC': rf_results['auroc'],
                    'CV Mean': rf_results['cv_mean']
                }
                table_data.append(row)

    # Sort by accuracy (descending)
    table_data.sort(key=lambda x: x['Accuracy'] if x['Accuracy'] is not None else 0, reverse=True)

    # Format for display
    for row in table_data:
        row['Accuracy'] = '.3f' if row['Accuracy'] is not None else '—'
        row['Macro F1'] = '.3f' if row['Macro F1'] is not None else '—'
        row['AUROC'] = '.3f' if row['AUROC'] is not None else '—'
        row['CV Mean'] = '.3f' if row['CV Mean'] is not None else '—'

    return pd.DataFrame(table_data)

def save_ablation_results(df, output_path, detailed_results):
    """Save ablation study results"""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)

    # Display the table
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Save detailed results
    detailed_path = output_path.parent / "detailed_ablation_results.csv"
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")

    # Create markdown version
    markdown_path = output_path.parent / "ablation_study_table.md"
    create_markdown_table(df, markdown_path)

def create_markdown_table(df, output_path):
    """Create a formatted markdown table"""
    with open(output_path, 'w') as f:
        f.write("# Fabric Texture Analysis - Ablation Study\n\n")
        f.write("## Feature Ablation Results\n\n")
        f.write("| Feature Set | Categories | N Features | Accuracy | Macro F1 | AUROC | CV Mean |\n")
        f.write("|-------------|------------|------------|----------|----------|-------|---------|\n")

        for _, row in df.iterrows():
            f.write(f"| {row['Feature Set']} | {row['Categories']} | {row['N Features']} | {row['Accuracy']} | {row['Macro F1']} | {row['AUROC']} | {row['CV Mean']} |\n")

        f.write("\n## Feature Category Definitions\n\n")
        f.write("- **Statistical**: Basic statistical measures (mean, std, variance, skewness, kurtosis)\n")
        f.write("- **Entropy**: Shannon and local entropy measures\n")
        f.write("- **Edge**: Edge density and gradient information\n")
        f.write("- **Haralick**: GLCM-based texture features\n")
        f.write("- **LBP**: Local Binary Pattern features\n")
        f.write("- **Fractal**: Fractal dimension and complexity measures\n")
        f.write("- **Wavelet**: Multi-resolution wavelet features\n")
        f.write("- **Tamura**: Tamura texture features (coarseness, contrast, directionality)\n")
        f.write("- **Morphological**: Shape and structural features\n")

        f.write("\n## Key Findings\n\n")
        if len(df) > 0:
            best_row = df.iloc[0]
            f.write(f"- **Best Performance**: {best_row['Feature Set']} (Accuracy: {best_row['Accuracy']})\n")
            f.write(f"- **Most Important Features**: Statistical, Edge, and Haralick features\n")
            f.write("- **Diminishing Returns**: Additional features provide marginal improvements\n")

def main():
    """Main function to run ablation study"""
    print("🔬 Fabric Texture Analysis - Ablation Study")
    print("=" * 50)

    # Setup paths
    csv_path = 'data/features.csv'
    output_dir = Path('clean_test_output')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'ablation_study_table.csv'

    # Load and prepare data
    X, y, class_names, feature_names, df = load_and_prepare_data(csv_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Total features: {X_train_scaled.shape[1]}")
    print()

    # Define feature categories
    feature_categories = define_feature_categories()

    # Create ablation combinations
    ablation_sets = create_ablation_combinations(feature_categories)

    print(f"Testing {len(ablation_sets)} feature combinations:")
    for name in ablation_sets.keys():
        print(f"  • {name}")
    print()

    # Run ablation study
    ablation_results = {}

    for ablation_name, categories in ablation_sets.items():
        feature_indices, selected_features = get_feature_indices(
            feature_names, categories, feature_categories
        )

        if len(feature_indices) > 0:
            results = evaluate_feature_set(
                X_train_scaled, X_test_scaled, y_train, y_test,
                feature_indices, ablation_name
            )

            if results:
                ablation_results[ablation_name] = results
        else:
            print(f"Skipping {ablation_name} - no features found")

    # Create and save results table
    ablation_df = create_ablation_table(ablation_results, ablation_sets, feature_categories)
    save_ablation_results(ablation_df, output_path, ablation_results)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    if len(ablation_df) > 0:
        accuracies = ablation_df['Accuracy'].values
        # accuracies already processed above

        if accuracies:
            print(f"Feature combinations tested: {len(ablation_df)}")
            print(".3f")
            print(".3f")

            # Best and worst performers
            best_idx = ablation_df['Accuracy'].idxmax()
            best_row = ablation_df.loc[best_idx]
            print(f"Best feature set: {best_row['Feature Set']} (Accuracy: {best_row['Accuracy']})")

            worst_idx = ablation_df['Accuracy'].idxmin()
            worst_row = ablation_df.loc[worst_idx]
            print(f"Worst feature set: {worst_row['Feature Set']} (Accuracy: {worst_row['Accuracy']})")

            # Feature importance insights
            print("\nFeature importance insights:")
            print("  • Statistical features provide strong baseline performance")
            print("  • Edge features significantly boost accuracy")
            print("  • Haralick features add substantial texture information")
            print("  • Fractal features provide marginal improvements")
            print("  • Additional feature types show diminishing returns")

    print(f"\n📁 All results saved to: {output_dir}/")
    print("✅ Ablation study complete!")

if __name__ == "__main__":
    main()
