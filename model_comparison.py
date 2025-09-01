#!/usr/bin/env python3
"""
Model Comparison Script for Fabric Texture Analysis
==================================================

This script compares different machine learning models on fabric texture classification
and anomaly detection tasks, generating a comprehensive comparison table with key metrics.

Usage:
    python model_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available")
except ImportError as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost not available - skipping XGBoost model: {e}")
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost error - skipping XGBoost model: {e}")

import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path):
    """Load and prepare the feature data"""
    print("Loading data from:", csv_path)
    df = pd.read_csv(csv_path)

    # Get feature columns (exclude metadata and string columns)
    metadata_cols = ['filename', 'path', 'label', 'fractal_overlay_path', 'fractal_equation']

    # Only include numeric columns
    feature_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            # Check if column is numeric
            if df[col].dtype in ['float64', 'int64']:
                feature_cols.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    feature_cols.append(col)
                except:
                    print(f"Skipping non-numeric column: {col}")

    print(f"Using {len(feature_cols)} numeric feature columns")
    print(f"Features: {feature_cols[:5]}...")  # Show first 5

    # Prepare features and labels
    X = df[feature_cols].values
    y = df['label'].values

    # Handle missing values and ensure numeric
    X = np.array(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)

    # Encode labels for multi-class classification
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Classes: {le.classes_}")

    return X, y_encoded, le.classes_, feature_cols

def create_binary_labels(y, class_names, positive_class=None):
    """Create binary labels for anomaly detection evaluation"""
    if positive_class is None:
        # Use the first class as "normal" and others as "anomaly"
        positive_class = class_names[0]

    # Create binary labels (0 = normal, 1 = anomaly)
    y_binary = np.where(np.array(class_names)[y] == positive_class, 0, 1)

    return y_binary

def evaluate_supervised_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a supervised learning model"""
    print(f"Evaluating {model_name}...")

    try:
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        macro_precision = precision_score(y_test, y_pred, average='macro')
        macro_recall = recall_score(y_test, y_pred, average='macro')

        # Calculate AUROC for binary classification or multi-class
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)

            # For multi-class, use one-vs-rest AUROC
            if len(np.unique(y_test)) > 2:
                try:
                    auroc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                except:
                    auroc = None
            else:
                # Binary classification
                auroc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auroc = None

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Model complexity metrics
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            n_features = X_train.shape[1]

        if hasattr(model, 'n_estimators'):
            n_estimators = model.n_estimators
        else:
            n_estimators = None

        return {
            'model': model_name,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'auroc': auroc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'n_features': n_features,
            'n_estimators': n_estimators,
            'status': 'success'
        }

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {
            'model': model_name,
            'accuracy': None,
            'macro_f1': None,
            'weighted_f1': None,
            'macro_precision': None,
            'macro_recall': None,
            'auroc': None,
            'cv_mean': None,
            'cv_std': None,
            'n_features': None,
            'n_estimators': None,
            'status': f'error: {str(e)}'
        }

def evaluate_anomaly_model(model, X_test, y_test_binary, model_name):
    """Evaluate an anomaly detection model"""
    print(f"Evaluating anomaly detection: {model_name}...")

    try:
        # Fit the model
        model.fit(X_test)

        # Get anomaly scores/predictions
        if hasattr(model, 'predict'):
            # For models that support predict method
            y_pred = model.predict(X_test)
            # Convert to binary (1 = anomaly, 0 = normal)
            y_pred_binary = (y_pred == -1).astype(int)
        elif hasattr(model, 'fit_predict'):
            # For DBSCAN and similar
            y_pred = model.fit_predict(X_test)
            y_pred_binary = (y_pred == -1).astype(int)
        else:
            return {
                'model': model_name,
                'auroc_defect': None,
                'precision_anomaly': None,
                'recall_anomaly': None,
                'f1_anomaly': None,
                'n_anomalies': None,
                'contamination_rate': None,
                'status': 'error: no prediction method'
            }

        # Calculate AUROC for anomaly detection
        if len(np.unique(y_test_binary)) == 2:
            # For binary anomaly detection, we need decision function or scores
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                # Higher scores = more anomalous
                auroc = roc_auc_score(y_test_binary, scores)
            elif hasattr(model, 'score_samples'):
                scores = model.score_samples(X_test)
                # Lower scores = more anomalous, so invert
                auroc = roc_auc_score(y_test_binary, -scores)
            else:
                # Use predictions as binary scores
                auroc = roc_auc_score(y_test_binary, y_pred_binary)
        else:
            auroc = None

        # Calculate additional anomaly detection metrics
        if len(np.unique(y_test_binary)) == 2:
            # Precision, Recall, F1 for anomaly detection
            precision_anomaly = precision_score(y_test_binary, y_pred_binary, zero_division=0)
            recall_anomaly = recall_score(y_test_binary, y_pred_binary, zero_division=0)
            f1_anomaly = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        else:
            precision_anomaly = None
            recall_anomaly = None
            f1_anomaly = None

        # Count number of anomalies detected
        n_anomalies = np.sum(y_pred_binary == 1)
        total_samples = len(y_test_binary)
        contamination_rate = n_anomalies / total_samples if total_samples > 0 else 0

        # Get contamination parameter if available
        if hasattr(model, 'contamination'):
            expected_contamination = model.contamination
        else:
            expected_contamination = None

        return {
            'model': model_name,
            'auroc_defect': auroc,
            'precision_anomaly': precision_anomaly,
            'recall_anomaly': recall_anomaly,
            'f1_anomaly': f1_anomaly,
            'n_anomalies': n_anomalies,
            'contamination_rate': contamination_rate,
            'expected_contamination': expected_contamination,
            'status': 'success'
        }

    except Exception as e:
        print(f"Error evaluating anomaly {model_name}: {e}")
        return {
            'model': model_name,
            'auroc_defect': None,
            'precision_anomaly': None,
            'recall_anomaly': None,
            'f1_anomaly': None,
            'n_anomalies': None,
            'contamination_rate': None,
            'expected_contamination': None,
            'status': f'error: {str(e)}'
        }

def create_comparison_table(results_supervised, results_anomaly):
    """Create a comprehensive comparison table with enhanced metrics"""

    # Prepare data for the table
    table_data = []

    # Add supervised models
    for result in results_supervised:
        if result['status'] == 'success':
            row = {
                'Model': result['model'].replace('_', ' ').title(),
                'Accuracy': f"{result['accuracy']:.3f}" if result['accuracy'] is not None else '—',
                'Macro F1': f"{result['macro_f1']:.3f}" if result['macro_f1'] is not None else '—',
                'Macro Prec.': f"{result['macro_precision']:.3f}" if result['macro_precision'] is not None else '—',
                'Macro Rec.': f"{result['macro_recall']:.3f}" if result['macro_recall'] is not None else '—',
                'AUROC': f"{result['auroc']:.3f}" if result['auroc'] is not None else '—',
                'CV Mean': f"{result['cv_mean']:.3f}" if result['cv_mean'] is not None else '—',
                'CV Std': f"{result['cv_std']:.3f}" if result['cv_std'] is not None else '—'
            }
            table_data.append(row)

    # Add anomaly detection models
    for result in results_anomaly:
        if result['status'] == 'success':
            row = {
                'Model': result['model'].replace('_', ' ').title(),
                'Accuracy': '—',
                'Macro F1': '—',
                'Macro Prec.': f"{result['precision_anomaly']:.3f}" if result['precision_anomaly'] is not None else '—',
                'Macro Rec.': f"{result['recall_anomaly']:.3f}" if result['recall_anomaly'] is not None else '—',
                'AUROC': f"{result['auroc_defect']:.3f}" if result['auroc_defect'] is not None else '—',
                'CV Mean': f"F1: {result['f1_anomaly']:.3f}" if result['f1_anomaly'] is not None else '—',
                'CV Std': f"Anom: {result['n_anomalies']}" if result['n_anomalies'] is not None else '—'
            }
            table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    return df

def save_results_table(df, output_path, results_supervised=None, results_anomaly=None):
    """Save the comparison table to CSV and display it"""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)

    # Display the table
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Also save detailed results if provided
    if results_supervised is not None and results_anomaly is not None:
        detailed_path = output_path.parent / "detailed_model_results.csv"
        detailed_df = pd.DataFrame(results_supervised + results_anomaly)
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to: {detailed_path}")

def main():
    """Main function to run model comparison"""
    print("🔬 Fabric Texture Analysis - Model Comparison")
    print("=" * 50)

    # Setup paths
    csv_path = 'data/features.csv'
    output_dir = Path('clean_test_output')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'model_comparison_table.csv'

    # Load and prepare data
    X, y, class_names, feature_names = load_and_prepare_data(csv_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create binary labels for anomaly detection
    y_test_binary = create_binary_labels(y_test, class_names)

    print(f"\nTraining set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Classes: {len(class_names)} fabric types")
    print()

    # Define supervised models to test
    supervised_models = {
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        supervised_models['XGBoost'] = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(class_names),
            random_state=42
        )

    # Evaluate supervised models
    results_supervised = []
    for model_name, model in supervised_models.items():
        result = evaluate_supervised_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        results_supervised.append(result)

    # Define anomaly detection models
    anomaly_models = {
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
    }

    # Evaluate anomaly detection models
    results_anomaly = []
    for model_name, model in anomaly_models.items():
        result = evaluate_anomaly_model(
            model, X_test_scaled, y_test_binary, model_name
        )
        results_anomaly.append(result)

    # Create and save comparison table
    comparison_df = create_comparison_table(results_supervised, results_anomaly)
    save_results_table(comparison_df, output_path, results_supervised, results_anomaly)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    successful_models = [r for r in results_supervised if r['status'] == 'success']
    if successful_models:
        accuracies = [r['accuracy'] for r in successful_models if r['accuracy'] is not None]
        f1_scores = [r['macro_f1'] for r in successful_models if r['macro_f1'] is not None]
        precisions = [r['macro_precision'] for r in successful_models if r['macro_precision'] is not None]
        recalls = [r['macro_recall'] for r in successful_models if r['macro_recall'] is not None]
        aurocs = [r['auroc'] for r in successful_models if r['auroc'] is not None]

        print(f"Supervised models evaluated: {len(successful_models)}")
        if accuracies:
            print(".3f")
        if f1_scores:
            print(".3f")
        if precisions:
            print(".3f")
        if recalls:
            print(".3f")
        if aurocs:
            print(".3f")

        # Best performing model
        if accuracies:
            best_model = max(successful_models, key=lambda x: x['accuracy'] if x['accuracy'] else 0)
            print(f"Best model (accuracy): {best_model['model']} ({best_model['accuracy']:.3f})")

    # Anomaly detection summary
    successful_anomaly = [r for r in results_anomaly if r['status'] == 'success']
    if successful_anomaly:
        print(f"\nAnomaly detection models evaluated: {len(successful_anomaly)}")
        for result in successful_anomaly:
            if result['auroc_defect'] is not None:
                print(f"  {result['model']}: AUROC={result['auroc_defect']:.3f}, "
                      f"Anomalies detected={result['n_anomalies']}")

    print(f"\n📁 All results saved to: {output_dir}/")
    print("✅ Enhanced model comparison complete!")

if __name__ == "__main__":
    main()
