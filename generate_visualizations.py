#!/usr/bin/env python3
"""
Standalone script to generate visualizations from existing data
without TensorFlow dependencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent, publication-ready plots
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for better plots
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150
})

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

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Classes: {le.classes_}")

    return X, y_encoded, le.classes_, feature_cols, df

def generate_feature_distributions(X, feature_names, output_dir):
    """Generate feature distribution plots"""
    print("Generating feature distributions...")

    # Select a subset of important features for visualization
    num_features = min(10, len(feature_names))
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.ravel()

    for i in range(num_features):
        if i < len(axes):
            axes[i].hist(X[:, i], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{feature_names[i].replace("_", " ").title()}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature distributions saved")

def generate_correlation_analysis(X, feature_names, output_dir):
    """Generate correlation analysis heatmap"""
    print("Generating correlation analysis...")

    # Select subset of features for correlation analysis
    subset_features = feature_names[:15]
    X_subset = X[:, :15]

    # Ensure X_subset is a proper 2D array
    X_subset = np.array(X_subset, dtype=np.float64)
    corr_matrix = np.corrcoef(X_subset.T)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix,
                xticklabels=[f.replace('_', ' ').title() for f in subset_features],
                yticklabels=[f.replace('_', ' ').title() for f in subset_features],
                cmap='coolwarm', center=0, annot=False)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Correlation analysis saved")

def generate_dimensionality_reduction(X, y, classes, output_dir):
    """Generate PCA and t-SNE plots"""
    print("Generating dimensionality reduction plots...")

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(15, 6))

    # PCA plot
    plt.subplot(1, 2, 1)
    for i, class_name in enumerate(classes):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=class_name, alpha=0.7, s=50)
    plt.title('PCA Projection')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # t-SNE (with smaller sample for speed)
    sample_size = min(500, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size//3))
    X_tsne = tsne.fit_transform(X_sample)

    plt.subplot(1, 2, 2)
    for i, class_name in enumerate(classes):
        mask = y_sample == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   label=class_name, alpha=0.7, s=50)
    plt.title('t-SNE Projection')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'dimensionality_reduction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Dimensionality reduction plots saved")

def generate_clustering_results(X, y, classes, output_dir):
    """Generate clustering analysis plots"""
    print("Generating clustering results...")

    # Use PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # K-means clustering
    n_clusters = len(classes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    plt.figure(figsize=(15, 6))

    # Original labels
    plt.subplot(1, 2, 1)
    for i, class_name in enumerate(classes):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=class_name, alpha=0.7, s=50)
    plt.title('Original Fabric Types')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Clustering results
    plt.subplot(1, 2, 2)
    for i in range(n_clusters):
        mask = clusters == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'Cluster {i+1}', alpha=0.7, s=50)
    plt.title('K-Means Clustering Results')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Clustering results saved")

def generate_model_comparison(X, y, classes, output_dir):
    """Generate model comparison plots"""
    print("Generating model comparison...")

    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simple models to compare
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}
    for name, model in models.items():
        try:
            # Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Calculate accuracy
            accuracy = np.mean(y_pred == y_test)
            results[name] = accuracy
            print(".3f")
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = 0.0

    # Plot results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = list(results.values())

    bars = plt.bar(names, accuracies, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                '.3f', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Model comparison saved")

def generate_confusion_matrices(X, y, classes, output_dir):
    """Generate confusion matrix visualization"""
    print("Generating confusion matrices...")

    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Random Forest Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices saved")

def generate_feature_importance(X, y, feature_names, output_dir):
    """Generate feature importance plot"""
    print("Generating feature importance...")

    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importance
    importance = rf.feature_importances_

    # Sort features by importance
    indices = np.argsort(importance)[::-1]

    # Plot top 20 features
    top_n = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))

    bars = plt.barh(range(top_n),
                   importance[indices[:top_n]][::-1],
                   color='skyblue', edgecolor='black', alpha=0.7)

    feature_labels = [feature_names[i].replace('_', ' ').title() for i in indices[:top_n]][::-1]
    plt.yticks(range(top_n), feature_labels)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance saved")

def generate_anomaly_detection(X, output_dir):
    """Generate simple anomaly detection visualization"""
    print("Generating anomaly detection plot...")

    # Use PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Simple anomaly detection using isolation forest concept
    # (simplified version without sklearn dependencies)
    from sklearn.ensemble import IsolationForest

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X)

    plt.figure(figsize=(10, 8))

    # Plot normal points
    normal_mask = anomaly_scores == 1
    plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
               c='blue', label='Normal', alpha=0.6, s=50)

    # Plot anomalies
    anomaly_mask = anomaly_scores == -1
    plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
               c='red', label='Anomaly', alpha=0.8, s=60, marker='x')

    plt.title('Anomaly Detection Results')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Anomaly detection saved")

def generate_robustness_analysis(X, y, output_dir):
    """Generate robustness analysis plot"""
    print("Generating robustness analysis...")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Test different sample sizes
    sample_sizes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    accuracies = []

    for size in sample_sizes:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, test_size=1-size, random_state=42, stratify=y
        )

        if len(X_sample) < 20:  # Skip if too small
            continue

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)

        # Determine appropriate number of CV folds based on smallest class
        unique_classes, class_counts = np.unique(y_sample, return_counts=True)
        min_class_count = min(class_counts)
        cv_folds = min(3, max(2, min_class_count))  # At least 2 folds, max 3

        if cv_folds < 2:
            continue

        try:
            # Cross-validation
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(rf, X_scaled, y_sample, cv=cv_folds, scoring='accuracy')
            accuracies.append(np.mean(scores))
        except:
            continue

    if len(accuracies) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot([s*100 for s in sample_sizes[:len(accuracies)]], accuracies,
                 marker='o', linewidth=2, markersize=8)
        plt.title('Model Robustness vs Sample Size')
        plt.xlabel('Training Sample Size (%)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Robustness analysis saved")
    else:
        # Create a simple placeholder plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Robustness analysis requires more samples per class\nfor reliable cross-validation',
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Model Robustness Analysis')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Robustness analysis saved (placeholder)")

def main():
    """Main function to generate all visualizations"""
    print("🔬 Generating Fabric Texture Analysis Visualizations")
    print("=" * 60)

    # Setup paths
    csv_path = 'data/features.csv'
    output_dir = Path('clean_test_output/visualizations')
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Load and prepare data
        X, y, classes, feature_names, df = load_and_prepare_data(csv_path)
        print()

        # Generate all visualizations
        generate_feature_distributions(X, feature_names, output_dir)
        generate_correlation_analysis(X, feature_names, output_dir)
        generate_dimensionality_reduction(X, y, classes, output_dir)
        generate_clustering_results(X, y, classes, output_dir)
        generate_model_comparison(X, y, classes, output_dir)
        generate_confusion_matrices(X, y, classes, output_dir)
        generate_feature_importance(X, y, feature_names, output_dir)
        generate_anomaly_detection(X, output_dir)
        generate_robustness_analysis(X, y, output_dir)

        print()
        print("🎉 All visualizations generated successfully!")
        print(f"📁 Visualizations saved to: {output_dir}")
        print(f"📊 Generated {len(list(output_dir.glob('*.png')))} visualization files")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
