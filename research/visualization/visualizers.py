import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from utils.captions import build_caption
except ImportError:
    build_caption = None

class ResearchVisualizer:
    """
    Comprehensive visualization suite for micro-texture fabric analysis research.
    
    Provides publication-ready visualizations for:
    - Fabric classification model performance
    - Damage detection and anomaly analysis
    - Feature importance and correlation analysis
    - Fabric type clustering and grouping
    - Quality assessment visualization
    """
    
    def __init__(self, output_dir):
        """
        Initialize the visualizer
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style for consistent, publication-ready plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots (readable fonts, high DPI)
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.labelsize': 12,
            'legend.fontsize': 11,
            'figure.dpi': 300,
            'figure.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
    
    def plot_model_comparison(self, supervised_results):
        """
        Create comprehensive model comparison visualizations
        
        Parameters:
        -----------
        supervised_results : dict
            Results from supervised model training
        """
        if not supervised_results:
            print("No supervised results to visualize")
            return
        
        print("Creating model comparison visualizations...")
        
        # Extract metrics for comparison
        models = []
        test_accuracies = []
        cv_means = []
        cv_stds = []
        f1_scores = []
        
        for model_name, results in supervised_results.items():
            models.append(model_name.replace('_', ' ').title())
            test_accuracies.append(results['test_accuracy'])
            cv_means.append(results['cv_mean'])
            cv_stds.append(results['cv_std'])
            f1_scores.append(results['f1_score'])
        
        # Create multi-panel comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Test Accuracy Comparison
        bars1 = axes[0, 0].bar(models, test_accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, test_accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Cross-Validation Scores with Error Bars
        axes[0, 1].errorbar(range(len(models)), cv_means, yerr=cv_stds, 
                           fmt='o', capsize=5, capthick=2, color='red')
        axes[0, 1].set_title('Cross-Validation Performance')
        axes[0, 1].set_ylabel('CV Accuracy')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. F1 Score Comparison
        bars3 = axes[1, 0].bar(models, f1_scores, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, f1 in zip(bars3, f1_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{f1:.3f}', ha='center', va='bottom')
        
        # 4. Performance Summary Table
        self._create_performance_table(axes[1, 1], models, test_accuracies, 
                                     cv_means, f1_scores)
        
        plt.tight_layout()
        # Add caption if available
        caption_text = ""
        if build_caption and hasattr(self, 'plot_config'):
            try:
                caption_text = build_caption(self.plot_config, {'metrics': ['accuracy', 'f1_score'], 'averaging': 'macro'})
                plt.figtext(0.5, 0.01, caption_text, ha='center', fontsize=9, wrap=True)
            except:
                pass
        
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save caption separately
        if caption_text:
            with open(self.output_dir / 'model_comparison_caption.txt', 'w') as f:
                f.write(caption_text)
        
        # Create confusion matrices for best models
        self._plot_confusion_matrices(supervised_results)
        
        print(f"  ✓ Model comparison plots saved to {self.output_dir}")
    
    def _create_performance_table(self, ax, models, test_accs, cv_means, f1_scores):
        """Create a performance summary table"""
        
        # Prepare data for table
        table_data = []
        for i, model in enumerate(models):
            table_data.append([
                model[:15] + '...' if len(model) > 15 else model,  # Truncate long names
                f'{test_accs[i]:.3f}',
                f'{cv_means[i]:.3f}',
                f'{f1_scores[i]:.3f}'
            ])
        
        # Sort by test accuracy (descending)
        table_data.sort(key=lambda x: float(x[1]), reverse=True)
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Model', 'Test Acc', 'CV Mean', 'F1 Score'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color best model row
        if len(table_data) > 0:
            for i in range(4):
                table[(1, i)].set_facecolor('#E8F5E8')
        
        ax.set_title('Model Performance Summary')
        ax.axis('off')
    
    def _plot_confusion_matrices(self, supervised_results):
        """Plot confusion matrices for all models"""
        n_models = len(supervised_results)
        if n_models == 0:
            return
        
        # Calculate grid size
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(supervised_results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_importance, feature_names, top_n=20):
        """
        Plot feature importance from tree-based models
        
        Parameters:
        -----------
        feature_importance : np.ndarray
            Feature importance scores
        feature_names : list
            Names of features
        top_n : int
            Number of top features to display
        """
        print("Creating feature importance visualization...")
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[-top_n:]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        bars = ax.barh(range(len(sorted_names)), sorted_importance, color='coral', alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Feature importance plot saved")
    
    def plot_correlation_heatmap(self, correlation_matrix, feature_names, threshold=0.8):
        """
        Plot correlation heatmap of features
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Feature correlation matrix
        feature_names : list
            Names of features
        threshold : float
            Threshold for highlighting strong correlations
        """
        print("Creating correlation heatmap...")
        
        # Create figure with appropriate size
        n_features = len(feature_names)
        fig_size = max(8, n_features * 0.3)
        
        fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
        
        # Full correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdBu_r',
                   center=0, square=True, ax=axes[0])
        axes[0].set_title('Feature Correlation Matrix')
        
        # High correlation pairs
        high_corr = correlation_matrix.abs() > threshold
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if high_corr.iloc[i, j]:
                    high_corr_pairs.append({
                        'Feature 1': correlation_matrix.index[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            # Plot high correlation pairs
            y_pos = range(len(high_corr_df))
            colors = ['red' if x > 0 else 'blue' for x in high_corr_df['Correlation']]
            
            axes[1].barh(y_pos, high_corr_df['Correlation'], color=colors, alpha=0.7)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f"{row['Feature 1'][:20]}..." if len(row['Feature 1']) > 20 
                                   else row['Feature 1'] for _, row in high_corr_df.iterrows()])
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].set_title(f'High Correlations (|r| > {threshold})')
            axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, f'No correlations > {threshold}', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(f'High Correlations (|r| > {threshold})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Correlation heatmap saved")
    
    def plot_dimensionality_reduction(self, reduced_data, y=None):
        """
        Plot dimensionality reduction results
        
        Parameters:
        -----------
        reduced_data : dict
            Dictionary containing reduced data for different methods
        y : np.ndarray, optional
            Target labels for coloring
        """
        print("Creating dimensionality reduction visualizations...")
        
        n_methods = len(reduced_data)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        
        # Ensure axes is always a list for consistent indexing
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, (method, data) in enumerate(reduced_data.items()):
            ax = axes[idx]
            
            if y is not None:
                scatter = ax.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(data[:, 0], data[:, 1], alpha=0.7)
            
            ax.set_title(f'{method.upper()} Projection')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        # Hide empty subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Dimensionality reduction plots saved")
    
    def plot_clustering_results(self, X_scaled, clustering_results, y_true=None):
        """
        Plot clustering results for different methods
        
        Parameters:
        -----------
        X_scaled : np.ndarray
            Scaled feature matrix
        clustering_results : dict
            Results from clustering analysis
        y_true : np.ndarray, optional
            True labels for comparison
        """
        print("Creating clustering visualizations...")
        
        n_methods = len(clustering_results)
        if n_methods == 0:
            return
        
        # Use PCA for visualization if more than 2 features
        if X_scaled.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X_scaled)
        else:
            X_vis = X_scaled
        
        cols = min(3, n_methods + (1 if y_true is not None else 0))
        rows = ((n_methods + (1 if y_true is not None else 0)) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot true labels if available
        if y_true is not None:
            row, col = plot_idx // cols, plot_idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_true, cmap='viridis', alpha=0.7)
            ax.set_title('True Labels')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            plt.colorbar(scatter, ax=ax)
            plot_idx += 1
        
        # Plot clustering results
        for method, results in clustering_results.items():
            row, col = plot_idx // cols, plot_idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            labels = results['labels']
            
            # Handle DBSCAN noise points
            if -1 in labels:
                # Plot noise points in black
                noise_mask = labels == -1
                ax.scatter(X_vis[noise_mask, 0], X_vis[noise_mask, 1], 
                          c='black', marker='x', alpha=0.5, label='Noise')
                
                # Plot clusters
                cluster_mask = labels != -1
                if np.any(cluster_mask):
                    scatter = ax.scatter(X_vis[cluster_mask, 0], X_vis[cluster_mask, 1],
                                       c=labels[cluster_mask], cmap='viridis', alpha=0.7)
            else:
                scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='viridis', alpha=0.7)
            
            ax.set_title(f'{method.title()} Clustering\n'
                        f'Clusters: {results["n_clusters"]}, '
                        f'Silhouette: {results["silhouette_score"]:.3f}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            
            plot_idx += 1
        
        # Hide empty subplots
        total_plots = n_methods + (1 if y_true is not None else 0)
        for idx in range(total_plots, rows * cols):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clustering_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Clustering plots saved")
    
    def plot_anomaly_detection(self, X_scaled, anomaly_results):
        """
        Plot anomaly detection results
        
        Parameters:
        -----------
        X_scaled : np.ndarray
            Scaled feature matrix
        anomaly_results : dict
            Results from anomaly detection
        """
        print("Creating anomaly detection visualizations...")
        
        if not anomaly_results:
            return
        
        # Use PCA for visualization if more than 2 features
        if X_scaled.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X_scaled)
        else:
            X_vis = X_scaled
        
        n_methods = len(anomaly_results)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (method, results) in enumerate(anomaly_results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            if results['anomaly_predictions'] is not None:
                predictions = results['anomaly_predictions']
                
                # Plot normal points
                normal_mask = predictions == 0
                if np.any(normal_mask):
                    ax.scatter(X_vis[normal_mask, 0], X_vis[normal_mask, 1],
                             c='blue', alpha=0.6, label='Normal', s=30)
                
                # Plot anomalies
                anomaly_mask = predictions == 1
                if np.any(anomaly_mask):
                    ax.scatter(X_vis[anomaly_mask, 0], X_vis[anomaly_mask, 1],
                             c='red', alpha=0.8, label='Anomaly', s=50, marker='^')
                
                ax.legend()
            
            ax.set_title(f'{method.replace("_", " ").title()}\n'
                        f'Anomalies: {results["n_anomalies"]} '
                        f'({results["anomaly_percentage"]:.1f}%)')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        # Hide empty subplots
        for idx in range(n_methods, rows * cols):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Anomaly detection plots saved")
    
    def plot_feature_distributions_by_class(self, X, y, feature_names, max_features=12):
        """
        Plot feature distributions split by class
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_names : list
            Names of features
        max_features : int
            Maximum number of features to plot
        """
        print("Creating feature distribution plots...")
        
        # Select most important features (or first N if no importance available)
        n_features = min(max_features, len(feature_names))
        selected_indices = range(n_features)
        
        cols = 3
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        unique_labels = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for idx, feature_idx in enumerate(selected_indices):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            feature_data = X[:, feature_idx]
            feature_name = feature_names[feature_idx]
            
            # Plot distributions for each class
            for label, color in zip(unique_labels, colors):
                mask = y == label
                class_data = feature_data[mask]
                ax.hist(class_data, alpha=0.6, label=f'Class {label}', 
                       color=color, bins=20, density=True)
            
            ax.set_title(feature_name[:30] + ('...' if len(feature_name) > 30 else ''))
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
        
        # Hide empty subplots
        for idx in range(n_features, rows * cols):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Feature distribution plots saved")
    
    def plot_robustness_results(self, robustness_results):
        """
        Plot model robustness test results
        
        Parameters:
        -----------
        robustness_results : dict
            Results from robustness testing
        """
        print("Creating robustness test visualizations...")
        
        if not robustness_results:
            return
        
        # Extract data for plotting
        models = list(robustness_results.keys())
        noise_levels = list(robustness_results[models[0]].keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy degradation with noise
        for model in models:
            accuracies = [robustness_results[model][noise]['accuracy'] for noise in noise_levels]
            axes[0].plot(noise_levels, accuracies, marker='o', label=model.replace('_', ' ').title())
        
        axes[0].set_xlabel('Noise Level (σ)')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Robustness to Noise')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Relative performance drop
        for model in models:
            baseline_acc = robustness_results[model][noise_levels[0]]['accuracy']
            relative_drops = []
            for noise in noise_levels:
                current_acc = robustness_results[model][noise]['accuracy']
                drop = (baseline_acc - current_acc) / baseline_acc * 100
                relative_drops.append(drop)
            axes[1].plot(noise_levels, relative_drops, marker='s', label=model.replace('_', ' ').title())
        
        axes[1].set_xlabel('Noise Level (σ)')
        axes[1].set_ylabel('Performance Drop (%)')
        axes[1].set_title('Relative Performance Degradation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Robustness analysis plots saved") 