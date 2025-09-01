import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive plots will be disabled.")


class AdvancedVisualizer:
    """
    Advanced visualization suite for texture analysis research.
    
    Provides sophisticated dimensionality reduction visualizations, feature space
    analysis, and interactive plots for comprehensive model interpretation.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the advanced visualizer.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style for publication-ready plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def plot_tsne_analysis(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: Optional[List[str]] = None,
                          perplexities: List[float] = [5, 30, 50],
                          n_components: int = 2,
                          class_names: Optional[List[str]] = None) -> None:
        """
        Create t-SNE analysis with multiple perplexity values.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        feature_names : List[str], optional
            Names of features
        perplexities : List[float]
            Perplexity values to test
        n_components : int
            Number of t-SNE components
        class_names : List[str], optional
            Names of classes
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn not available. Cannot perform t-SNE analysis.")
            return
        
        print("Creating t-SNE analysis...")
        
        # Create subplots for different perplexity values
        n_perplexities = len(perplexities)
        fig, axes = plt.subplots(1, n_perplexities, figsize=(6*n_perplexities, 6))
        
        if n_perplexities == 1:
            axes = [axes]
        
        unique_labels = np.unique(y)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, perplexity in enumerate(perplexities):
            print(f"  Computing t-SNE with perplexity={perplexity}...")
            
            # Perform t-SNE
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42,
                n_iter=1000,
                n_jobs=-1
            )
            
            X_tsne = tsne.fit_transform(X)
            
            # Plot results
            ax = axes[i]
            
            for label_idx, label in enumerate(unique_labels):
                mask = y == label
                label_name = class_names[label] if class_names else f'Class {label}'
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                          c=[colors[label_idx]], label=label_name, 
                          alpha=0.7, s=50)
            
            ax.set_title(f't-SNE (perplexity={perplexity})')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tsne_analysis.png')
        plt.close()
        
        # Interactive plot if Plotly available
        if PLOTLY_AVAILABLE:
            self._create_interactive_tsne(X, y, perplexities, class_names)
    
    def plot_umap_analysis(self, X: np.ndarray, y: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          n_neighbors: List[int] = [5, 15, 50],
                          min_dist: float = 0.1,
                          class_names: Optional[List[str]] = None) -> None:
        """
        Create UMAP analysis with different neighbor settings.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        feature_names : List[str], optional
            Names of features
        n_neighbors : List[int]
            Number of neighbors to test
        min_dist : float
            Minimum distance parameter for UMAP
        class_names : List[str], optional
            Names of classes
        """
        if not UMAP_AVAILABLE:
            print("UMAP not available. Install with: pip install umap-learn")
            return
        
        print("Creating UMAP analysis...")
        
        # Create subplots for different neighbor values
        n_settings = len(n_neighbors)
        fig, axes = plt.subplots(1, n_settings, figsize=(6*n_settings, 6))
        
        if n_settings == 1:
            axes = [axes]
        
        unique_labels = np.unique(y)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, n_neighbor in enumerate(n_neighbors):
            print(f"  Computing UMAP with n_neighbors={n_neighbor}...")
            
            # Perform UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbor,
                min_dist=min_dist,
                n_components=2,
                random_state=42,
                n_jobs=1  # Set to 1 to avoid issues
            )
            
            X_umap = reducer.fit_transform(X)
            
            # Plot results
            ax = axes[i]
            
            for label_idx, label in enumerate(unique_labels):
                mask = y == label
                label_name = class_names[label] if class_names else f'Class {label}'
                ax.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                          c=[colors[label_idx]], label=label_name, 
                          alpha=0.7, s=50)
            
            ax.set_title(f'UMAP (n_neighbors={n_neighbor})')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'umap_analysis.png')
        plt.close()
        
        # Interactive plot if Plotly available
        if PLOTLY_AVAILABLE:
            self._create_interactive_umap(X, y, n_neighbors, min_dist, class_names)
    
    def plot_feature_space_evolution(self, X_before: np.ndarray, 
                                   X_after: np.ndarray,
                                   y: np.ndarray,
                                   method: str = 'pca',
                                   title_suffix: str = '',
                                   class_names: Optional[List[str]] = None) -> None:
        """
        Visualize feature space before and after training/transformation.
        
        Parameters:
        -----------
        X_before : np.ndarray
            Feature matrix before transformation
        X_after : np.ndarray
            Feature matrix after transformation
        y : np.ndarray
            Labels
        method : str
            Dimensionality reduction method ('pca', 'tsne', 'umap')
        title_suffix : str
            Suffix for plot title
        class_names : List[str], optional
            Names of classes
        """
        print(f"Creating feature space evolution plot using {method.upper()}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        unique_labels = np.unique(y)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        datasets = [('Before', X_before), ('After', X_after)]
        
        for i, (stage, X) in enumerate(datasets):
            if method == 'pca' and SKLEARN_AVAILABLE:
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                explained_var = reducer.explained_variance_ratio_
                xlabel = f'PC1 ({explained_var[0]:.2%} var)'
                ylabel = f'PC2 ({explained_var[1]:.2%} var)'
                
            elif method == 'tsne' and SKLEARN_AVAILABLE:
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                X_reduced = reducer.fit_transform(X)
                xlabel = 't-SNE 1'
                ylabel = 't-SNE 2'
                
            elif method == 'umap' and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                xlabel = 'UMAP 1'
                ylabel = 'UMAP 2'
                
            else:
                print(f"Method {method} not available, falling back to PCA")
                if SKLEARN_AVAILABLE:
                    reducer = PCA(n_components=2, random_state=42)
                    X_reduced = reducer.fit_transform(X)
                    explained_var = reducer.explained_variance_ratio_
                    xlabel = f'PC1 ({explained_var[0]:.2%} var)'
                    ylabel = f'PC2 ({explained_var[1]:.2%} var)'
                else:
                    continue
            
            ax = axes[i]
            
            for label_idx, label in enumerate(unique_labels):
                mask = y == label
                label_name = class_names[label] if class_names else f'Class {label}'
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                          c=[colors[label_idx]], label=label_name, 
                          alpha=0.7, s=50)
            
            ax.set_title(f'{stage} {title_suffix}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if i == 1:  # Only show legend on second plot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'feature_space_evolution_{method}_{title_suffix.lower().replace(" ", "_")}.png'
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_dimensionality_reduction_comparison(self, X: np.ndarray, y: np.ndarray,
                                               class_names: Optional[List[str]] = None) -> None:
        """
        Compare different dimensionality reduction methods.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        class_names : List[str], optional
            Names of classes
        """
        print("Creating dimensionality reduction comparison...")
        
        methods = []
        if SKLEARN_AVAILABLE:
            methods.extend(['PCA', 't-SNE'])
        if UMAP_AVAILABLE:
            methods.append('UMAP')
        
        if not methods:
            print("No dimensionality reduction methods available")
            return
        
        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 6))
        
        if len(methods) == 1:
            axes = [axes]
        
        unique_labels = np.unique(y)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, method in enumerate(methods):
            print(f"  Computing {method}...")
            
            if method == 'PCA':
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                explained_var = reducer.explained_variance_ratio_
                xlabel = f'PC1 ({explained_var[0]:.2%})'
                ylabel = f'PC2 ({explained_var[1]:.2%})'
                
            elif method == 't-SNE':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                X_reduced = reducer.fit_transform(X)
                xlabel = 't-SNE 1'
                ylabel = 't-SNE 2'
                
            elif method == 'UMAP':
                reducer = umap.UMAP(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                xlabel = 'UMAP 1'
                ylabel = 'UMAP 2'
            
            ax = axes[i]
            
            for label_idx, label in enumerate(unique_labels):
                mask = y == label
                label_name = class_names[label] if class_names else f'Class {label}'
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                          c=[colors[label_idx]], label=label_name, 
                          alpha=0.7, s=50)
            
            ax.set_title(f'{method}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if i == len(methods) - 1:  # Show legend on last plot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimensionality_reduction_comparison.png')
        plt.close()
    
    def plot_feature_importance_evolution(self, importance_history: Dict[str, List[float]],
                                        feature_names: List[str]) -> None:
        """
        Plot how feature importance changes during training.
        
        Parameters:
        -----------
        importance_history : Dict[str, List[float]]
            Dictionary mapping iteration/epoch to importance values
        feature_names : List[str]
            Names of features
        """
        print("Creating feature importance evolution plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convert to DataFrame for easier plotting
        importance_df = pd.DataFrame(importance_history, index=feature_names)
        
        # Plot 1: Heatmap of feature importance over time
        sns.heatmap(importance_df, ax=ax1, cmap='viridis', cbar_kws={'label': 'Importance'})
        ax1.set_title('Feature Importance Evolution')
        ax1.set_xlabel('Training Stage')
        ax1.set_ylabel('Features')
        
        # Plot 2: Line plot for top features
        top_features = importance_df.iloc[:, -1].nlargest(10).index
        
        for feature in top_features:
            ax2.plot(importance_df.columns, importance_df.loc[feature], 
                    label=feature[:20] + ('...' if len(feature) > 20 else ''),
                    marker='o', markersize=3)
        
        ax2.set_title('Top 10 Feature Importance Trends')
        ax2.set_xlabel('Training Stage')
        ax2.set_ylabel('Importance')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_evolution.png')
        plt.close()
    
    def _create_interactive_tsne(self, X: np.ndarray, y: np.ndarray,
                               perplexities: List[float],
                               class_names: Optional[List[str]] = None) -> None:
        """Create interactive t-SNE plot with Plotly."""
        if not PLOTLY_AVAILABLE:
            return
        
        print("Creating interactive t-SNE plot...")
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=len(perplexities),
            subplot_titles=[f'Perplexity = {p}' for p in perplexities],
            specs=[[{"type": "scatter"}] * len(perplexities)]
        )
        
        unique_labels = np.unique(y)
        colors = px.colors.qualitative.Set3[:len(unique_labels)]
        
        for i, perplexity in enumerate(perplexities):
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            for label_idx, label in enumerate(unique_labels):
                mask = y == label
                label_name = class_names[label] if class_names else f'Class {label}'
                
                fig.add_trace(
                    go.Scatter(
                        x=X_tsne[mask, 0],
                        y=X_tsne[mask, 1],
                        mode='markers',
                        name=label_name,
                        marker=dict(color=colors[label_idx], size=8),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title="Interactive t-SNE Analysis",
            height=500,
            showlegend=True
        )
        
        fig.write_html(self.output_dir / 'interactive_tsne.html')
    
    def _create_interactive_umap(self, X: np.ndarray, y: np.ndarray,
                               n_neighbors: List[int], min_dist: float,
                               class_names: Optional[List[str]] = None) -> None:
        """Create interactive UMAP plot with Plotly."""
        if not PLOTLY_AVAILABLE or not UMAP_AVAILABLE:
            return
        
        print("Creating interactive UMAP plot...")
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=len(n_neighbors),
            subplot_titles=[f'n_neighbors = {n}' for n in n_neighbors],
            specs=[[{"type": "scatter"}] * len(n_neighbors)]
        )
        
        unique_labels = np.unique(y)
        colors = px.colors.qualitative.Set3[:len(unique_labels)]
        
        for i, n_neighbor in enumerate(n_neighbors):
            reducer = umap.UMAP(
                n_neighbors=n_neighbor, 
                min_dist=min_dist, 
                n_components=2, 
                random_state=42
            )
            X_umap = reducer.fit_transform(X)
            
            for label_idx, label in enumerate(unique_labels):
                mask = y == label
                label_name = class_names[label] if class_names else f'Class {label}'
                
                fig.add_trace(
                    go.Scatter(
                        x=X_umap[mask, 0],
                        y=X_umap[mask, 1],
                        mode='markers',
                        name=label_name,
                        marker=dict(color=colors[label_idx], size=8),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title="Interactive UMAP Analysis",
            height=500,
            showlegend=True
        )
        
        fig.write_html(self.output_dir / 'interactive_umap.html')
    
    def create_comprehensive_visualization_report(self, X: np.ndarray, y: np.ndarray,
                                                feature_names: List[str],
                                                class_names: Optional[List[str]] = None) -> None:
        """
        Create a comprehensive visualization report.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        feature_names : List[str]
            Names of features
        class_names : List[str], optional
            Names of classes
        """
        print("Creating comprehensive visualization report...")
        
        # Create all visualizations
        self.plot_dimensionality_reduction_comparison(X, y, class_names)
        
        if SKLEARN_AVAILABLE:
            self.plot_tsne_analysis(X, y, feature_names, class_names=class_names)
        
        if UMAP_AVAILABLE:
            self.plot_umap_analysis(X, y, feature_names, class_names=class_names)
        
        print(f"Comprehensive visualization report saved to {self.output_dir}") 