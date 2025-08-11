import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import os
from datetime import datetime
from pathlib import Path

# Import our custom analysis modules
from .models.supervised_models import SupervisedModelSuite
from .models.anomaly_detection import AnomalyDetectionSuite
from .visualization.visualizers import ResearchVisualizer
from .evaluation.evaluators import ModelEvaluator
from .utils.data_utils import DataProcessor

class ResearchAnalyzer:
    """
    Comprehensive research analysis framework for micro-texture fabric analysis
    
    Provides end-to-end ML pipeline for:
    - Supervised fabric classification (fabric type, material identification)
    - Fabric quality assessment and damage classification
    - Anomaly detection for micro-damage and tear identification
    - Feature analysis and visualization
    - Model evaluation and comparison
    - Statistical analysis
    """
    
    def __init__(self, data_path="data/features.csv", output_dir="research_output"):
        """
        Initialize the research analyzer for fabric analysis
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing extracted features
        output_dir : str
            Directory to save all research outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "evaluation").mkdir(exist_ok=True)
        (self.output_dir / "statistical_analysis").mkdir(exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.supervised_models = SupervisedModelSuite()
        self.anomaly_models = AnomalyDetectionSuite()
        self.visualizer = ResearchVisualizer(self.output_dir / "visualizations")
        self.evaluator = ModelEvaluator(self.output_dir / "evaluation")
        
        # Data storage
        self.features_df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # Results storage
        self.results = {
            'supervised_results': {},
            'anomaly_results': {},
            'feature_analysis': {},
            'visualizations': [],
            'statistical_tests': {}
        }
    
    def load_and_prepare_data(self, target_column=None, create_synthetic_labels=True):
        """
        Load feature data and prepare for analysis
        
        Parameters:
        -----------
        target_column : str, optional
            Column name to use as target variable
        create_synthetic_labels : bool
            If True and no target_column, create synthetic labels for demonstration
        """
        print("Loading and preparing feature data...")
        
        # Load features CSV
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Features file not found: {self.data_path}")
        
        self.features_df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.features_df)} samples with {len(self.features_df.columns)} features")
        
        # Process data using DataProcessor
        self.X, self.y, self.feature_names = self.data_processor.prepare_features(
            self.features_df, target_column, create_synthetic_labels
        )
        
        print(f"Prepared feature matrix: {self.X.shape}")
        print(f"Feature names: {len(self.feature_names)}")
        
        if self.y is not None:
            unique_labels = np.unique(self.y)
            print(f"Target labels: {unique_labels} (counts: {np.bincount(self.y)})")
        
        return self.X, self.y, self.feature_names
    
    def run_supervised_analysis(self, test_size=0.2, cv_folds=5):
        """
        Run comprehensive supervised learning analysis
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        cv_folds : int
            Number of cross-validation folds
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running supervised learning analysis...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all supervised models
        self.results['supervised_results'] = self.supervised_models.train_all_models(
            X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names, cv_folds
        )
        
        # Generate model comparison visualization
        self.visualizer.plot_model_comparison(self.results['supervised_results'])
        
        print("Supervised analysis complete!")
        return self.results['supervised_results']
    
    def run_anomaly_detection(self, contamination=0.1):
        """
        Run anomaly detection analysis for micro-damage detection
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of anomalies in the data
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running anomaly detection analysis...")
        
        # Scale features for anomaly detection
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Run anomaly detection models
        self.results['anomaly_results'] = self.anomaly_models.detect_anomalies(
            X_scaled, contamination, self.feature_names
        )
        
        # Visualize anomaly detection results
        self.visualizer.plot_anomaly_detection(X_scaled, self.results['anomaly_results'])
        
        print("Anomaly detection analysis complete!")
        return self.results['anomaly_results']
    
    def run_feature_analysis(self):
        """
        Comprehensive feature analysis including correlation, importance, and statistics
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running feature analysis...")
        
        # Feature correlation analysis
        correlation_matrix = self.data_processor.compute_feature_correlations(self.X, self.feature_names)
        self.visualizer.plot_correlation_heatmap(correlation_matrix, self.feature_names)
        
        # Feature importance from Random Forest
        if 'random_forest' in self.results.get('supervised_results', {}):
            rf_model = self.results['supervised_results']['random_forest']['model']
            feature_importance = rf_model.feature_importances_
            self.visualizer.plot_feature_importance(feature_importance, self.feature_names)
        
        # Statistical analysis of features by class
        if self.y is not None:
            statistical_tests = self.data_processor.statistical_feature_comparison(
                self.X, self.y, self.feature_names
            )
            self.results['statistical_tests'] = statistical_tests
            self.visualizer.plot_feature_distributions_by_class(self.X, self.y, self.feature_names)
        
        print("Feature analysis complete!")
    
    def run_dimensionality_reduction(self, methods=['pca', 'tsne', 'umap']):
        """
        Run dimensionality reduction and visualization
        
        Parameters:
        -----------
        methods : list
            List of methods to use: ['pca', 'tsne', 'umap']
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running dimensionality reduction...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Run dimensionality reduction
        reduced_data = {}
        
        if 'pca' in methods:
            print("Running PCA...")
            pca = PCA(n_components=2)
            reduced_data['pca'] = pca.fit_transform(X_scaled)
            
        if 'tsne' in methods:
            print("Running t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
            reduced_data['tsne'] = tsne.fit_transform(X_scaled)
            
        if 'umap' in methods:
            print("Running UMAP...")
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            reduced_data['umap'] = umap_reducer.fit_transform(X_scaled)
        
        # Visualize results
        self.visualizer.plot_dimensionality_reduction(reduced_data, self.y)
        
        print("Dimensionality reduction complete!")
        return reduced_data
    
    def run_clustering_analysis(self, methods=['kmeans', 'dbscan', 'hierarchical']):
        """
        Run unsupervised clustering analysis
        
        Parameters:
        -----------
        methods : list
            Clustering methods to use
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running clustering analysis...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Run clustering
        clustering_results = self.data_processor.perform_clustering(X_scaled, methods)
        
        # Visualize clustering results
        self.visualizer.plot_clustering_results(X_scaled, clustering_results, self.y)
        
        print("Clustering analysis complete!")
        return clustering_results
    
    def run_robustness_testing(self, noise_levels=[0.01, 0.05, 0.1]):
        """
        Test model robustness with noise injection
        
        Parameters:
        -----------
        noise_levels : list
            Standard deviations for Gaussian noise
        """
        if self.X is None or 'supervised_results' not in self.results:
            raise ValueError("Need to run supervised analysis first.")
        
        print("Running robustness testing...")
        
        robustness_results = self.evaluator.test_model_robustness(
            self.X, self.y, self.results['supervised_results'], noise_levels
        )
        
        self.visualizer.plot_robustness_results(robustness_results)
        
        print("Robustness testing complete!")
        return robustness_results
    
    def run_complete_analysis(self, target_column=None):
        """
        Run the complete research analysis pipeline
        
        Parameters:
        -----------
        target_column : str, optional
            Target column for supervised learning
        """
        print("Starting complete research analysis pipeline...")
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data(target_column)
        
        # Step 2: Run supervised learning
        if self.y is not None:
            self.run_supervised_analysis()
        
        # Step 3: Run anomaly detection
        self.run_anomaly_detection()
        
        # Step 4: Feature analysis
        self.run_feature_analysis()
        
        # Step 5: Dimensionality reduction
        self.run_dimensionality_reduction()
        
        # Step 6: Clustering
        self.run_clustering_analysis()
        
        # Step 7: Robustness testing (if supervised models exist)
        if 'supervised_results' in self.results and self.results['supervised_results']:
            self.run_robustness_testing()
        
        # Step 8: Generate comprehensive report
        self.generate_research_report()
        
        print(f"\nResearch analysis complete! Results saved to: {self.output_dir}")
    
    def generate_research_report(self):
        """
        Generate a comprehensive research report summarizing all findings
        """
        report_path = self.output_dir / "research_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Micro-Texture Fabric Analysis Research Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write("## Data Summary\n")
            f.write(f"- Total samples: {len(self.features_df)}\n")
            f.write(f"- Features extracted: {len(self.feature_names)}\n")
            if self.y is not None:
                f.write(f"- Target classes: {len(np.unique(self.y))}\n")
            f.write("\n")
            
            # Supervised learning results
            if 'supervised_results' in self.results and self.results['supervised_results']:
                f.write("## Supervised Learning Results\n\n")
                for model_name, results in self.results['supervised_results'].items():
                    f.write(f"### {model_name.replace('_', ' ').title()}\n")
                    f.write(f"- Test Accuracy: {results['test_accuracy']:.4f}\n")
                    f.write(f"- CV Mean Accuracy: {results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}\n")
                    if 'test_auc' in results and results['test_auc'] is not None:
                        f.write(f"- Test AUC: {results['test_auc']:.4f}\n")
                    f.write("\n")
            
            # Anomaly detection results
            if 'anomaly_results' in self.results and self.results['anomaly_results']:
                f.write("## Anomaly Detection Results\n\n")
                for model_name, results in self.results['anomaly_results'].items():
                    f.write(f"### {model_name.replace('_', ' ').title()}\n")
                    f.write(f"- Anomalies detected: {results['n_anomalies']}\n")
                    f.write(f"- Anomaly percentage: {results['anomaly_percentage']:.2f}%\n")
                    f.write("\n")
            
            # Statistical tests
            if 'statistical_tests' in self.results and self.results['statistical_tests']:
                f.write("## Statistical Analysis\n\n")
                significant_features = [
                    feature for feature, stats_dict in self.results['statistical_tests'].items()
                    if isinstance(stats_dict, dict) and stats_dict.get('p_value', 1.0) < 0.05
                ]
                f.write(f"- Features with significant class differences (p < 0.05): {len(significant_features)}\n")
                f.write("\n")
            
            f.write("## Files Generated\n\n")
            f.write("- Model comparison plots in `visualizations/`\n")
            f.write("- Feature analysis plots in `visualizations/`\n")
            f.write("- Dimensionality reduction plots in `visualizations/`\n")
            f.write("- Evaluation metrics in `evaluation/`\n")
            f.write("- Statistical analysis results in `statistical_analysis/`\n")
        
        print(f"Research report saved to: {report_path}")

# Convenience function for quick analysis
def run_quick_analysis(data_path="data/features.csv", target_column=None, output_dir="research_output"):
    """
    Quick function to run complete analysis pipeline
    
    Parameters:
    -----------
    data_path : str
        Path to features CSV file
    target_column : str, optional
        Target column for supervised learning
    output_dir : str
        Output directory for results
    """
    analyzer = ResearchAnalyzer(data_path, output_dir)
    analyzer.run_complete_analysis(target_column)
    return analyzer 