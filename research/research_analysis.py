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
from .models.deep_learning_models import DeepTextureExtractor, HybridTextureClassifier
from .visualization.visualizers import ResearchVisualizer
from .visualization.advanced_visualizers import AdvancedVisualizer
from .evaluation.evaluators import ModelEvaluator
from .utils.data_utils import DataProcessor
from .utils.sample_aware_splitting import SampleAwareSplitter
from .utils.advanced_feature_selection import AdvancedFeatureSelector, HyperparameterOptimizer

# Import augmentation modules
import sys
sys.path.append('..')
try:
    from preprocess.augmentation import TextureAugmentation, FeatureAugmentation, ImagingConditionSimulator
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("Warning: Augmentation modules not available")

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
        self.advanced_visualizer = AdvancedVisualizer(self.output_dir / "advanced_visualizations")
        self.evaluator = ModelEvaluator(self.output_dir / "evaluation")
        
        # Initialize new components
        self.sample_splitter = SampleAwareSplitter()
        self.feature_selector = AdvancedFeatureSelector()
        self.hyperopt = HyperparameterOptimizer()
        self.deep_extractor = DeepTextureExtractor()
        self.hybrid_classifier = HybridTextureClassifier()
        
        # Initialize augmentation components if available
        if AUGMENTATION_AVAILABLE:
            self.texture_augmentation = TextureAugmentation()
            self.feature_augmentation = FeatureAugmentation()
            self.imaging_simulator = ImagingConditionSimulator()
        else:
            self.texture_augmentation = None
            self.feature_augmentation = None
            self.imaging_simulator = None
        
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
            'statistical_tests': {},
            'deep_learning_results': {},
            'hybrid_results': {},
            'feature_selection_results': {},
            'hyperparameter_optimization': {},
            'augmentation_results': {},
            'advanced_visualizations': []
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
    
    def run_supervised_analysis(self, test_size=0.2, val_size=0.1, cv_folds=5, 
                              use_sample_aware_splitting=True, apply_augmentation=False):
        """
        Run comprehensive supervised learning analysis with sample-aware splitting
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        val_size : float
            Proportion of data for validation
        cv_folds : int
            Number of cross-validation folds
        use_sample_aware_splitting : bool
            Whether to use sample-aware splitting to prevent data leakage
        apply_augmentation : bool
            Whether to apply feature augmentation
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running enhanced supervised learning analysis...")
        
        # Split data using sample-aware splitting or traditional splitting
        if use_sample_aware_splitting and self.features_df is not None:
            print("Using sample-aware splitting to prevent data leakage...")
            try:
                train_idx, val_idx, test_idx = self.sample_splitter.split_by_samples(
                    self.features_df, test_size=test_size, val_size=val_size, 
                    stratify_column='label'
                )
                
                X_train, X_val, X_test = self.X[train_idx], self.X[val_idx], self.X[test_idx]
                y_train, y_val, y_test = self.y[train_idx], self.y[val_idx], self.y[test_idx]
                
                # Validate split
                self.sample_splitter.validate_split(self.features_df, train_idx, val_idx, test_idx)
                
            except Exception as e:
                print(f"Sample-aware splitting failed ({e}), falling back to random splitting")
                use_sample_aware_splitting = False
        
        if not use_sample_aware_splitting:
            print("Using traditional random splitting...")
            X_temp, X_test, y_temp, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
            )
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
            )
        
        # Apply feature augmentation if requested
        if apply_augmentation and self.feature_augmentation is not None:
            print("Applying feature augmentation...")
            X_train_aug, y_train_aug = self.feature_augmentation.transform(
                X_train, y_train, apply_mixup=True, apply_noise=True, apply_smote=True
            )
            print(f"  Augmented training set: {X_train.shape[0]} -> {X_train_aug.shape[0]} samples")
            X_train, y_train = X_train_aug, y_train_aug
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all supervised models
        self.results['supervised_results'] = self.supervised_models.train_all_models(
            X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names, cv_folds
        )
        
        # Generate model comparison visualization
        self.visualizer.plot_model_comparison(self.results['supervised_results'])
        
        # Advanced visualizations
        self.advanced_visualizer.plot_dimensionality_reduction_comparison(
            X_train_scaled, y_train, class_names=None
        )
        
        print("Enhanced supervised analysis complete!")
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

    def run_feature_selection_analysis(self, methods=['ensemble'], optimize_selection=False):
        """
        Run comprehensive feature selection analysis.
        
        Parameters:
        -----------
        methods : List[str]
            Feature selection methods to apply
        optimize_selection : bool
            Whether to optimize feature selection parameters
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running advanced feature selection analysis...")
        
        if optimize_selection:
            print("Optimizing feature selection parameters...")
            best_params = self.hyperopt.optimize_feature_selection(
                self.X, self.y, self.feature_names
            )
            self.results['hyperparameter_optimization']['feature_selection'] = best_params
        
        # Apply selected feature selection methods
        for method in methods:
            print(f"Applying {method} feature selection...")
            
            if method == 'ensemble':
                X_selected, selected_features = self.feature_selector.ensemble_feature_selection(
                    self.X, self.y, self.feature_names
                )
            elif method == 'mutual_information':
                X_selected, selected_features = self.feature_selector.mutual_information_selection(
                    self.X, self.y, self.feature_names
                )
            elif method == 'rfe_cv':
                X_selected, selected_features = self.feature_selector.recursive_feature_elimination_cv(
                    self.X, self.y, self.feature_names
                )
            elif method == 'stability_selection':
                X_selected, selected_features = self.feature_selector.stability_selection(
                    self.X, self.y, self.feature_names
                )
            else:
                print(f"Unknown method: {method}")
                continue
        
        # Store results
        self.results['feature_selection_results'] = self.feature_selector.selection_results
        
        # Print summary
        self.feature_selector.print_selection_summary()
        
        print("Feature selection analysis complete!")
        return self.results['feature_selection_results']
    
    def run_deep_learning_analysis(self, models_to_use=None):
        """
        Run deep learning feature extraction and analysis.
        
        Parameters:
        -----------
        models_to_use : List[str], optional
            List of deep learning models to use
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running deep learning analysis...")
        
        # Note: This would require actual images, not just extracted features
        # For now, we'll demonstrate the capability with a comparison framework
        
        print("Deep learning analysis requires raw images for CNN feature extraction.")
        print("Current implementation works with pre-extracted handcrafted features.")
        print("To use deep learning features, extract CNN features during image processing.")
        
        # Placeholder for future implementation
        self.results['deep_learning_results'] = {
            'status': 'requires_raw_images',
            'available_models': list(self.deep_extractor.models.keys()) if hasattr(self.deep_extractor, 'models') else [],
            'note': 'Deep learning analysis needs to be integrated with image processing pipeline'
        }
        
        return self.results['deep_learning_results']
    
    def run_hybrid_analysis(self, X_deep_features=None):
        """
        Run hybrid analysis combining handcrafted and deep learning features.
        
        Parameters:
        -----------
        X_deep_features : Dict[str, np.ndarray], optional
            Dictionary of deep learning features from different models
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running hybrid feature analysis...")
        
        if X_deep_features is None:
            print("No deep learning features provided. Analyzing handcrafted features only.")
            self.results['hybrid_results'] = {
                'handcrafted_only': True,
                'feature_count': self.X.shape[1],
                'note': 'Provide X_deep_features for full hybrid analysis'
            }
        else:
            # Compare feature types
            comparison_results = self.hybrid_classifier.compare_feature_types(
                self.X, X_deep_features, self.y
            )
            
            # Test fusion strategies
            fusion_results = {}
            for strategy in ['concatenate', 'weighted', 'attention']:
                try:
                    X_fused = self.hybrid_classifier.create_feature_fusion(
                        self.X, X_deep_features, strategy
                    )
                    fusion_results[strategy] = {
                        'feature_count': X_fused.shape[1],
                        'fusion_successful': True
                    }
                except Exception as e:
                    fusion_results[strategy] = {
                        'error': str(e),
                        'fusion_successful': False
                    }
            
            self.results['hybrid_results'] = {
                'comparison_results': comparison_results,
                'fusion_results': fusion_results,
                'handcrafted_features': self.X.shape[1],
                'deep_features': {model: features.shape[1] for model, features in X_deep_features.items()}
            }
        
        print("Hybrid analysis complete!")
        return self.results['hybrid_results']
    
    def run_comprehensive_benchmarking(self, include_augmentation=True, 
                                     include_feature_selection=True,
                                     optimize_hyperparameters=False):
        """
        Run comprehensive benchmarking comparing all approaches.
        
        Parameters:
        -----------
        include_augmentation : bool
            Whether to include augmentation in benchmarking
        include_feature_selection : bool
            Whether to include feature selection in benchmarking
        optimize_hyperparameters : bool
            Whether to optimize hyperparameters for each approach
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running comprehensive benchmarking...")
        
        benchmark_results = {}
        
        # 1. Baseline: Original features, random splitting
        print("1. Baseline approach (original features, random splitting)...")
        baseline_results = self.run_supervised_analysis(
            use_sample_aware_splitting=False, apply_augmentation=False
        )
        benchmark_results['baseline'] = {
            'approach': 'Original features, random splitting',
            'best_accuracy': max([r['test_accuracy'] for r in baseline_results.values()]),
            'feature_count': self.X.shape[1],
            'data_leakage_risk': 'High'
        }
        
        # 2. Sample-aware splitting only
        print("2. Sample-aware splitting (no augmentation)...")
        sample_aware_results = self.run_supervised_analysis(
            use_sample_aware_splitting=True, apply_augmentation=False
        )
        benchmark_results['sample_aware'] = {
            'approach': 'Sample-aware splitting',
            'best_accuracy': max([r['test_accuracy'] for r in sample_aware_results.values()]),
            'feature_count': self.X.shape[1],
            'data_leakage_risk': 'None'
        }
        
        # 3. Sample-aware + augmentation
        if include_augmentation and self.feature_augmentation is not None:
            print("3. Sample-aware splitting + feature augmentation...")
            augmented_results = self.run_supervised_analysis(
                use_sample_aware_splitting=True, apply_augmentation=True
            )
            benchmark_results['augmented'] = {
                'approach': 'Sample-aware + augmentation',
                'best_accuracy': max([r['test_accuracy'] for r in augmented_results.values()]),
                'feature_count': self.X.shape[1],
                'data_leakage_risk': 'None'
            }
        
        # 4. Feature selection
        if include_feature_selection:
            print("4. Advanced feature selection...")
            self.run_feature_selection_analysis(optimize_selection=optimize_hyperparameters)
            
            # Get selected features
            if 'ensemble' in self.feature_selector.selection_results:
                selected_features = self.feature_selector.selection_results['ensemble']['selected_features']
                selected_indices = [i for i, f in enumerate(self.feature_names) if f in selected_features]
                X_selected = self.X[:, selected_indices]
                
                # Temporarily replace features
                original_X = self.X.copy()
                original_features = self.feature_names.copy()
                self.X = X_selected
                self.feature_names = selected_features
                
                selected_results = self.run_supervised_analysis(
                    use_sample_aware_splitting=True, apply_augmentation=include_augmentation
                )
                
                benchmark_results['feature_selected'] = {
                    'approach': 'Feature selection + sample-aware',
                    'best_accuracy': max([r['test_accuracy'] for r in selected_results.values()]),
                    'feature_count': len(selected_features),
                    'data_leakage_risk': 'None'
                }
                
                # Restore original features
                self.X = original_X
                self.feature_names = original_features
        
        # 5. Hyperparameter optimization
        if optimize_hyperparameters:
            print("5. Hyperparameter optimization...")
            best_params = self.hyperopt.optimize_random_forest(self.X, self.y)
            self.results['hyperparameter_optimization']['random_forest'] = best_params
        
        # Store comprehensive results
        self.results['comprehensive_benchmark'] = benchmark_results
        
        # Create comparison visualization
        self._visualize_benchmark_results(benchmark_results)
        
        print("Comprehensive benchmarking complete!")
        return benchmark_results
    
    def _visualize_benchmark_results(self, benchmark_results):
        """Create visualization of benchmark results."""
        import matplotlib.pyplot as plt
        
        approaches = list(benchmark_results.keys())
        accuracies = [benchmark_results[app]['best_accuracy'] for app in approaches]
        feature_counts = [benchmark_results[app]['feature_count'] for app in approaches]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(approaches, accuracies, alpha=0.7, color='skyblue')
        ax1.set_title('Best Test Accuracy by Approach')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Feature count comparison
        bars2 = ax2.bar(approaches, feature_counts, alpha=0.7, color='lightcoral')
        ax2.set_title('Number of Features by Approach')
        ax2.set_ylabel('Feature Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars2, feature_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_advanced_visualization_suite(self):
        """
        Run comprehensive advanced visualization analysis.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running advanced visualization suite...")
        
        # Create comprehensive visualization report
        self.advanced_visualizer.create_comprehensive_visualization_report(
            self.X, self.y, self.feature_names
        )
        
        # Feature space evolution (before/after scaling)
        X_scaled = self.scaler.fit_transform(self.X)
        self.advanced_visualizer.plot_feature_space_evolution(
            self.X, X_scaled, self.y, method='pca', title_suffix='Scaling'
        )
        
        self.results['advanced_visualizations'] = [
            'dimensionality_reduction_comparison.png',
            'tsne_analysis.png',
            'umap_analysis.png',
            'feature_space_evolution_pca_scaling.png'
        ]
        
        print("Advanced visualization suite complete!")
        return self.results['advanced_visualizations']
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report.
        """
        print("Generating comprehensive analysis report...")
        
        report_content = []
        report_content.append("# Comprehensive Texture Analysis Report")
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Data summary
        if self.X is not None:
            report_content.append("## Data Summary")
            report_content.append(f"- Total samples: {self.X.shape[0]}")
            report_content.append(f"- Total features: {self.X.shape[1]}")
            if self.y is not None:
                unique_labels, counts = np.unique(self.y, return_counts=True)
                report_content.append(f"- Classes: {len(unique_labels)}")
                for label, count in zip(unique_labels, counts):
                    report_content.append(f"  - Class {label}: {count} samples")
            report_content.append("")
        
        # Feature selection results
        if 'feature_selection_results' in self.results:
            report_content.append("## Feature Selection Results")
            for method, results in self.results['feature_selection_results'].items():
                report_content.append(f"### {method.title()}")
                report_content.append(f"- Selected features: {results['n_features']}")
                if 'feature_scores' in results:
                    top_features = sorted(results['feature_scores'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]
                    report_content.append("- Top 5 features:")
                    for feat, score in top_features:
                        report_content.append(f"  - {feat}: {score:.4f}")
                report_content.append("")
        
        # Model performance
        if 'supervised_results' in self.results:
            report_content.append("## Model Performance")
            for model_name, results in self.results['supervised_results'].items():
                report_content.append(f"### {model_name.replace('_', ' ').title()}")
                report_content.append(f"- Test Accuracy: {results['test_accuracy']:.4f}")
                report_content.append(f"- CV Mean: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
                report_content.append(f"- F1 Score: {results['f1_score']:.4f}")
                report_content.append("")
        
        # Comprehensive benchmark
        if 'comprehensive_benchmark' in self.results:
            report_content.append("## Comprehensive Benchmark Results")
            for approach, results in self.results['comprehensive_benchmark'].items():
                report_content.append(f"### {approach.replace('_', ' ').title()}")
                report_content.append(f"- Approach: {results['approach']}")
                report_content.append(f"- Best Accuracy: {results['best_accuracy']:.4f}")
                report_content.append(f"- Feature Count: {results['feature_count']}")
                report_content.append(f"- Data Leakage Risk: {results['data_leakage_risk']}")
                report_content.append("")
        
        # Recommendations
        report_content.append("## Recommendations")
        
        if 'comprehensive_benchmark' in self.results:
            best_approach = max(self.results['comprehensive_benchmark'].items(), 
                              key=lambda x: x[1]['best_accuracy'])
            report_content.append(f"- Best performing approach: {best_approach[0]} "
                                f"(Accuracy: {best_approach[1]['best_accuracy']:.4f})")
        
        report_content.append("- Always use sample-aware splitting to prevent data leakage")
        report_content.append("- Consider feature augmentation for small datasets")
        report_content.append("- Apply feature selection to reduce overfitting")
        report_content.append("- Use ensemble methods for robust predictions")
        report_content.append("")
        
        # Write report
        report_path = self.output_dir / "comprehensive_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path 