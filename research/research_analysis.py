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
import hashlib
import json

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

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

# Import new utilities
from .utils.determinism import set_global_seeds
from .utils.data_manifest import load_manifest, encode_labels, validate_labels
from .utils.schema_migration import auto_fix_schema, check_feature_schema
from .utils.split_reporting import split_stats
from .utils.result_serializer import serialize_results
from .utils.captions import build_caption
from .evaluation.stat_tests import compare_configurations, ci95
from .evaluation.diagnostics import confusion_matrix_report, per_class_metrics
from .evaluation.feature_importance import compute_permutation_importance, analyze_feature_correlations

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
    
    def __init__(self, data_path="data/features.csv", output_dir="research_output", 
                 manifest_path=None, deep_features_path=None):
        """
        Initialize the research analyzer for fabric analysis
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing extracted features
        output_dir : str
            Directory to save all research outputs
        manifest_path : str, optional
            Path to manifest CSV with filename,label columns for label integrity
        deep_features_path : str, optional
            Path to deep features CSV (if using deep or hybrid modes)
        """
        self.data_path = data_path
        self.manifest_path = manifest_path
        self.deep_features_path = deep_features_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "evaluation").mkdir(exist_ok=True)
        (self.output_dir / "statistical_analysis").mkdir(exist_ok=True)
        (self.output_dir / "splits").mkdir(exist_ok=True)
        
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
        self.deep_features_df = None
        self.manifest_dict = None
        self.class_names = None
        self.X = None
        self.y = None
        self.y_encoded = None
        self.feature_names = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        
        # Configuration
        self.config = {
            'data_path': data_path,
            'manifest_path': manifest_path,
            'deep_features_path': deep_features_path,
            'output_dir': str(output_dir)
        }
        
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
    
    def load_and_prepare_data(self, target_column=None, create_synthetic_labels=True,
                            use_manifest=True, migrate_schema=True):
        """
        Load feature data and prepare for analysis with manifest support and schema migration
        
        Parameters:
        -----------
        target_column : str, optional
            Column name to use as target variable (default: 'label')
        create_synthetic_labels : bool
            If True and no target_column, create synthetic labels for demonstration
        use_manifest : bool
            If True and manifest_path provided, use manifest for label integrity
        migrate_schema : bool
            If True, auto-fix schema by removing legacy features
        """
        print("Loading and preparing feature data...")
        
        # Load features CSV
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Features file not found: {self.data_path}")
        
        self.features_df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.features_df)} samples with {len(self.features_df.columns)} columns")
        
        # Migrate schema if requested
        if migrate_schema:
            print("Migrating feature schema...")
            self.features_df = auto_fix_schema(self.features_df, log_migrations=True)
        
        # Load manifest if provided
        if use_manifest and self.manifest_path and os.path.exists(self.manifest_path):
            print(f"Loading manifest from {self.manifest_path}...")
            try:
                self.manifest_dict, self.class_names = load_manifest(self.manifest_path)
                print(f"Loaded manifest with {len(self.manifest_dict)} entries, {len(self.class_names)} classes")
                
                # Validate labels
                mismatches = validate_labels(self.features_df, self.manifest_dict)
                if mismatches:
                    print(f"Warning: Found {len(mismatches)} label mismatches")
                else:
                    print("✓ Label validation passed")
                
                # Override labels from manifest
                if 'filename' in self.features_df.columns:
                    self.features_df['label'] = self.features_df['filename'].map(self.manifest_dict)
                    # Remove rows without labels in manifest
                    self.features_df = self.features_df.dropna(subset=['label'])
                    print(f"Updated labels from manifest: {len(self.features_df)} samples")
            except Exception as e:
                print(f"Warning: Manifest loading failed: {e}")
                use_manifest = False
        
        # Merge deep features if provided (do this early to ensure alignment)
        if self.deep_features_path and Path(self.deep_features_path).exists():
            print(f"Loading deep features from {self.deep_features_path}...")
            try:
                deep_df = pd.read_csv(self.deep_features_path)
                print(f"Loaded deep features: {deep_df.shape}")
                
                # Merge with main dataframe on filename
                if 'filename' in self.features_df.columns and 'filename' in deep_df.columns:
                    # Store original length
                    original_len = len(self.features_df)
                    
                    # Merge
                    self.features_df = self.features_df.merge(
                        deep_df, on='filename', how='inner', suffixes=('', '_deep')
                    )
                    
                    if len(self.features_df) != original_len:
                        print(f"Warning: Merge reduced samples from {original_len} to {len(self.features_df)}")
                    else:
                        print(f"Successfully merged deep features: {len(self.features_df)} samples")
                    
                    # Store deep feature column names for later use
                    self.deep_feature_cols = [col for col in deep_df.columns 
                                            if col.startswith('feat_') and col not in ['filename', 'label']]
                    print(f"Deep feature columns: {len(self.deep_feature_cols)} features")
                else:
                    print("Warning: Cannot merge deep features (filename column missing)")
                    self.deep_feature_cols = None
            except Exception as e:
                print(f"Warning: Failed to merge deep features: {e}")
                self.deep_feature_cols = None
        
        # Use target_column or default to 'label'
        if target_column is None:
            target_column = 'label' if 'label' in self.features_df.columns else None
        
        # Process data using DataProcessor
        self.X, self.y, self.feature_names = self.data_processor.prepare_features(
            self.features_df, target_column, create_synthetic_labels
        )
        
        # Encode labels if they're strings (after merge, so indices align)
        if self.y is not None:
            if isinstance(self.y[0], str) or (hasattr(self.y, 'dtype') and self.y.dtype == object):
                self.y_encoded, self.label_encoder, self.class_names = encode_labels(
                    self.y, encoder_path=str(self.output_dir / "label_encoder.pkl")
                )
                print(f"Encoded labels: {len(self.class_names)} classes")
                print(f"Classes: {self.class_names}")
            else:
                self.y_encoded = self.y
                # Create encoder from unique labels
                unique_labels = np.unique(self.y)
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(unique_labels)
                self.class_names = self.label_encoder.classes_.tolist()
            
            # Verify alignment
            assert len(self.y_encoded) == len(self.features_df), \
                f"Label length ({len(self.y_encoded)}) doesn't match features_df length ({len(self.features_df)})"
            assert len(self.X) == len(self.features_df), \
                f"Feature matrix length ({len(self.X)}) doesn't match features_df length ({len(self.features_df)})"
        
        print(f"Prepared feature matrix: {self.X.shape}")
        print(f"Feature names: {len(self.feature_names)}")
        
        if self.y_encoded is not None:
            unique_labels = np.unique(self.y_encoded)
            print(f"Target labels: {unique_labels} (counts: {np.bincount(self.y_encoded)})")
        
        return self.X, self.y_encoded, self.feature_names
    
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
        if self.X is None or self.y_encoded is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("Running enhanced supervised learning analysis...")
        
        # Split data using sample-aware splitting or traditional splitting
        if use_sample_aware_splitting and self.features_df is not None:
            print("Using sample-aware splitting to prevent data leakage...")
            try:
                train_idx, val_idx, test_idx, leak_summary = self.sample_splitter.split_by_samples(
                    self.features_df, test_size=test_size, val_size=val_size, 
                    stratify_column='label', seed=42
                )
                
                X_train, X_val, X_test = self.X[train_idx], self.X[val_idx], self.X[test_idx]
                y_train, y_val, y_test = self.y_encoded[train_idx], self.y_encoded[val_idx], self.y_encoded[test_idx]
                
                # Validate split (already done in split_by_samples via leak_check)
                print(f"Leak check: {'✓ PASSED' if leak_summary['is_valid'] else '✗ FAILED'}")
                
            except Exception as e:
                print(f"Sample-aware splitting failed ({e}), falling back to random splitting")
                use_sample_aware_splitting = False
        
        if not use_sample_aware_splitting:
            print("Using traditional random splitting...")
            X_temp, X_test, y_temp, y_test = train_test_split(
                self.X, self.y_encoded, test_size=test_size, random_state=42, stratify=self.y_encoded
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
            X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names, cv_folds, encoder=self.label_encoder
        )
        
        # Generate model comparison visualization
        self.visualizer.plot_model_comparison(self.results['supervised_results'])
        
        # Advanced visualizations
        self.advanced_visualizer.plot_dimensionality_reduction_comparison(
            X_train_scaled, y_train, class_names=None
        )
        
        print("Enhanced supervised analysis complete!")
        return self.results['supervised_results']
    
    def _prepare_features_with_fusion(self, feature_mode='handcrafted', fusion_strategy='concatenate'):
        """
        Prepare features based on mode: handcrafted, deep-only, or hybrid.
        
        Parameters:
        -----------
        feature_mode : str
            'handcrafted', 'deep_only', or 'hybrid'
        fusion_strategy : str
            'concatenate', 'weighted', or 'attention' (for hybrid mode)
            
        Returns:
        --------
        tuple : (X, feature_names)
        """
        if feature_mode == 'handcrafted':
            return self.X, self.feature_names
        
        elif feature_mode == 'deep_only':
            # Deep features should already be merged in load_and_prepare_data
            if not hasattr(self, 'deep_feature_cols') or self.deep_feature_cols is None:
                raise ValueError("Deep features not available. Ensure deep_features_path is provided and merge succeeded.")
            
            # Extract deep feature columns from already-merged dataframe
            X_deep = self.features_df[self.deep_feature_cols].values
            
            # Scale deep features
            scaler_deep = StandardScaler()
            X_deep_scaled = scaler_deep.fit_transform(X_deep)
            
            print(f"Using deep-only features: {X_deep_scaled.shape}")
            return X_deep_scaled, self.deep_feature_cols
        
        elif feature_mode == 'hybrid':
            # Deep features should already be merged in load_and_prepare_data
            if not hasattr(self, 'deep_feature_cols') or self.deep_feature_cols is None:
                raise ValueError("Deep features not available for hybrid mode. Ensure deep_features_path is provided.")
            
            # Use existing handcrafted features (already extracted)
            X_handcrafted = self.X
            
            # Extract deep features from already-merged dataframe
            X_deep = self.features_df[self.deep_feature_cols].values
            
            # Scale both feature sets
            scaler_hand = StandardScaler()
            scaler_deep = StandardScaler()
            X_handcrafted_scaled = scaler_hand.fit_transform(X_handcrafted)
            X_deep_scaled = scaler_deep.fit_transform(X_deep)
            
            print(f"Hybrid features: handcrafted {X_handcrafted_scaled.shape}, deep {X_deep_scaled.shape}")
            
            # Fuse features
            if fusion_strategy == 'concatenate':
                X_fused = np.hstack([X_handcrafted_scaled, X_deep_scaled])
                feature_names_fused = list(self.feature_names) + self.deep_feature_cols
            elif fusion_strategy == 'weighted':
                # Simple weighted combination (can be enhanced with learned weights)
                # Normalize weights so they sum to 1
                weight_hand = 0.5
                weight_deep = 0.5
                X_fused = np.hstack([X_handcrafted_scaled * weight_hand, X_deep_scaled * weight_deep])
                feature_names_fused = list(self.feature_names) + self.deep_feature_cols
            elif fusion_strategy == 'attention':
                # Placeholder for attention-based fusion
                # For now, use concatenate
                X_fused = np.hstack([X_handcrafted_scaled, X_deep_scaled])
                feature_names_fused = list(self.feature_names) + self.deep_feature_cols
                print("Note: Attention fusion not yet implemented, using concatenate")
            else:
                # Default to concatenate
                X_fused = np.hstack([X_handcrafted_scaled, X_deep_scaled])
                feature_names_fused = list(self.feature_names) + self.deep_feature_cols
            
            print(f"Fused features: {X_fused.shape} (fusion strategy: {fusion_strategy})")
            return X_fused, feature_names_fused
        
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
    
    def run_supervised_analysis_multiseed(self, test_size=0.2, val_size=0.1, cv_folds=5,
                                         n_seeds=5, seeds=None, feature_mode='handcrafted',
                                         fusion_strategy='concatenate', save_splits_dir=None,
                                         use_saved_splits=False, stratify_column='label'):
        """
        Run supervised analysis across multiple seeds with support for different feature modes.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        val_size : float
            Proportion of data for validation
        cv_folds : int
            Number of cross-validation folds
        n_seeds : int
            Number of seeds to use
        seeds : List[int], optional
            List of seeds to use
        feature_mode : str
            'handcrafted', 'deep_only', or 'hybrid'
        fusion_strategy : str
            'concatenate', 'weighted', or 'attention' (for hybrid mode)
        save_splits_dir : str, optional
            Directory to save/load splits
        use_saved_splits : bool
            If True, try to load saved splits
        stratify_column : str
            Column name for stratification
            
        Returns:
        --------
        dict : Results with 'aggregated' and 'per_seed' keys
        """
        if self.X is None or self.y_encoded is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        # Prepare features based on mode
        X, feature_names = self._prepare_features_with_fusion(feature_mode, fusion_strategy)
        
        # Default seeds
        if seeds is None:
            seeds = [42, 123, 456, 789, 1011][:n_seeds]
        
        print(f"Running supervised analysis with {feature_mode} features across {len(seeds)} seeds...")
        
        # Create config hash
        config_str = json.dumps({
            'test_size': test_size,
            'val_size': val_size,
            'feature_mode': feature_mode,
            'fusion_strategy': fusion_strategy,
            'stratify_column': stratify_column
        }, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Create split key
        class_list = self.class_names if self.class_names else sorted(np.unique(self.y_encoded).tolist())
        data_hash = hashlib.sha256(str(sorted(self.features_df['filename'].values if 'filename' in self.features_df.columns else range(len(self.features_df)))).encode()).hexdigest()[:16]
        split_key = self.sample_splitter.make_split_key(seeds[0], 'sample_aware', class_list, data_hash)
        
        per_seed_results = {}
        all_metrics = {}
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'='*80}")
            print(f"Seed {seed_idx+1}/{len(seeds)}: {seed}")
            print(f"{'='*80}")
            
            # Set global seed
            set_global_seeds(seed)
            
            # Try to load splits if requested
            train_idx = None
            val_idx = None
            test_idx = None
            
            if use_saved_splits and save_splits_dir:
                split_dir = Path(save_splits_dir) / f"seed_{seed}"
                loaded = self.sample_splitter.load_split_indices(str(split_dir), split_key)
                if loaded:
                    train_idx, val_idx, test_idx, metadata = loaded
                    print(f"Loaded splits from {split_dir}")
            
            # Create splits if not loaded
            if train_idx is None:
                train_idx, val_idx, test_idx, leak_summary = self.sample_splitter.split_by_samples(
                    self.features_df, test_size=test_size, val_size=val_size,
                    stratify_column=stratify_column, seed=seed
                )
                
                # Save splits if requested
                if save_splits_dir:
                    split_dir = Path(save_splits_dir) / f"seed_{seed}"
                    split_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Compute label counts
                    y_train_labels = self.y_encoded[train_idx]
                    y_val_labels = self.y_encoded[val_idx]
                    y_test_labels = self.y_encoded[test_idx]
                    label_counts = {
                        'train': {cls: int(np.sum(y_train_labels == i)) for i, cls in enumerate(class_list)},
                        'val': {cls: int(np.sum(y_val_labels == i)) for i, cls in enumerate(class_list)},
                        'test': {cls: int(np.sum(y_test_labels == i)) for i, cls in enumerate(class_list)}
                    }
                    
                    self.sample_splitter.save_split_indices(
                        train_idx, val_idx, test_idx,
                        str(split_dir), config_hash, seed, split_key, leak_summary, label_counts
                    )
                    print(f"Saved splits to {split_dir}")
            
            # Get splits for this seed
            X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
            y_train, y_val, y_test = self.y_encoded[train_idx], self.y_encoded[val_idx], self.y_encoded[test_idx]
            
            # Scale features (if not already scaled in _prepare_features_with_fusion)
            if feature_mode == 'handcrafted':
                self.scaler.fit(X_train)
                X_train_scaled = self.scaler.transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                # Already scaled in _prepare_features_with_fusion
                X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
            
            # Train models
            model_suite = SupervisedModelSuite(random_state=seed)
            seed_results = model_suite.train_all_models(
                X_train_scaled, X_test_scaled, y_train, y_test, feature_names, cv_folds, encoder=self.label_encoder
            )
            
            per_seed_results[seed] = seed_results
            
            # Collect metrics
            for model_name, results in seed_results.items():
                if model_name not in all_metrics:
                    all_metrics[model_name] = []
                all_metrics[model_name].append({
                    'accuracy': results['test_accuracy'],
                    'macro_f1': results.get('macro_f1', results.get('f1_score', 0)),
                    'macro_auc': results.get('macro_auc', results.get('test_auc'))
                })
        
        # Aggregate results
        aggregated_results = {}
        for model_name, seed_metrics in all_metrics.items():
            if not seed_metrics:
                continue
            
            accuracies = [m['accuracy'] for m in seed_metrics if m['accuracy'] is not None]
            macro_f1s = [m['macro_f1'] for m in seed_metrics if m['macro_f1'] is not None]
            macro_aucs = [m['macro_auc'] for m in seed_metrics if m['macro_auc'] is not None]
            
            if accuracies:
                acc_mean, acc_std, acc_ci_lower, acc_ci_upper = ci95(np.array(accuracies))
            else:
                acc_mean = acc_std = acc_ci_lower = acc_ci_upper = None
            
            if macro_f1s:
                f1_mean, f1_std, f1_ci_lower, f1_ci_upper = ci95(np.array(macro_f1s))
            else:
                f1_mean = f1_std = f1_ci_lower = f1_ci_upper = None
            
            auc_mean = auc_std = auc_ci_lower = auc_ci_upper = None
            if macro_aucs and all(a is not None for a in macro_aucs):
                auc_mean, auc_std, auc_ci_lower, auc_ci_upper = ci95(np.array(macro_aucs))
            
            aggregated_results[model_name] = {
                'accuracy': {'mean': acc_mean, 'std': acc_std, 'ci_lower': acc_ci_lower, 'ci_upper': acc_ci_upper},
                'macro_f1': {'mean': f1_mean, 'std': f1_std, 'ci_lower': f1_ci_lower, 'ci_upper': f1_ci_upper},
                'macro_auc': {'mean': auc_mean, 'std': auc_std, 'ci_lower': auc_ci_lower, 'ci_upper': auc_ci_upper} if auc_mean else None
            }
        
        # Save aggregated results
        results_summary = {
            'aggregated': aggregated_results,
            'per_seed': per_seed_results,
            'config': {
                'feature_mode': feature_mode,
                'fusion_strategy': fusion_strategy,
                'seeds': seeds,
                'config_hash': config_hash
            }
        }
        
        # Serialize results
        serialize_results(results_summary, {**self.config, **results_summary['config']}, 
                         str(self.output_dir / "results"))
        
        # Store in results
        self.results['supervised_results'] = results_summary
        
        return results_summary
    
    def generate_significance_summary(self, results_handcrafted=None, results_deep=None, results_hybrid=None,
                                     output_path=None):
        """
        Generate significance summary comparing handcrafted, deep, and hybrid configurations.
        
        Parameters:
        -----------
        results_handcrafted : dict, optional
            Results from handcrafted features (from run_supervised_analysis_multiseed)
        results_deep : dict, optional
            Results from deep-only features
        results_hybrid : dict, optional
            Results from hybrid features
        output_path : str, optional
            Path to save significance_summary.csv
            
        Returns:
        --------
        pd.DataFrame : Significance summary with comparisons
        """
        from research.evaluation.stat_tests import compare_configurations
        
        comparisons = []
        
        # Compare handcrafted vs deep
        if results_handcrafted and results_deep:
            if 'per_seed' in results_handcrafted and 'per_seed' in results_deep:
                comp_df = compare_configurations(
                    results_handcrafted['per_seed'], results_deep['per_seed'], metric='macro_f1'
                )
                comp_df['comparison'] = 'handcrafted_vs_deep'
                comparisons.append(comp_df)
        
        # Compare handcrafted vs hybrid
        if results_handcrafted and results_hybrid:
            if 'per_seed' in results_handcrafted and 'per_seed' in results_hybrid:
                comp_df = compare_configurations(
                    results_handcrafted['per_seed'], results_hybrid['per_seed'], metric='macro_f1'
                )
                comp_df['comparison'] = 'handcrafted_vs_hybrid'
                comparisons.append(comp_df)
        
        # Compare deep vs hybrid
        if results_deep and results_hybrid:
            if 'per_seed' in results_deep and 'per_seed' in results_hybrid:
                comp_df = compare_configurations(
                    results_deep['per_seed'], results_hybrid['per_seed'], metric='macro_f1'
                )
                comp_df['comparison'] = 'deep_vs_hybrid'
                comparisons.append(comp_df)
        
        if comparisons:
            summary_df = pd.concat(comparisons, ignore_index=True)
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                summary_df.to_csv(output_path, index=False)
                print(f"Saved significance summary to {output_path}")
            
            return summary_df
        else:
            print("Warning: No comparisons available for significance summary")
            return pd.DataFrame()
    
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