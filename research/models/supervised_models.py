import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import standardized metrics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import compute_macro_f1, compute_macro_auc, validate_probabilities
from evaluation.stat_tests import ci95

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class SupervisedModelSuite:
    """
    Comprehensive suite of supervised learning models for fabric analysis tasks
    including fabric type classification and fabric quality/damage classification
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the supervised model suite
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all supervised learning models with optimized hyperparameters"""
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Support Vector Machine - Linear
        # Note: probability=True enables predict_proba but requires calibration (O(n^2) cost)
        self.models['svm_linear'] = SVC(
            kernel='linear',
            random_state=self.random_state,
            class_weight='balanced',
            probability=True
        )
        
        # Support Vector Machine - RBF
        # Note: probability=True enables predict_proba but requires calibration (O(n^2) cost)
        self.models['svm_rbf'] = SVC(
            kernel='rbf',
            random_state=self.random_state,
            class_weight='balanced',
            probability=True,
            gamma='scale'
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            random_state=self.random_state,
            learning_rate=0.1,
            max_depth=3
        )
        
        # Multi-layer Perceptron (Neural Network)
        self.models['mlp_classifier'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=self.random_state,
            max_iter=500,
            alpha=0.001,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # XGBoost (if available)
        # Note: objective and num_class will be set during training based on number of classes
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        
        # LightGBM (if available)
        # Note: objective and num_class will be set during training based on number of classes
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbosity=-1
            )
    
    def train_single_model(self, model_name, X_train, X_test, y_train, y_test, 
                          cv_folds=5, encoder=None):
        """
        Train a single model and evaluate performance with standardized metrics.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train, X_test : np.ndarray
            Training and test feature matrices
        y_train, y_test : np.ndarray
            Training and test target vectors (integer-encoded)
        cv_folds : int
            Number of cross-validation folds
        encoder : LabelEncoder, optional
            Label encoder to infer number of classes for XGBoost/LightGBM
            
        Returns:
        --------
        dict : Model results including accuracy, macro-F1, macro-AUC, etc.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        print(f"Training {model_name}...")
        
        # Get model and configure for multi-class if needed
        model = self.models[model_name]
        
        # Infer number of classes
        n_classes = len(np.unique(y_train))
        
        # Configure XGBoost and LightGBM for multi-class
        if model_name == 'xgboost' and XGBOOST_AVAILABLE:
            model.set_params(objective='multi:softprob', num_class=n_classes)
        elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model.set_params(objective='multiclass', num_class=n_classes)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Get prediction probabilities (if available)
        y_pred_proba_test = None
        try:
            y_pred_proba_test = model.predict_proba(X_test)
            # Validate probabilities
            if y_pred_proba_test is not None:
                validate_probabilities(y_pred_proba_test, n_classes)
        except Exception as e:
            print(f"  Warning: Could not get probabilities for {model_name}: {e}")
            y_pred_proba_test = None
        
        # Calculate standardized metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        macro_f1_test = compute_macro_f1(y_test, y_pred_test)
        
        # Macro ROC-AUC (guarded)
        macro_auc_test = None
        if y_pred_proba_test is not None:
            try:
                macro_auc_test = compute_macro_auc(y_test, y_pred_proba_test)
            except Exception as e:
                print(f"  Warning: AUC computation failed for {model_name}: {e}")
                macro_auc_test = None
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Classification report
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        
        # Store results
        results = {
            'model': model,
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'macro_f1': macro_f1_test,
            'macro_auc': macro_auc_test,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions_test': y_pred_test,
            'predictions_proba_test': y_pred_proba_test
        }
        
        auc_str = f"{macro_auc_test:.4f}" if macro_auc_test is not None else "N/A"
        print(f"  ✓ {model_name} - Accuracy: {test_accuracy:.4f}, Macro-F1: {macro_f1_test:.4f}, Macro-AUC: {auc_str}")
        
        return results
    
    def train_all_models(self, X_train, X_test, y_train, y_test, feature_names=None, 
                        cv_folds=5, encoder=None):
        """
        Train all available models and compare performance
        
        Parameters:
        -----------
        X_train, X_test : np.ndarray
            Training and test feature matrices
        y_train, y_test : np.ndarray
            Training and test target vectors (integer-encoded)
        feature_names : list, optional
            Names of features
        cv_folds : int
            Number of cross-validation folds
        encoder : LabelEncoder, optional
            Label encoder to infer number of classes
            
        Returns:
        --------
        dict : Results for all models
        """
        print(f"Training {len(self.models)} models on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        
        self.results = {}
        
        for model_name in self.models.keys():
            try:
                self.results[model_name] = self.train_single_model(
                    model_name, X_train, X_test, y_train, y_test, cv_folds, encoder
                )
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {str(e)}")
                continue
        
        # Perform model comparison
        self._compare_models()
        
        return self.results
    
    def train_all_models_multiseed(self, X_train, X_test, y_train, y_test, 
                                   feature_names=None, cv_folds=5, n_seeds=5, 
                                   seeds=None, encoder=None):
        """
        Train all models across multiple seeds and aggregate results.
        
        Parameters:
        -----------
        X_train, X_test : np.ndarray
            Training and test feature matrices
        y_train, y_test : np.ndarray
            Training and test target vectors (integer-encoded)
        feature_names : list, optional
            Names of features
        cv_folds : int
            Number of cross-validation folds
        n_seeds : int
            Number of seeds to use (if seeds not provided)
        seeds : List[int], optional
            List of seeds to use. If None, uses default seeds.
        encoder : LabelEncoder, optional
            Label encoder to infer number of classes
            
        Returns:
        --------
        dict : Results with 'aggregated' and 'per_seed' keys
        """
        if seeds is None:
            seeds = [42, 123, 456, 789, 1011][:n_seeds]
        
        print(f"Training {len(self.models)} models across {len(seeds)} seeds...")
        
        per_seed_results = {}
        all_results = {model_name: [] for model_name in self.models.keys()}
        
        for seed in seeds:
            print(f"\n=== Seed {seed} ===")
            # Reinitialize models with new seed
            old_random_state = self.random_state
            self.random_state = seed
            self._initialize_models()
            
            # Train all models
            seed_results = self.train_all_models(
                X_train, X_test, y_train, y_test, feature_names, cv_folds, encoder
            )
            
            per_seed_results[seed] = seed_results
            
            # Collect metrics
            for model_name, results in seed_results.items():
                all_results[model_name].append({
                    'accuracy': results['test_accuracy'],
                    'macro_f1': results['macro_f1'],
                    'macro_auc': results['macro_auc']
                })
            
            # Restore random state
            self.random_state = old_random_state
        
        # Aggregate results
        aggregated_results = {}
        for model_name, seed_metrics in all_results.items():
            if not seed_metrics:
                continue
            
            # Extract metric arrays
            accuracies = [m['accuracy'] for m in seed_metrics if m['accuracy'] is not None]
            macro_f1s = [m['macro_f1'] for m in seed_metrics if m['macro_f1'] is not None]
            macro_aucs = [m['macro_auc'] for m in seed_metrics if m['macro_auc'] is not None]
            
            # Compute statistics
            acc_mean, acc_std, acc_ci_lower, acc_ci_upper = ci95(np.array(accuracies))
            f1_mean, f1_std, f1_ci_lower, f1_ci_upper = ci95(np.array(macro_f1s))
            
            auc_mean = None
            auc_std = None
            auc_ci_lower = None
            auc_ci_upper = None
            if macro_aucs and all(a is not None for a in macro_aucs):
                auc_mean, auc_std, auc_ci_lower, auc_ci_upper = ci95(np.array(macro_aucs))
            
            aggregated_results[model_name] = {
                'accuracy': {
                    'mean': acc_mean,
                    'std': acc_std,
                    'ci_lower': acc_ci_lower,
                    'ci_upper': acc_ci_upper
                },
                'macro_f1': {
                    'mean': f1_mean,
                    'std': f1_std,
                    'ci_lower': f1_ci_lower,
                    'ci_upper': f1_ci_upper
                },
                'macro_auc': {
                    'mean': auc_mean,
                    'std': auc_std,
                    'ci_lower': auc_ci_lower,
                    'ci_upper': auc_ci_upper
                } if auc_mean is not None else None
            }
        
        return {
            'aggregated': aggregated_results,
            'per_seed': per_seed_results
        }
    
    def _compare_models(self):
        """Compare and rank models by performance"""
        if not self.results:
            return
        
        print("\nModel Performance Comparison:")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test Accuracy': results['test_accuracy'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std'],
                'Macro F1': results.get('macro_f1', results.get('f1_score', 'N/A')),
                'Macro AUC': results.get('macro_auc', results.get('test_auc', 'N/A'))
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Test Accuracy']
        print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    def get_feature_importance(self, model_name='random_forest'):
        """
        Get feature importance from tree-based models
        
        Parameters:
        -----------
        model_name : str
            Name of the model to get feature importance from
            
        Returns:
        --------
        np.ndarray or None : Feature importance scores
        """
        if model_name not in self.results:
            print(f"Model {model_name} not trained yet")
            return None
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            return np.abs(model.coef_[0])
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
    
    def save_models(self, save_dir):
        """
        Save all trained models to disk
        
        Parameters:
        -----------
        save_dir : str or Path
            Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, results in self.results.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.joblib")
            joblib.dump(results['model'], model_path)
            print(f"Saved {model_name} to {model_path}")
    
    def load_models(self, save_dir):
        """
        Load previously saved models from disk
        
        Parameters:
        -----------
        save_dir : str or Path
            Directory containing saved models
        """
        import os
        
        for model_name in self.models.keys():
            model_path = os.path.join(save_dir, f"{model_name}_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                if model_name not in self.results:
                    self.results[model_name] = {}
                self.results[model_name]['model'] = model
                print(f"Loaded {model_name} from {model_path}")
    
    def predict_new_samples(self, X_new, model_name='random_forest'):
        """
        Make predictions on new samples using a trained model
        
        Parameters:
        -----------
        X_new : np.ndarray
            New samples to predict
        model_name : str
            Name of the model to use for prediction
            
        Returns:
        --------
        tuple : (predictions, prediction_probabilities)
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.results[model_name]['model']
        predictions = model.predict(X_new)
        
        try:
            pred_proba = model.predict_proba(X_new)
        except:
            pred_proba = None
        
        return predictions, pred_proba
    
    def perform_ablation_study(self, X_train, X_test, y_train, y_test, 
                             feature_groups, feature_names=None):
        """
        Perform ablation study to understand feature group contributions
        
        Parameters:
        -----------
        X_train, X_test : np.ndarray
            Training and test feature matrices
        y_train, y_test : np.ndarray
            Training and test target vectors
        feature_groups : dict
            Dictionary mapping group names to feature indices
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        dict : Ablation study results
        """
        print("\nPerforming Ablation Study...")
        print("=" * 40)
        
        ablation_results = {}
        
        # Test with all features (baseline)
        baseline_results = self.train_single_model(
            'random_forest', X_train, X_test, y_train, y_test
        )
        ablation_results['all_features'] = {
            'accuracy': baseline_results['test_accuracy'],
            'features_used': 'All features'
        }
        
        # Test without each feature group
        for group_name, feature_indices in feature_groups.items():
            print(f"Testing without {group_name} features...")
            
            # Create feature mask (all True except for this group)
            feature_mask = np.ones(X_train.shape[1], dtype=bool)
            feature_mask[feature_indices] = False
            
            # Train without this feature group
            X_train_reduced = X_train[:, feature_mask]
            X_test_reduced = X_test[:, feature_mask]
            
            results = self.train_single_model(
                'random_forest', X_train_reduced, X_test_reduced, y_train, y_test
            )
            
            accuracy_drop = baseline_results['test_accuracy'] - results['test_accuracy']
            
            ablation_results[f'without_{group_name}'] = {
                'accuracy': results['test_accuracy'],
                'accuracy_drop': accuracy_drop,
                'features_removed': len(feature_indices),
                'importance_score': accuracy_drop  # Higher drop = more important
            }
            
            print(f"  Accuracy without {group_name}: {results['test_accuracy']:.4f} "
                  f"(drop: {accuracy_drop:.4f})")
        
        return ablation_results 