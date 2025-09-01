import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, f1_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

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
        self.models['svm_linear'] = SVC(
            kernel='linear',
            random_state=self.random_state,
            class_weight='balanced',
            probability=True
        )
        
        # Support Vector Machine - RBF
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
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbosity=-1
            )
    
    def train_single_model(self, model_name, X_train, X_test, y_train, y_test, cv_folds=5):
        """
        Train a single model and evaluate performance
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train, X_test : np.ndarray
            Training and test feature matrices
        y_train, y_test : np.ndarray
            Training and test target vectors
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Model results including accuracy, cross-validation scores, etc.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        print(f"Training {model_name}...")
        
        model = self.models[model_name]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Get prediction probabilities (if available)
        try:
            y_pred_proba_test = model.predict_proba(X_test)
        except:
            y_pred_proba_test = None
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='weighted')
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # ROC AUC (for binary classification)
        test_auc = None
        if len(np.unique(y_test)) == 2 and y_pred_proba_test is not None:
            try:
                test_auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
            except:
                pass
        
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
            'f1_score': f1_test,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions_test': y_pred_test,
            'predictions_proba_test': y_pred_proba_test,
            'test_auc': test_auc
        }
        
        print(f"  ✓ {model_name} - Test Accuracy: {test_accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def train_all_models(self, X_train, X_test, y_train, y_test, feature_names=None, cv_folds=5):
        """
        Train all available models and compare performance
        
        Parameters:
        -----------
        X_train, X_test : np.ndarray
            Training and test feature matrices
        y_train, y_test : np.ndarray
            Training and test target vectors
        feature_names : list, optional
            Names of features
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Results for all models
        """
        print(f"Training {len(self.models)} models on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        
        self.results = {}
        
        for model_name in self.models.keys():
            try:
                self.results[model_name] = self.train_single_model(
                    model_name, X_train, X_test, y_train, y_test, cv_folds
                )
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {str(e)}")
                continue
        
        # Perform model comparison
        self._compare_models()
        
        return self.results
    
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
                'F1 Score': results['f1_score'],
                'AUC': results['test_auc'] if results['test_auc'] else 'N/A'
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