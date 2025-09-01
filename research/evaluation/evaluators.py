import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

class ModelEvaluator:
    """
    Comprehensive model evaluation suite for research analysis
    """
    
    def __init__(self, output_dir):
        """
        Initialize the model evaluator
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save evaluation outputs
        """
        self.output_dir = output_dir
        self.evaluation_history = {}
        
    def test_model_robustness(self, X, y, trained_models, noise_levels=[0.01, 0.05, 0.1]):
        """
        Test model robustness by adding Gaussian noise to features
        
        Parameters:
        -----------
        X : np.ndarray
            Original feature matrix
        y : np.ndarray
            Target labels
        trained_models : dict
            Dictionary of trained models
        noise_levels : list
            Standard deviations for Gaussian noise
            
        Returns:
        --------
        dict : Robustness test results
        """
        print("Testing model robustness with noise injection...")
        
        robustness_results = {}
        
        for model_name, model_info in trained_models.items():
            print(f"  Testing {model_name}...")
            
            model = model_info['model']
            robustness_results[model_name] = {}
            
            # Test at each noise level
            for noise_level in noise_levels:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, X.shape)
                X_noisy = X + noise
                
                # Make predictions on noisy data
                try:
                    y_pred = model.predict(X_noisy)
                    accuracy = accuracy_score(y, y_pred)
                    f1 = f1_score(y, y_pred, average='weighted')
                    
                    # Get prediction probabilities if available
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_noisy)
                        if len(np.unique(y)) == 2:  # Binary classification
                            auc = roc_auc_score(y, y_proba[:, 1])
                        else:
                            auc = None
                    else:
                        auc = None
                    
                    robustness_results[model_name][noise_level] = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'auc': auc,
                        'predictions': y_pred
                    }
                    
                except Exception as e:
                    print(f"    Error testing {model_name} at noise level {noise_level}: {e}")
                    robustness_results[model_name][noise_level] = {
                        'accuracy': 0.0,
                        'f1_score': 0.0,
                        'auc': None,
                        'error': str(e)
                    }
        
        return robustness_results
    
    def generate_learning_curves(self, model, X, y, cv=5, train_sizes=None):
        """
        Generate learning curves to analyze model performance vs training set size
        
        Parameters:
        -----------
        model : sklearn model
            Model to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        cv : int
            Number of cross-validation folds
        train_sizes : array-like, optional
            Training set sizes to evaluate
            
        Returns:
        --------
        dict : Learning curve data
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, scoring='accuracy',
            random_state=42, n_jobs=-1
        )
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
    
    def explain_predictions_shap(self, model, X_train, X_test, feature_names, sample_size=100):
        """
        Generate SHAP explanations for model predictions
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_train : np.ndarray
            Training data for SHAP background
        X_test : np.ndarray
            Test data to explain
        feature_names : list
            Names of features
        sample_size : int
            Number of samples to explain
            
        Returns:
        --------
        dict : SHAP explanation results
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for explanations")
            return None
        
        print("Generating SHAP explanations...")
        
        try:
            # Sample data if too large
            if len(X_test) > sample_size:
                indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_explain = X_test[indices]
            else:
                X_explain = X_test
            
            # Create SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For probabilistic models
                explainer = shap.Explainer(model.predict_proba, X_train)
            else:
                # For other models
                explainer = shap.Explainer(model.predict, X_train)
            
            # Calculate SHAP values
            shap_values = explainer(X_explain)
            
            # Calculate feature importance (mean absolute SHAP values)
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:  # Multi-class
                    feature_importance = np.mean(np.abs(shap_values.values), axis=(0, 2))
                else:  # Binary or regression
                    feature_importance = np.mean(np.abs(shap_values.values), axis=0)
            else:
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'X_explained': X_explain
            }
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            return None
    
    def explain_predictions_lime(self, model, X_train, X_test, feature_names, 
                                class_names=None, sample_size=10):
        """
        Generate LIME explanations for model predictions
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_train : np.ndarray
            Training data for LIME background
        X_test : np.ndarray
            Test data to explain
        feature_names : list
            Names of features
        class_names : list, optional
            Names of classes
        sample_size : int
            Number of samples to explain
            
        Returns:
        --------
        dict : LIME explanation results
        """
        if not LIME_AVAILABLE:
            print("LIME not available for explanations")
            return None
        
        print("Generating LIME explanations...")
        
        try:
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification',
                random_state=42
            )
            
            # Sample data if too large
            if len(X_test) > sample_size:
                indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_explain = X_test[indices]
            else:
                X_explain = X_test
            
            explanations = []
            
            for i, instance in enumerate(X_explain):
                exp = explainer.explain_instance(
                    instance, 
                    model.predict_proba,
                    num_features=len(feature_names)
                )
                explanations.append(exp)
            
            return {
                'explanations': explanations,
                'feature_names': feature_names,
                'X_explained': X_explain
            }
            
        except Exception as e:
            print(f"Error generating LIME explanations: {e}")
            return None
    
    def comprehensive_model_evaluation(self, model, X_train, X_test, y_train, y_test,
                                     feature_names, class_names=None):
        """
        Comprehensive evaluation of a single model
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_train, X_test : np.ndarray
            Training and test data
        y_train, y_test : np.ndarray
            Training and test labels
        feature_names : list
            Names of features
        class_names : list, optional
            Names of classes
            
        Returns:
        --------
        dict : Comprehensive evaluation results
        """
        print("Performing comprehensive model evaluation...")
        
        evaluation_results = {}
        
        # Basic predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Basic metrics
        evaluation_results['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        evaluation_results['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        evaluation_results['train_f1'] = f1_score(y_train, y_pred_train, average='weighted')
        evaluation_results['test_f1'] = f1_score(y_test, y_pred_test, average='weighted')
        
        # Detailed classification metrics
        evaluation_results['classification_report'] = classification_report(
            y_test, y_pred_test, output_dict=True
        )
        evaluation_results['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            y_proba_test = model.predict_proba(X_test)[:, 1]
            evaluation_results['roc_auc'] = roc_auc_score(y_test, y_proba_test)
            
            # ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_proba_test)
            evaluation_results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba_test)
            evaluation_results['pr_curve'] = {'precision': precision, 'recall': recall}
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        evaluation_results['cv_scores'] = cv_scores
        evaluation_results['cv_mean'] = cv_scores.mean()
        evaluation_results['cv_std'] = cv_scores.std()
        
        # Learning curves
        evaluation_results['learning_curves'] = self.generate_learning_curves(
            model, X_train, y_train
        )
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            evaluation_results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            evaluation_results['feature_importance'] = np.abs(model.coef_[0])
        
        # SHAP explanations
        shap_results = self.explain_predictions_shap(
            model, X_train, X_test, feature_names
        )
        if shap_results:
            evaluation_results['shap_explanations'] = shap_results
        
        # LIME explanations
        lime_results = self.explain_predictions_lime(
            model, X_train, X_test, feature_names, class_names
        )
        if lime_results:
            evaluation_results['lime_explanations'] = lime_results
        
        return evaluation_results
    
    def compare_models_statistical(self, model_results, metric='test_accuracy'):
        """
        Perform statistical comparison of models
        
        Parameters:
        -----------
        model_results : dict
            Results from multiple models
        metric : str
            Metric to compare
            
        Returns:
        --------
        dict : Statistical comparison results
        """
        print(f"Performing statistical comparison of models using {metric}...")
        
        from scipy import stats
        
        # Extract cross-validation scores for each model
        cv_scores = {}
        for model_name, results in model_results.items():
            if 'cv_scores' in results:
                cv_scores[model_name] = results['cv_scores']
        
        if len(cv_scores) < 2:
            print("Need at least 2 models with CV scores for comparison")
            return None
        
        # Pairwise statistical tests
        model_names = list(cv_scores.keys())
        comparison_results = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # Paired t-test
                scores1 = cv_scores[model1]
                scores2 = cv_scores[model2]
                
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                comparison_results[f'{model1}_vs_{model2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'mean_diff': np.mean(scores1) - np.mean(scores2),
                    'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
                        (np.var(scores1) + np.var(scores2)) / 2
                    )
                }
        
        return comparison_results
    
    def analyze_prediction_errors(self, model, X_test, y_test, feature_names):
        """
        Analyze prediction errors to understand model failures
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        feature_names : list
            Names of features
            
        Returns:
        --------
        dict : Error analysis results
        """
        print("Analyzing prediction errors...")
        
        y_pred = model.predict(X_test)
        
        # Identify incorrect predictions
        error_mask = y_pred != y_test
        error_indices = np.where(error_mask)[0]
        
        if len(error_indices) == 0:
            return {'perfect_predictions': True, 'error_count': 0}
        
        # Get prediction probabilities for error confidence analysis
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            max_proba = np.max(y_proba, axis=1)
            error_confidences = max_proba[error_mask]
        else:
            error_confidences = None
        
        # Analyze feature patterns in errors
        error_features = X_test[error_mask]
        correct_features = X_test[~error_mask]
        
        # Calculate feature statistics for errors vs correct predictions
        feature_analysis = {}
        for i, feature_name in enumerate(feature_names):
            error_vals = error_features[:, i]
            correct_vals = correct_features[:, i]
            
            feature_analysis[feature_name] = {
                'error_mean': np.mean(error_vals),
                'correct_mean': np.mean(correct_vals),
                'error_std': np.std(error_vals),
                'correct_std': np.std(correct_vals),
                'difference': np.mean(error_vals) - np.mean(correct_vals)
            }
        
        return {
            'perfect_predictions': False,
            'error_count': len(error_indices),
            'error_rate': len(error_indices) / len(y_test),
            'error_indices': error_indices,
            'predicted_labels': y_pred[error_mask],
            'true_labels': y_test[error_mask],
            'error_confidences': error_confidences,
            'feature_analysis': feature_analysis
        }
    
    def validate_model_stability(self, model_class, X, y, n_runs=10, **model_params):
        """
        Test model stability across multiple training runs
        
        Parameters:
        -----------
        model_class : class
            Model class to instantiate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        n_runs : int
            Number of training runs
        **model_params : dict
            Parameters for model initialization
            
        Returns:
        --------
        dict : Stability analysis results
        """
        print(f"Testing model stability across {n_runs} runs...")
        
        from sklearn.model_selection import train_test_split
        
        scores = []
        feature_importances = []
        
        for run in range(n_runs):
            # Random train-test split for each run
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=run, stratify=y
            )
            
            # Train model
            model = model_class(random_state=run, **model_params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)
        
        scores = np.array(scores)
        
        stability_results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'coefficient_of_variation': np.std(scores) / np.mean(scores),
            'stable': np.std(scores) < 0.05  # Consider stable if std < 5%
        }
        
        if feature_importances:
            feature_importances = np.array(feature_importances)
            stability_results['feature_importance_stability'] = {
                'mean_importance': np.mean(feature_importances, axis=0),
                'std_importance': np.std(feature_importances, axis=0),
                'cv_importance': np.std(feature_importances, axis=0) / 
                               (np.mean(feature_importances, axis=0) + 1e-8)
            }
        
        return stability_results
    
    def save_evaluation_report(self, evaluation_results, filename="evaluation_report.txt"):
        """
        Save comprehensive evaluation report to file
        
        Parameters:
        -----------
        evaluation_results : dict
            Results from model evaluation
        filename : str
            Output filename
        """
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, results in evaluation_results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                # Basic metrics
                if 'test_accuracy' in results:
                    f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
                if 'test_f1' in results:
                    f.write(f"Test F1-Score: {results['test_f1']:.4f}\n")
                if 'cv_mean' in results:
                    f.write(f"CV Mean: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
                if 'roc_auc' in results:
                    f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
                
                f.write("\n")
        
        print(f"Evaluation report saved to {report_path}") 