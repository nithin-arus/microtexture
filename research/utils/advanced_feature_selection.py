import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import (
    SelectKBest, RFE, RFECV, SelectFromModel,
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, chi2
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import joblib

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

try:
    from sklearn.feature_selection import VarianceThreshold
    VARIANCE_THRESHOLD_AVAILABLE = True
except ImportError:
    VARIANCE_THRESHOLD_AVAILABLE = False


class AdvancedFeatureSelector:
    """
    Advanced feature selection suite for texture analysis.
    
    Provides multiple sophisticated feature selection methods including
    mutual information, recursive feature elimination, stability selection,
    and hyperparameter optimization.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the advanced feature selector.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.selection_results = {}
        self.selected_features = {}
        
    def variance_threshold_selection(self, X: np.ndarray, 
                                   feature_names: List[str],
                                   threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
        """
        Remove features with low variance.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : List[str]
            Names of features
        threshold : float
            Variance threshold
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]] : Selected features and their names
        """
        if not VARIANCE_THRESHOLD_AVAILABLE:
            print("VarianceThreshold not available")
            return X, feature_names
        
        print(f"Applying variance threshold selection (threshold={threshold})...")
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"  Selected {len(selected_features)} features out of {len(feature_names)}")
        
        self.selection_results['variance_threshold'] = {
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'threshold': threshold,
            'support_mask': selected_mask
        }
        
        return X_selected, selected_features
    
    def mutual_information_selection(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   k: int = 20,
                                   task_type: str = 'classification') -> Tuple[np.ndarray, List[str]]:
        """
        Select features based on mutual information.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        feature_names : List[str]
            Names of features
        k : int
            Number of features to select
        task_type : str
            'classification' or 'regression'
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]] : Selected features and their names
        """
        print(f"Applying mutual information selection (k={k}, task={task_type})...")
        
        # Choose appropriate mutual information function
        if task_type == 'classification':
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression
        
        # Calculate mutual information scores
        mi_scores = mi_func(X, y, random_state=self.random_state)
        
        # Select top k features
        selector = SelectKBest(score_func=mi_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        # Store scores for analysis
        feature_scores = dict(zip(feature_names, mi_scores))
        
        print(f"  Selected {len(selected_features)} features")
        print(f"  Top 5 features: {sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        self.selection_results['mutual_information'] = {
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'support_mask': selected_mask,
            'k': k
        }
        
        return X_selected, selected_features
    
    def recursive_feature_elimination_cv(self, X: np.ndarray, y: np.ndarray,
                                       feature_names: List[str],
                                       estimator=None,
                                       cv: int = 5,
                                       min_features: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Cross-validated recursive feature elimination.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        feature_names : List[str]
            Names of features
        estimator : sklearn estimator, optional
            Estimator to use for feature ranking
        cv : int
            Number of cross-validation folds
        min_features : int
            Minimum number of features to select
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]] : Selected features and their names
        """
        print(f"Applying RFE-CV (cv={cv}, min_features={min_features})...")
        
        # Use RandomForest as default estimator
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # Perform RFE-CV
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            min_features_to_select=min_features,
            n_jobs=-1
        )
        
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"  Selected {len(selected_features)} features")
        print(f"  Optimal number of features: {selector.n_features_}")
        
        self.selection_results['rfe_cv'] = {
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'support_mask': selected_mask,
            'cv_scores': selector.cv_results_,
            'optimal_n_features': selector.n_features_
        }
        
        return X_selected, selected_features
    
    def stability_selection(self, X: np.ndarray, y: np.ndarray,
                          feature_names: List[str],
                          n_bootstrap: int = 100,
                          threshold: float = 0.6,
                          sample_fraction: float = 0.8) -> Tuple[np.ndarray, List[str]]:
        """
        Stability-based feature selection using bootstrap sampling.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        feature_names : List[str]
            Names of features
        n_bootstrap : int
            Number of bootstrap iterations
        threshold : float
            Selection frequency threshold
        sample_fraction : float
            Fraction of samples to use in each bootstrap
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]] : Selected features and their names
        """
        print(f"Applying stability selection (n_bootstrap={n_bootstrap}, threshold={threshold})...")
        
        n_samples, n_features = X.shape
        selection_counts = np.zeros(n_features)
        
        # Bootstrap sampling and feature selection
        for i in range(n_bootstrap):
            # Sample data
            sample_size = int(n_samples * sample_fraction)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Select features using Lasso
            try:
                selector = LassoCV(cv=3, random_state=self.random_state, max_iter=1000)
                selector.fit(X_boot, y_boot)
                
                # Count selected features (non-zero coefficients)
                selected_mask = np.abs(selector.coef_) > 1e-5
                selection_counts += selected_mask
                
            except Exception as e:
                print(f"Warning: Bootstrap iteration {i} failed: {e}")
                continue
        
        # Calculate selection frequencies
        selection_frequencies = selection_counts / n_bootstrap
        
        # Select stable features
        stable_mask = selection_frequencies >= threshold
        stable_features = [feature_names[i] for i, stable in enumerate(stable_mask) if stable]
        
        if len(stable_features) > 0:
            X_selected = X[:, stable_mask]
        else:
            print("Warning: No stable features found, selecting top 10 by frequency")
            top_indices = np.argsort(selection_frequencies)[-10:]
            stable_mask = np.zeros(n_features, dtype=bool)
            stable_mask[top_indices] = True
            stable_features = [feature_names[i] for i in top_indices]
            X_selected = X[:, stable_mask]
        
        print(f"  Selected {len(stable_features)} stable features")
        
        # Store detailed results
        frequency_dict = dict(zip(feature_names, selection_frequencies))
        
        self.selection_results['stability_selection'] = {
            'n_features': len(stable_features),
            'selected_features': stable_features,
            'selection_frequencies': frequency_dict,
            'support_mask': stable_mask,
            'threshold': threshold,
            'n_bootstrap': n_bootstrap
        }
        
        return X_selected, stable_features
    
    def correlation_based_selection(self, X: np.ndarray, 
                                  feature_names: List[str],
                                  correlation_threshold: float = 0.95) -> Tuple[np.ndarray, List[str]]:
        """
        Remove highly correlated features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : List[str]
            Names of features
        correlation_threshold : float
            Correlation threshold for removal
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]] : Selected features and their names
        """
        print(f"Applying correlation-based selection (threshold={correlation_threshold})...")
        
        # Calculate correlation matrix
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr().abs()
        
        # Find features to remove
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= correlation_threshold:
                    # Remove the feature with lower variance
                    feat_i = corr_matrix.columns[i]
                    feat_j = corr_matrix.columns[j]
                    var_i = df[feat_i].var()
                    var_j = df[feat_j].var()
                    
                    if var_i < var_j:
                        to_remove.add(feat_i)
                    else:
                        to_remove.add(feat_j)
        
        # Keep features not in removal set
        selected_features = [f for f in feature_names if f not in to_remove]
        selected_indices = [i for i, f in enumerate(feature_names) if f not in to_remove]
        
        X_selected = X[:, selected_indices]
        
        print(f"  Removed {len(to_remove)} highly correlated features")
        print(f"  Selected {len(selected_features)} features")
        
        self.selection_results['correlation_based'] = {
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'removed_features': list(to_remove),
            'correlation_threshold': correlation_threshold
        }
        
        return X_selected, selected_features
    
    def ensemble_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str],
                                 methods: List[str] = None,
                                 voting_threshold: float = 0.5) -> Tuple[np.ndarray, List[str]]:
        """
        Ensemble feature selection combining multiple methods.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        feature_names : List[str]
            Names of features
        methods : List[str], optional
            Methods to use in ensemble
        voting_threshold : float
            Threshold for ensemble voting
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]] : Selected features and their names
        """
        if methods is None:
            methods = ['mutual_information', 'rfe_cv', 'stability_selection']
        
        print(f"Applying ensemble feature selection with methods: {methods}")
        
        # Apply each method and collect results
        method_results = {}
        
        for method in methods:
            if method == 'mutual_information':
                k = min(50, X.shape[1] // 2)  # Select half of features or 50, whichever is smaller
                _, selected = self.mutual_information_selection(X, y, feature_names, k=k)
            elif method == 'rfe_cv':
                _, selected = self.recursive_feature_elimination_cv(X, y, feature_names)
            elif method == 'stability_selection':
                _, selected = self.stability_selection(X, y, feature_names)
            elif method == 'correlation_based':
                _, selected = self.correlation_based_selection(X, feature_names)
            else:
                print(f"Warning: Unknown method {method}")
                continue
                
            method_results[method] = set(selected)
        
        # Ensemble voting
        feature_votes = {}
        for feature in feature_names:
            votes = sum(1 for method_features in method_results.values() if feature in method_features)
            feature_votes[feature] = votes / len(method_results)
        
        # Select features with sufficient votes
        ensemble_selected = [f for f, vote in feature_votes.items() if vote >= voting_threshold]
        
        if len(ensemble_selected) == 0:
            print("Warning: No features passed voting threshold, selecting top 20 by votes")
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            ensemble_selected = [f for f, _ in sorted_features[:20]]
        
        # Create final feature matrix
        selected_indices = [i for i, f in enumerate(feature_names) if f in ensemble_selected]
        X_selected = X[:, selected_indices]
        
        print(f"  Ensemble selected {len(ensemble_selected)} features")
        
        self.selection_results['ensemble'] = {
            'n_features': len(ensemble_selected),
            'selected_features': ensemble_selected,
            'feature_votes': feature_votes,
            'methods_used': methods,
            'voting_threshold': voting_threshold,
            'method_results': {k: list(v) for k, v in method_results.items()}
        }
        
        return X_selected, ensemble_selected
    
    def save_selection_results(self, filepath: str):
        """Save feature selection results to file."""
        joblib.dump(self.selection_results, filepath)
        print(f"Feature selection results saved to {filepath}")
    
    def load_selection_results(self, filepath: str):
        """Load feature selection results from file."""
        self.selection_results = joblib.load(filepath)
        print(f"Feature selection results loaded from {filepath}")
    
    def print_selection_summary(self):
        """Print summary of all feature selection results."""
        print("\n=== FEATURE SELECTION SUMMARY ===")
        
        for method, results in self.selection_results.items():
            print(f"\n{method.upper()}:")
            print(f"  Selected features: {results['n_features']}")
            
            if 'feature_scores' in results:
                top_features = sorted(results['feature_scores'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top 5 by score: {[f[0] for f in top_features]}")
            elif 'selection_frequencies' in results:
                top_features = sorted(results['selection_frequencies'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top 5 by stability: {[f[0] for f in top_features]}")
            else:
                print(f"  Selected: {results['selected_features'][:5]}...")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna for texture analysis models.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the hyperparameter optimizer."""
        self.random_state = random_state
        self.study = None
        self.best_params = None
        
    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray, 
                             cv_folds: int = 5, n_trials: int = 100) -> Dict:
        """
        Optimize Random Forest hyperparameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        cv_folds : int
            Number of CV folds
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        Dict : Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Using default parameters.")
            return {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
        
        print(f"Optimizing Random Forest hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
            model = RandomForestClassifier(**params)
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            return cv_scores.mean()
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def optimize_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str],
                                 cv_folds: int = 5, n_trials: int = 50) -> Dict:
        """
        Optimize feature selection parameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        feature_names : List[str]
            Names of features
        cv_folds : int
            Number of CV folds
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        Dict : Best feature selection parameters
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Using default parameters.")
            return {'method': 'mutual_information', 'k': 20}
        
        print(f"Optimizing feature selection parameters ({n_trials} trials)...")
        
        def objective(trial):
            method = trial.suggest_categorical('method', 
                                             ['mutual_information', 'rfe_cv', 'stability_selection'])
            
            selector = AdvancedFeatureSelector(random_state=self.random_state)
            
            if method == 'mutual_information':
                k = trial.suggest_int('k', 10, min(100, X.shape[1]))
                X_selected, _ = selector.mutual_information_selection(X, y, feature_names, k=k)
            elif method == 'rfe_cv':
                min_features = trial.suggest_int('min_features', 5, 50)
                X_selected, _ = selector.recursive_feature_elimination_cv(
                    X, y, feature_names, min_features=min_features
                )
            elif method == 'stability_selection':
                threshold = trial.suggest_float('threshold', 0.3, 0.8)
                X_selected, _ = selector.stability_selection(
                    X, y, feature_names, threshold=threshold
                )
            
            # Evaluate with Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_selected, y, cv=cv_folds, scoring='accuracy')
            return cv_scores.mean()
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        print(f"Best feature selection parameters: {self.best_params}")
        print(f"Best CV score: {self.study.best_value:.4f}")
        
        return self.best_params 