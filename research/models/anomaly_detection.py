import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class AnomalyDetectionSuite:
    """
    Comprehensive suite of anomaly detection models specifically designed for 
    micro-damage detection, tear identification, and fabric irregularity detection.
    
    This suite uses unsupervised learning to identify fabric samples that deviate
    from normal patterns, which may indicate:
    - Micro-tears and small holes
    - Fabric deterioration and wear
    - Manufacturing defects
    - Surface irregularities and damage
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the anomaly detection suite for fabric damage detection
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def _initialize_models(self, contamination=0.1):
        """
        Initialize all anomaly detection models optimized for fabric damage detection
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of damaged/defective fabric samples in the data.
            Typical values: 0.05-0.15 (5-15% damage rate)
        """
        # One-Class SVM
        self.models['one_class_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=contamination  # nu parameter controls the fraction of support vectors
        )
        
        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Elliptic Envelope (Robust Covariance)
        self.models['elliptic_envelope'] = EllipticEnvelope(
            contamination=contamination,
            random_state=self.random_state
        )
        
        # Local Outlier Factor
        self.models['local_outlier_factor'] = LocalOutlierFactor(
            contamination=contamination,
            novelty=True  # Enable prediction on new data
        )
        
        # DBSCAN (density-based clustering for anomaly detection)
        self.models['dbscan'] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
    
    def detect_anomalies(self, X, contamination=0.1, feature_names=None):
        """
        Run all anomaly detection models on the data
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (should be scaled)
        contamination : float
            Expected proportion of anomalies
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        dict : Results from all anomaly detection models
        """
        self._initialize_models(contamination)
        print(f"Running {len(self.models)} anomaly detection methods on {X.shape[0]} samples (contamination: {contamination:.1%})...")
        
        self.results = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"Running {model_name}...")
                self.results[model_name] = self._run_single_anomaly_detector(
                    model, model_name, X, contamination
                )
            except Exception as e:
                print(f"  ✗ Error with {model_name}: {str(e)}")
                continue
        
        # Run autoencoder if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            try:
                print("Running autoencoder...")
                self.results['autoencoder'] = self._run_autoencoder_anomaly_detection(
                    X, contamination
                )
            except Exception as e:
                print(f"  ✗ Error with autoencoder: {str(e)}")
        
        self._summarize_anomaly_results()
        return self.results
    
    def _run_single_anomaly_detector(self, model, model_name, X, contamination):
        """
        Run a single anomaly detection model
        
        Parameters:
        -----------
        model : sklearn model
            Anomaly detection model
        model_name : str
            Name of the model
        X : np.ndarray
            Feature matrix
        contamination : float
            Expected contamination rate
            
        Returns:
        --------
        dict : Model results
        """
        if model_name == 'dbscan':
            # DBSCAN works differently - fit and predict simultaneously
            cluster_labels = model.fit_predict(X)
            # Points labeled as -1 are anomalies
            anomaly_predictions = (cluster_labels == -1).astype(int)
            anomaly_scores = np.ones(len(X))  # DBSCAN doesn't provide scores
            anomaly_scores[cluster_labels == -1] = -1
        else:
            # Standard anomaly detection workflow
            model.fit(X)
            
            if hasattr(model, 'predict'):
                anomaly_predictions = model.predict(X)
                # Convert to binary (1 for anomaly, 0 for normal)
                anomaly_predictions = (anomaly_predictions == -1).astype(int)
            else:
                anomaly_predictions = None
            
            # Get anomaly scores if available
            if hasattr(model, 'decision_function'):
                anomaly_scores = model.decision_function(X)
            elif hasattr(model, 'score_samples'):
                anomaly_scores = model.score_samples(X)
            else:
                anomaly_scores = None
        
        # Calculate statistics
        n_anomalies = np.sum(anomaly_predictions) if anomaly_predictions is not None else 0
        anomaly_percentage = (n_anomalies / len(X)) * 100
        
        # Store results
        results = {
            'model': model,
            'model_name': model_name,
            'anomaly_predictions': anomaly_predictions,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_indices': np.where(anomaly_predictions)[0] if anomaly_predictions is not None else None
        }
        
        print(f"  ✓ {model_name} - Detected {n_anomalies} anomalies ({anomaly_percentage:.2f}%)")
        
        return results
    
    def _run_autoencoder_anomaly_detection(self, X, contamination):
        """
        Run autoencoder-based anomaly detection
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        contamination : float
            Expected contamination rate
            
        Returns:
        --------
        dict : Autoencoder results
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for autoencoder")
        
        # Build autoencoder
        input_dim = X.shape[1]
        encoding_dim = max(2, input_dim // 4)  # Compress to 1/4 of input size
        
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        history = autoencoder.fit(
            X, X,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            shuffle=True
        )
        
        # Calculate reconstruction errors
        X_pred = autoencoder.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_pred), axis=1)
        
        # Determine threshold for anomalies
        threshold = np.percentile(mse, (1 - contamination) * 100)
        anomaly_predictions = (mse > threshold).astype(int)
        
        # Statistics
        n_anomalies = np.sum(anomaly_predictions)
        anomaly_percentage = (n_anomalies / len(X)) * 100
        
        results = {
            'model': autoencoder,
            'model_name': 'autoencoder',
            'anomaly_predictions': anomaly_predictions,
            'anomaly_scores': -mse,  # Negative MSE (higher = more normal)
            'reconstruction_errors': mse,
            'threshold': threshold,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_indices': np.where(anomaly_predictions)[0],
            'training_history': history.history
        }
        
        print(f"  ✓ autoencoder - Detected {n_anomalies} anomalies ({anomaly_percentage:.2f}%)")
        
        return results
    
    def _summarize_anomaly_results(self):
        """Summarize and compare anomaly detection results"""
        if not self.results:
            return
        
        print("\nAnomaly Detection Summary:")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Anomalies Detected': results['n_anomalies'],
                'Percentage': f"{results['anomaly_percentage']:.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    
    def evaluate_with_labels(self, true_labels):
        """
        Evaluate anomaly detection performance when ground truth is available
        
        Parameters:
        -----------
        true_labels : np.ndarray
            Ground truth labels (1 for anomaly, 0 for normal)
            
        Returns:
        --------
        dict : Evaluation metrics for each model
        """
        if not self.results:
            raise ValueError("No anomaly detection results available")
        
        print("\nAnomaly Detection Evaluation:")
        
        evaluation_results = {}
        
        for model_name, results in self.results.items():
            if results['anomaly_predictions'] is not None:
                pred_labels = results['anomaly_predictions']
                
                # Calculate metrics
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_labels, zero_division=0)
                accuracy = accuracy_score(true_labels, pred_labels)
                
                evaluation_results[model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'confusion_matrix': confusion_matrix(true_labels, pred_labels)
                }
                
                print(f"{model_name.replace('_', ' ').title()}:")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Accuracy: {accuracy:.4f}")
                print()
        
        return evaluation_results
    
    def get_anomaly_feature_analysis(self, X, feature_names=None, model_name='isolation_forest'):
        """
        Analyze which features contribute most to anomaly detection
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : list, optional
            Names of features
        model_name : str
            Model to use for analysis
            
        Returns:
        --------
        dict : Feature analysis results
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not available")
        
        results = self.results[model_name]
        if results['anomaly_predictions'] is None:
            raise ValueError(f"Model {model_name} doesn't provide predictions")
        
        anomaly_mask = results['anomaly_predictions'].astype(bool)
        normal_mask = ~anomaly_mask
        
        if np.sum(anomaly_mask) == 0 or np.sum(normal_mask) == 0:
            print("No anomalies or no normal samples found for analysis")
            return None
        
        # Calculate feature statistics for normal vs anomalous samples
        normal_stats = {
            'mean': np.mean(X[normal_mask], axis=0),
            'std': np.std(X[normal_mask], axis=0),
            'median': np.median(X[normal_mask], axis=0)
        }
        
        anomaly_stats = {
            'mean': np.mean(X[anomaly_mask], axis=0),
            'std': np.std(X[anomaly_mask], axis=0),
            'median': np.median(X[anomaly_mask], axis=0)
        }
        
        # Calculate differences
        mean_diff = np.abs(anomaly_stats['mean'] - normal_stats['mean'])
        std_ratio = anomaly_stats['std'] / (normal_stats['std'] + 1e-8)
        
        # Create feature analysis
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        feature_analysis = []
        for i, feature_name in enumerate(feature_names):
            feature_analysis.append({
                'feature': feature_name,
                'normal_mean': normal_stats['mean'][i],
                'anomaly_mean': anomaly_stats['mean'][i],
                'mean_difference': mean_diff[i],
                'normal_std': normal_stats['std'][i],
                'anomaly_std': anomaly_stats['std'][i],
                'std_ratio': std_ratio[i]
            })
        
        # Sort by mean difference (most discriminative features)
        feature_analysis.sort(key=lambda x: x['mean_difference'], reverse=True)
        
        return {
            'feature_analysis': feature_analysis,
            'top_discriminative_features': feature_analysis[:10],  # Top 10
            'normal_samples': np.sum(normal_mask),
            'anomaly_samples': np.sum(anomaly_mask)
        }
    
    def detect_anomalies_in_new_data(self, X_new, model_name='isolation_forest'):
        """
        Detect anomalies in new data using a previously trained model
        
        Parameters:
        -----------
        X_new : np.ndarray
            New data to analyze
        model_name : str
            Name of the trained model to use
            
        Returns:
        --------
        dict : Anomaly detection results for new data
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.results[model_name]['model']
        
        if model_name == 'autoencoder':
            # For autoencoder, calculate reconstruction error
            X_pred = model.predict(X_new, verbose=0)
            mse = np.mean(np.square(X_new - X_pred), axis=1)
            threshold = self.results[model_name]['threshold']
            anomaly_predictions = (mse > threshold).astype(int)
            anomaly_scores = -mse
        else:
            # For sklearn models
            if hasattr(model, 'predict'):
                anomaly_predictions = model.predict(X_new)
                anomaly_predictions = (anomaly_predictions == -1).astype(int)
            else:
                anomaly_predictions = None
            
            if hasattr(model, 'decision_function'):
                anomaly_scores = model.decision_function(X_new)
            elif hasattr(model, 'score_samples'):
                anomaly_scores = model.score_samples(X_new)
            else:
                anomaly_scores = None
        
        # Calculate statistics
        n_anomalies = np.sum(anomaly_predictions) if anomaly_predictions is not None else 0
        anomaly_percentage = (n_anomalies / len(X_new)) * 100
        
        return {
            'anomaly_predictions': anomaly_predictions,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_indices': np.where(anomaly_predictions)[0] if anomaly_predictions is not None else None
        } 