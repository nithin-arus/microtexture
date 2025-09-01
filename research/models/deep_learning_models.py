import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.applications import (
        ResNet50, EfficientNetB0, DenseNet121, VGG16,
        MobileNetV2, InceptionV3
    )
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Input
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Deep learning features will be disabled.")
    TENSORFLOW_AVAILABLE = False

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib


class DeepTextureExtractor:
    """
    Extract features using pre-trained CNN models for texture analysis.
    
    Provides feature extraction from multiple state-of-the-art CNN architectures
    and combines them with handcrafted texture features for improved performance.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224), include_top: bool = False):
        """
        Initialize the deep texture extractor.
        
        Parameters:
        -----------
        input_size : Tuple[int, int]
            Input image size for CNN models
        include_top : bool
            Whether to include the final classification layer
        """
        self.input_size = input_size
        self.include_top = include_top
        self.models = {}
        self.feature_extractors = {}
        
        if TENSORFLOW_AVAILABLE:
            self._initialize_models()
        else:
            print("TensorFlow not available. Deep learning features disabled.")
    
    def _initialize_models(self):
        """Initialize pre-trained CNN models."""
        print("Initializing pre-trained CNN models...")
        
        try:
            # ResNet50
            self.models['resnet50'] = ResNet50(
                weights='imagenet',
                include_top=self.include_top,
                input_shape=(*self.input_size, 3)
            )
            
            # EfficientNetB0
            self.models['efficientnet_b0'] = EfficientNetB0(
                weights='imagenet',
                include_top=self.include_top,
                input_shape=(*self.input_size, 3)
            )
            
            # DenseNet121
            self.models['densenet121'] = DenseNet121(
                weights='imagenet',
                include_top=self.include_top,
                input_shape=(*self.input_size, 3)
            )
            
            # VGG16
            self.models['vgg16'] = VGG16(
                weights='imagenet',
                include_top=self.include_top,
                input_shape=(*self.input_size, 3)
            )
            
            # MobileNetV2 (lightweight option)
            self.models['mobilenet_v2'] = MobileNetV2(
                weights='imagenet',
                include_top=self.include_top,
                input_shape=(*self.input_size, 3)
            )
            
            # Create feature extractors (remove final layers)
            for name, model in self.models.items():
                if not self.include_top:
                    # Add global average pooling for feature extraction
                    x = GlobalAveragePooling2D()(model.output)
                    self.feature_extractors[name] = Model(
                        inputs=model.input, 
                        outputs=x
                    )
                else:
                    self.feature_extractors[name] = model
                    
            print(f"Initialized {len(self.models)} CNN models for feature extraction")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.models = {}
            self.feature_extractors = {}
    
    def preprocess_image(self, image: np.ndarray, model_name: str) -> np.ndarray:
        """
        Preprocess image for specific CNN model.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        model_name : str
            Name of the CNN model
            
        Returns:
        --------
        np.ndarray : Preprocessed image
        """
        # Resize image
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
            
        resized = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 255] if needed
        if resized.max() <= 1.0:
            resized = (resized * 255).astype(np.uint8)
            
        # Add batch dimension
        preprocessed = np.expand_dims(resized, axis=0)
        
        # Apply model-specific preprocessing
        if model_name in ['resnet50', 'densenet121', 'vgg16']:
            preprocessed = preprocess_input(preprocessed)
        elif model_name == 'efficientnet_b0':
            preprocessed = tf.keras.applications.efficientnet.preprocess_input(preprocessed)
        elif model_name == 'mobilenet_v2':
            preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(preprocessed)
        
        return preprocessed
    
    def extract_features_single_model(self, image: np.ndarray, model_name: str) -> np.ndarray:
        """
        Extract features using a single CNN model.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        model_name : str
            Name of the CNN model
            
        Returns:
        --------
        np.ndarray : Extracted features
        """
        if not TENSORFLOW_AVAILABLE or model_name not in self.feature_extractors:
            return np.array([])
        
        try:
            # Preprocess image
            preprocessed = self.preprocess_image(image, model_name)
            
            # Extract features
            features = self.feature_extractors[model_name].predict(
                preprocessed, verbose=0
            )
            
            return features.flatten()
            
        except Exception as e:
            print(f"Error extracting features with {model_name}: {e}")
            return np.array([])
    
    def extract_deep_features(self, image: np.ndarray, 
                            models_to_use: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple CNN models.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        models_to_use : List[str], optional
            List of model names to use. If None, uses all available models.
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary of features from each model
        """
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        if models_to_use is None:
            models_to_use = list(self.feature_extractors.keys())
        
        deep_features = {}
        
        for model_name in models_to_use:
            if model_name in self.feature_extractors:
                features = self.extract_features_single_model(image, model_name)
                if len(features) > 0:
                    deep_features[model_name] = features
                    
        return deep_features
    
    def extract_features_batch(self, images: List[np.ndarray],
                             models_to_use: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract features from a batch of images.
        
        Parameters:
        -----------
        images : List[np.ndarray]
            List of images
        models_to_use : List[str], optional
            List of model names to use
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary of feature matrices
        """
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        if models_to_use is None:
            models_to_use = list(self.feature_extractors.keys())
        
        batch_features = {model: [] for model in models_to_use}
        
        for image in images:
            image_features = self.extract_deep_features(image, models_to_use)
            for model_name, features in image_features.items():
                batch_features[model_name].append(features)
        
        # Convert to numpy arrays
        for model_name in batch_features:
            if batch_features[model_name]:
                batch_features[model_name] = np.array(batch_features[model_name])
            else:
                batch_features[model_name] = np.array([])
                
        return batch_features


class HybridTextureClassifier:
    """
    Combine handcrafted texture features with deep learning features.
    
    Provides multiple fusion strategies for combining traditional texture analysis
    with modern deep learning approaches.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the hybrid classifier.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.fusion_models = {}
        self.feature_importance = {}
        
    def create_feature_fusion(self, X_handcrafted: np.ndarray, 
                            X_deep_dict: Dict[str, np.ndarray],
                            fusion_strategy: str = 'concatenate') -> np.ndarray:
        """
        Fuse handcrafted and deep learning features.
        
        Parameters:
        -----------
        X_handcrafted : np.ndarray
            Handcrafted texture features
        X_deep_dict : Dict[str, np.ndarray]
            Deep learning features from multiple models
        fusion_strategy : str
            Strategy for fusion ('concatenate', 'weighted', 'attention')
            
        Returns:
        --------
        np.ndarray : Fused feature matrix
        """
        if fusion_strategy == 'concatenate':
            # Simple concatenation
            all_features = [X_handcrafted]
            for model_name, features in X_deep_dict.items():
                if len(features) > 0:
                    all_features.append(features)
            
            if len(all_features) > 1:
                fused_features = np.concatenate(all_features, axis=1)
            else:
                fused_features = X_handcrafted
                
        elif fusion_strategy == 'weighted':
            # Weighted combination based on feature importance
            fused_features = self._weighted_fusion(X_handcrafted, X_deep_dict)
            
        elif fusion_strategy == 'attention':
            # Attention-based fusion
            fused_features = self._attention_fusion(X_handcrafted, X_deep_dict)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
        return fused_features
    
    def _weighted_fusion(self, X_handcrafted: np.ndarray, 
                        X_deep_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted fusion of features."""
        # For simplicity, use equal weights initially
        # In practice, weights could be learned or based on individual model performance
        
        all_features = [X_handcrafted]
        weights = [1.0]  # Weight for handcrafted features
        
        for model_name, features in X_deep_dict.items():
            if len(features) > 0:
                all_features.append(features)
                weights.append(1.0)  # Equal weight for now
        
        # Normalize features and apply weights
        normalized_features = []
        for i, features in enumerate(all_features):
            # Z-score normalization
            normalized = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
            weighted = normalized * weights[i]
            normalized_features.append(weighted)
        
        return np.concatenate(normalized_features, axis=1)
    
    def _attention_fusion(self, X_handcrafted: np.ndarray, 
                         X_deep_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Attention-based fusion (simplified version)."""
        # This is a simplified attention mechanism
        # In practice, you might want to use more sophisticated attention
        
        if not TENSORFLOW_AVAILABLE:
            return self.create_feature_fusion(X_handcrafted, X_deep_dict, 'concatenate')
        
        all_features = [X_handcrafted]
        for model_name, features in X_deep_dict.items():
            if len(features) > 0:
                all_features.append(features)
        
        if len(all_features) == 1:
            return X_handcrafted
        
        # Simple attention: compute attention weights based on feature variance
        attention_weights = []
        for features in all_features:
            variance = np.var(features, axis=0)
            weight = np.mean(variance)
            attention_weights.append(weight)
        
        # Normalize attention weights
        attention_weights = np.array(attention_weights)
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Apply attention weights
        attended_features = []
        for i, features in enumerate(all_features):
            attended = features * attention_weights[i]
            attended_features.append(attended)
        
        return np.concatenate(attended_features, axis=1)
    
    def create_ensemble_classifier(self, base_classifiers: List, 
                                 ensemble_type: str = 'voting') -> Union[VotingClassifier, StackingClassifier]:
        """
        Create ensemble classifier.
        
        Parameters:
        -----------
        base_classifiers : List
            List of (name, classifier) tuples
        ensemble_type : str
            Type of ensemble ('voting' or 'stacking')
            
        Returns:
        --------
        Ensemble classifier
        """
        if ensemble_type == 'voting':
            ensemble = VotingClassifier(
                estimators=base_classifiers,
                voting='soft'
            )
        elif ensemble_type == 'stacking':
            ensemble = StackingClassifier(
                estimators=base_classifiers,
                final_estimator=LogisticRegression(random_state=self.random_state),
                cv=5
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        return ensemble
    
    def compare_feature_types(self, X_handcrafted: np.ndarray, 
                            X_deep_dict: Dict[str, np.ndarray],
                            y: np.ndarray,
                            test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Compare performance of different feature types.
        
        Parameters:
        -----------
        X_handcrafted : np.ndarray
            Handcrafted features
        X_deep_dict : Dict[str, np.ndarray]
            Deep learning features
        y : np.ndarray
            Labels
        test_size : float
            Test set proportion
            
        Returns:
        --------
        Dict[str, Dict] : Performance comparison results
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        results = {}
        
        # Test handcrafted features
        if len(X_handcrafted) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_handcrafted, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = RandomForestClassifier(random_state=self.random_state)
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['handcrafted'] = {
                'accuracy': accuracy,
                'feature_count': X_handcrafted.shape[1],
                'classifier': 'RandomForest'
            }
        
        # Test individual deep learning models
        for model_name, X_deep in X_deep_dict.items():
            if len(X_deep) > 0 and X_deep.shape[0] == len(y):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_deep, y, test_size=test_size, random_state=self.random_state, stratify=y
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                clf = RandomForestClassifier(random_state=self.random_state)
                clf.fit(X_train_scaled, y_train)
                
                y_pred = clf.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[f'deep_{model_name}'] = {
                    'accuracy': accuracy,
                    'feature_count': X_deep.shape[1],
                    'classifier': 'RandomForest'
                }
        
        # Test fusion approaches
        for fusion_strategy in ['concatenate', 'weighted']:
            try:
                X_fused = self.create_feature_fusion(X_handcrafted, X_deep_dict, fusion_strategy)
                
                if len(X_fused) > 0 and X_fused.shape[0] == len(y):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_fused, y, test_size=test_size, random_state=self.random_state, stratify=y
                    )
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    clf = RandomForestClassifier(random_state=self.random_state)
                    clf.fit(X_train_scaled, y_train)
                    
                    y_pred = clf.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    results[f'fusion_{fusion_strategy}'] = {
                        'accuracy': accuracy,
                        'feature_count': X_fused.shape[1],
                        'classifier': 'RandomForest'
                    }
                    
            except Exception as e:
                print(f"Error testing fusion strategy {fusion_strategy}: {e}")
        
        return results
    
    def save_model(self, model, filepath: str):
        """Save trained model to disk."""
        joblib.dump(model, filepath)
        
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        return joblib.load(filepath) 