import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data processing utilities for micro-texture fabric analysis.
    
    Provides tools for:
    - Feature preprocessing and scaling for fabric data
    - Fabric type label encoding and handling
    - Statistical analysis of textile properties
    - Clustering analysis for fabric grouping
    - Damage detection data preparation
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_statistics = {}
        
    def prepare_features(self, features_df, target_column=None, create_synthetic_labels=True):
        """
        Prepare feature matrix and target vector from CSV data
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing extracted features
        target_column : str, optional
            Column name to use as target variable
        create_synthetic_labels : bool
            If True and no target_column, create synthetic labels for demonstration
            
        Returns:
        --------
        tuple : (X, y, feature_names)
        """
        print("Preparing features for analysis...")
        
        # Identify feature columns (exclude metadata columns)
        metadata_columns = [
            'image_path', 'fractal_overlay_path', 'timestamp', 
            'processing_time', 'file_size', 'filename', 'path',
            'fractal_equation'  # Exclude text-based fractal equation
        ]
        
        # Get feature columns
        feature_columns = [col for col in features_df.columns 
                          if col not in metadata_columns and col != target_column]
        
        # Extract feature matrix and ensure numeric types
        X_df = features_df[feature_columns].copy()
        
        # Convert to numeric, handling different data types
        for col in feature_columns:
            if X_df[col].dtype == 'bool':
                # Convert boolean to int (True=1, False=0)
                X_df[col] = X_df[col].astype(int)
            else:
                # Convert to numeric, forcing non-numeric values to NaN
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        X = X_df.values.astype(float)  # Ensure all values are float
        feature_names = feature_columns
        
        # Handle missing values
        if np.any(np.isnan(X)):
            print("  Warning: Found missing values, filling with median...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Handle target variable
        y = None
        if target_column and target_column in features_df.columns:
            print(f"  Using '{target_column}' as target variable")
            y = features_df[target_column].values
            
            # Encode string labels to integers
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)
                
        elif create_synthetic_labels:
            print("  Creating synthetic labels for demonstration...")
            y = self._create_synthetic_labels(X)
        
        # Compute feature statistics
        self._compute_feature_statistics(X, feature_names)
        
        print(f"  Prepared {X.shape[0]} samples with {X.shape[1]} features")
        if y is not None:
            unique_labels, counts = np.unique(y, return_counts=True)
            print(f"  Target distribution: {dict(zip(unique_labels, counts))}")
        
        return X, y, feature_names
    
    def _create_synthetic_labels(self, X):
        """
        Create synthetic fabric type labels for demonstration purposes when no target column is provided.
        Uses clustering to group similar fabric samples.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray : Synthetic fabric type labels (0, 1, 2 representing different fabric clusters)
        """
        print("  Creating synthetic fabric type labels using clustering...")
        
        # Use K-means clustering to create synthetic fabric type labels
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Scale features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters (fabric types)
        n_clusters = min(3, max(2, len(X) // 10))  # 2-3 fabric types for demo
        
        # Cluster to create fabric type groups
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        synthetic_labels = kmeans.fit_predict(X_scaled)
        
        print(f"  Created {n_clusters} synthetic fabric type groups")
        return synthetic_labels
    
    def _compute_feature_statistics(self, X, feature_names):
        """
        Compute comprehensive statistics for each feature
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : list
            Names of features
        """
        self.feature_statistics = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            self.feature_statistics[feature_name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'median': np.median(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75),
                'skewness': stats.skew(feature_data),
                'kurtosis': stats.kurtosis(feature_data),
                'variance': np.var(feature_data)
            }
    
    def compute_feature_correlations(self, X, feature_names):
        """
        Compute feature correlation matrix
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : list
            Names of features
            
        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        correlation_matrix = np.corrcoef(X.T)
        correlation_df = pd.DataFrame(
            correlation_matrix, 
            index=feature_names, 
            columns=feature_names
        )
        
        return correlation_df
    
    def identify_feature_groups(self, feature_names):
        """
        Automatically group features by type (texture, fractal, statistical, etc.)
        
        Parameters:
        -----------
        feature_names : list
            Names of features
            
        Returns:
        --------
        dict : Dictionary mapping group names to feature indices
        """
        feature_groups = {
            'texture': [],
            'fractal': [],
            'statistical': [],
            'geometric': [],
            'frequency': [],
            'edge': [],
            'other': []
        }
        
        for i, feature_name in enumerate(feature_names):
            feature_lower = feature_name.lower()
            
            if any(keyword in feature_lower for keyword in ['texture', 'glcm', 'lbp', 'haralick']):
                feature_groups['texture'].append(i)
            elif any(keyword in feature_lower for keyword in ['fractal', 'higuchi', 'katz', 'dfa', 'hurst']):
                feature_groups['fractal'].append(i)
            elif any(keyword in feature_lower for keyword in ['mean', 'std', 'var', 'skew', 'kurt', 'entropy']):
                feature_groups['statistical'].append(i)
            elif any(keyword in feature_lower for keyword in ['area', 'perimeter', 'compactness', 'extent']):
                feature_groups['geometric'].append(i)
            elif any(keyword in feature_lower for keyword in ['freq', 'fft', 'spectral', 'energy']):
                feature_groups['frequency'].append(i)
            elif any(keyword in feature_lower for keyword in ['edge', 'gradient', 'sobel', 'canny']):
                feature_groups['edge'].append(i)
            else:
                feature_groups['other'].append(i)
        
        # Remove empty groups
        feature_groups = {k: v for k, v in feature_groups.items() if v}
        
        return feature_groups
    
    def statistical_feature_comparison(self, X, y, feature_names, alpha=0.05):
        """
        Perform statistical tests to compare features across classes
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_names : list
            Names of features
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Statistical test results for each feature
        """
        print("Performing statistical feature comparison...")
        
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        statistical_results = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            # Prepare data for each class
            class_data = []
            for label in unique_labels:
                class_mask = y == label
                class_data.append(feature_data[class_mask])
            
            # Choose appropriate statistical test
            if n_classes == 2:
                # Two-class comparison: Mann-Whitney U test (non-parametric)
                try:
                    statistic, p_value = mannwhitneyu(
                        class_data[0], class_data[1], 
                        alternative='two-sided'
                    )
                    test_name = 'Mann-Whitney U'
                except:
                    # Fallback to t-test
                    statistic, p_value = stats.ttest_ind(class_data[0], class_data[1])
                    test_name = 'T-test'
            else:
                # Multi-class comparison: Kruskal-Wallis test (non-parametric)
                try:
                    statistic, p_value = kruskal(*class_data)
                    test_name = 'Kruskal-Wallis'
                except:
                    # Fallback to ANOVA
                    statistic, p_value = stats.f_oneway(*class_data)
                    test_name = 'ANOVA'
            
            # Store results
            statistical_results[feature_name] = {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': self._calculate_effect_size(class_data, test_name)
            }
        
        # Count significant features
        significant_count = sum(1 for result in statistical_results.values() 
                              if result['significant'])
        
        print(f"  Found {significant_count}/{len(feature_names)} features with significant differences (α={alpha})")
        
        return statistical_results
    
    def _calculate_effect_size(self, class_data, test_name):
        """
        Calculate effect size for statistical tests
        
        Parameters:
        -----------
        class_data : list
            List of arrays containing data for each class
        test_name : str
            Name of the statistical test used
            
        Returns:
        --------
        float : Effect size
        """
        if len(class_data) == 2:
            # Cohen's d for two groups
            mean1, mean2 = np.mean(class_data[0]), np.mean(class_data[1])
            std1, std2 = np.std(class_data[0], ddof=1), np.std(class_data[1], ddof=1)
            n1, n2 = len(class_data[0]), len(class_data[1])
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean1 - mean2) / pooled_std
            return abs(cohens_d)
        else:
            # Eta-squared for multiple groups (ANOVA effect size)
            all_data = np.concatenate(class_data)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in class_data)
            
            # Total sum of squares
            ss_total = np.sum((all_data - grand_mean)**2)
            
            if ss_total == 0:
                return 0.0
            
            eta_squared = ss_between / ss_total
            return eta_squared
    
    def perform_clustering(self, X, methods=['kmeans', 'dbscan', 'hierarchical']):
        """
        Perform unsupervised clustering using multiple methods
        
        Parameters:
        -----------
        X : np.ndarray
            Scaled feature matrix
        methods : list
            Clustering methods to use
            
        Returns:
        --------
        dict : Clustering results for each method
        """
        print("Performing clustering analysis...")
        
        clustering_results = {}
        
        # Determine optimal number of clusters using elbow method
        optimal_k = self._find_optimal_clusters(X)
        
        for method in methods:
            print(f"  Running {method} clustering...")
            
            try:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    cluster_labels = clusterer.fit_predict(X)
                    
                elif method == 'dbscan':
                    # Use heuristic for eps based on k-distance
                    eps = self._estimate_dbscan_eps(X)
                    clusterer = DBSCAN(eps=eps, min_samples=5)
                    cluster_labels = clusterer.fit_predict(X)
                    
                elif method == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=optimal_k)
                    cluster_labels = clusterer.fit_predict(X)
                
                # Calculate clustering metrics
                n_clusters = len(np.unique(cluster_labels))
                if n_clusters > 1 and -1 not in cluster_labels:  # Valid clustering
                    silhouette = silhouette_score(X, cluster_labels)
                else:
                    silhouette = -1
                
                clustering_results[method] = {
                    'clusterer': clusterer,
                    'labels': cluster_labels,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'cluster_sizes': np.bincount(cluster_labels[cluster_labels >= 0])
                }
                
                print(f"    ✓ {method}: {n_clusters} clusters, silhouette: {silhouette:.3f}")
                
            except Exception as e:
                print(f"    ✗ Error with {method}: {str(e)}")
                continue
        
        return clustering_results
    
    def _find_optimal_clusters(self, X, max_k=10):
        """
        Find optimal number of clusters using elbow method
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        max_k : int
            Maximum number of clusters to test
            
        Returns:
        --------
        int : Optimal number of clusters
        """
        max_k = min(max_k, len(X) // 2)  # Ensure reasonable max_k
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (maximum decrease in inertia)
        if len(inertias) > 1:
            decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            optimal_k = k_range[np.argmax(decreases)]
        else:
            optimal_k = 3  # Default
        
        return optimal_k
    
    def _estimate_dbscan_eps(self, X, k=5):
        """
        Estimate eps parameter for DBSCAN using k-distance graph
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        k : int
            Number of nearest neighbors
            
        Returns:
        --------
        float : Estimated eps value
        """
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        
        # Sort distances to k-th nearest neighbor
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Use 90th percentile as eps estimate
        eps = np.percentile(distances, 90)
        
        return eps
    
    def create_feature_summary_table(self, X, y, feature_names):
        """
        Create a comprehensive summary table of features
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray, optional
            Target labels
        feature_names : list
            Names of features
            
        Returns:
        --------
        pd.DataFrame : Feature summary table
        """
        summary_data = []
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            # Basic statistics
            summary_row = {
                'Feature': feature_name,
                'Mean': np.mean(feature_data),
                'Std': np.std(feature_data),
                'Min': np.min(feature_data),
                'Max': np.max(feature_data),
                'Median': np.median(feature_data),
                'Skewness': stats.skew(feature_data),
                'Kurtosis': stats.kurtosis(feature_data),
                'Missing_%': (np.sum(np.isnan(feature_data)) / len(feature_data)) * 100
            }
            
            # Add class-specific statistics if target is available
            if y is not None:
                unique_labels = np.unique(y)
                for label in unique_labels:
                    mask = y == label
                    class_data = feature_data[mask]
                    summary_row[f'Mean_Class_{label}'] = np.mean(class_data)
                    summary_row[f'Std_Class_{label}'] = np.std(class_data)
            
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def detect_outliers(self, X, feature_names, method='iqr', threshold=1.5):
        """
        Detect outliers in feature data
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : list
            Names of features
        method : str
            Method for outlier detection ('iqr', 'zscore')
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        dict : Outlier information for each feature
        """
        outlier_info = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            if method == 'iqr':
                Q1 = np.percentile(feature_data, 25)
                Q3 = np.percentile(feature_data, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(feature_data))
                outlier_mask = z_scores > threshold
            
            outlier_indices = np.where(outlier_mask)[0]
            
            outlier_info[feature_name] = {
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(feature_data)) * 100,
                'outlier_values': feature_data[outlier_indices] if len(outlier_indices) > 0 else []
            }
        
        return outlier_info
    
    def normalize_features(self, X, method='standard'):
        """
        Normalize feature matrix using different scaling methods
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        method : str
            Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
        --------
        np.ndarray : Normalized feature matrix
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        X_normalized = scaler.fit_transform(X)
        return X_normalized, scaler 