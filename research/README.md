# Research Analysis Framework

Comprehensive machine learning and statistical analysis framework for micro-texture fabric analysis, providing end-to-end research capabilities for **fabric classification** and **micro-damage detection** in textile materials.

## 🚀 Features

### Machine Learning Models
- **Supervised Classification**: Logistic Regression, SVM (Linear/RBF), Random Forest, Gradient Boosting, XGBoost, LightGBM, MLP
- **Anomaly Detection**: One-Class SVM, Isolation Forest, Elliptic Envelope, Local Outlier Factor, DBSCAN, Autoencoders
- **Unsupervised Learning**: K-Means, DBSCAN, Hierarchical Clustering

### Analysis Capabilities
- **Feature Analysis**: Correlation analysis, statistical comparisons, feature importance
- **Dimensionality Reduction**: PCA, t-SNE, UMAP visualizations
- **Model Evaluation**: Cross-validation, robustness testing, learning curves
- **Explainability**: SHAP values, LIME explanations (optional)
- **Statistical Testing**: Mann-Whitney U, Kruskal-Wallis, effect size analysis

### Visualization Suite
- Model performance comparisons
- Feature importance and correlation heatmaps
- Confusion matrices and ROC curves
- Dimensionality reduction plots
- Clustering visualizations
- Anomaly detection scatter plots

## 📦 Installation

The research framework requires additional dependencies beyond the base project:

```bash
# Required packages (add to requirements.txt)
pip install scikit-learn>=1.0.0
pip install umap-learn
pip install seaborn

# Optional but recommended for enhanced functionality
pip install xgboost              # For XGBoost classifier
pip install lightgbm             # For LightGBM classifier  
pip install tensorflow>=2.0      # For autoencoder anomaly detection
pip install shap                 # For SHAP explainability
pip install lime                 # For LIME explainability
```

## 🔧 Quick Start

### 1. Complete Analysis Pipeline

Run the entire research analysis with one command:

```python
from research import ResearchAnalyzer

# Load your fabric feature data
analyzer = ResearchAnalyzer("data/features.csv", "fabric_analysis_output")

# Run complete analysis (works without labeled data)
analyzer.run_complete_analysis()

# Results saved to fabric_analysis_output/
```

### 2. Fabric Classification (when labels available)

```python
# If you have fabric type labels in your CSV
analyzer = ResearchAnalyzer("data/features.csv", "fabric_classification")
analyzer.load_and_prepare_data(target_column="label")  # or "fabric_type"
analyzer.run_supervised_analysis()
analyzer.run_feature_analysis()
```

### 3. Micro-Damage Detection (unsupervised)

```python
# Detect anomalies that may indicate damage
analyzer = ResearchAnalyzer("data/features.csv", "damage_detection")
analyzer.load_and_prepare_data()
analyzer.run_anomaly_detection(contamination=0.05)  # Expect 5% damaged samples
```

### 4. Custom Analysis Pipeline

```python
analyzer = ResearchAnalyzer("data/features.csv", "custom_analysis")

# Step-by-step analysis
analyzer.load_and_prepare_data()
analyzer.run_feature_analysis()        # Explore feature relationships
analyzer.run_anomaly_detection()       # Find unusual patterns
analyzer.run_clustering_analysis()     # Group similar fabrics
analyzer.run_dimensionality_reduction()  # Visualize in 2D/3D
analyzer.generate_research_report()    # Create comprehensive report
```

## 📁 Output Structure

```
research_output/
├── research_report.md              # Comprehensive analysis report
├── models/                         # Trained model files
├── visualizations/                 # All plots and charts
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── correlation_analysis.png
│   ├── dimensionality_reduction.png
│   ├── clustering_results.png
│   ├── anomaly_detection.png
│   └── confusion_matrices.png
├── evaluation/                     # Detailed evaluation metrics
└── statistical_analysis/          # Statistical test results
```

## 🎯 Use Cases

### 1. Fabric Type Classification
```python
# Train models to classify different fabric types (cotton, polyester, wool, etc.)
analyzer = ResearchAnalyzer("data/features.csv", "fabric_classification")
analyzer.load_and_prepare_data(target_column="fabric_type")
analyzer.run_supervised_analysis()

# Results: Accuracy metrics, confusion matrices, feature importance
```

### 2. Fabric Quality Assessment
```python
# Binary classification for high vs. low quality fabrics
analyzer = ResearchAnalyzer("data/features.csv", "quality_assessment")
analyzer.load_and_prepare_data(target_column="quality_grade")
analyzer.run_supervised_analysis()
```

### 3. Micro-Damage Detection
```python
# Unsupervised anomaly detection for damage identification
analyzer = ResearchAnalyzer("data/features.csv", "damage_detection")
analyzer.load_and_prepare_data()
analyzer.run_anomaly_detection(contamination=0.08)  # 8% expected damage rate

# Results: Anomaly scores, damaged sample identification
```

### 4. Fabric Wear Analysis
```python
# Detect progressive wear and deterioration
analyzer = ResearchAnalyzer("data/features.csv", "wear_analysis")
analyzer.load_and_prepare_data(target_column="wear_level")  # if available
analyzer.run_supervised_analysis()
analyzer.run_anomaly_detection()  # Identify unusual wear patterns
```

### 5. Manufacturing Defect Detection
```python
# Find samples with manufacturing irregularities
analyzer = ResearchAnalyzer("data/features.csv", "defect_detection")
analyzer.load_and_prepare_data()
analyzer.run_anomaly_detection(contamination=0.03)  # Low defect rate expected
```

### 6. Feature Discovery
```python
# Explore which features are most important for fabric analysis
analyzer = ResearchAnalyzer("data/features.csv", "feature_analysis")
analyzer.load_and_prepare_data()
analyzer.run_feature_analysis()
analyzer.run_dimensionality_reduction()

# Results: Feature correlations, importance rankings, PCA plots
```

## 🔬 Research Workflow

1. **Data Collection**: Use the main data collection pipeline to capture images and extract features
2. **Feature Preparation**: Load CSV data and prepare for analysis (works without labels)
3. **Exploratory Analysis**: Run feature analysis and dimensionality reduction
4. **Fabric Classification**: Train supervised models if fabric type labels are available
5. **Damage Detection**: Use anomaly detection to identify micro-damage and irregularities
6. **Evaluation**: Assess model performance and robustness
7. **Interpretation**: Use explainability tools to understand model decisions
8. **Reporting**: Generate comprehensive research report

## ⚙️ Configuration Options

### Supervised Learning (Fabric Classification)
- `test_size`: Proportion of data for testing (default: 0.2)
- `cv_folds`: Number of cross-validation folds (default: 5)
- Model-specific hyperparameters

### Anomaly Detection (Damage Detection)
- `contamination`: Expected proportion of damaged samples (default: 0.1)
  - Micro-damage: 0.05-0.10 (5-10%)
  - Manufacturing defects: 0.02-0.05 (2-5%)
  - General quality issues: 0.10-0.15 (10-15%)
- Model-specific parameters (eps for DBSCAN, nu for One-Class SVM)

### Visualization
- Output formats: PNG, PDF
- Resolution: Configurable DPI
- Color schemes: Multiple palettes available

## 🤝 Integration with Main Pipeline

The research framework works seamlessly with the main data collection pipeline:

1. **Collect Data**: `python main.py` (captures images + extracts 40+ features)
2. **Analyze Data**: `python run_research_analysis.py` 
3. **Review Results**: Open `research_output/research_report.md`

## 📝 Example Research Questions

The framework can help answer questions like:

- **Fabric Classification**: Can we automatically classify fabric types based on micro-texture features?
- **Quality Assessment**: Which features best distinguish high-quality from low-quality fabrics?
- **Damage Detection**: Can we detect micro-damage before it becomes visible to the naked eye?
- **Wear Analysis**: How do fabrics change over time and use cycles?
- **Feature Importance**: Which fractal or texture features are most discriminative for fabric analysis?
- **Robustness**: How stable are our models to noise and variations in imaging conditions?
- **Clustering**: Do fabrics naturally group into categories based on their micro-texture?
- **Manufacturing QC**: Can we detect subtle manufacturing defects automatically?

## 🛠️ Extending the Framework

The modular design makes it easy to add new capabilities:

### Adding New Models
```python
# In SupervisedModelSuite._initialize_models()
self.models['custom_fabric_classifier'] = CustomClassifier(
    fabric_specific_param=value,
    random_state=self.random_state
)
```

### Adding New Damage Detection Methods
```python
# In AnomalyDetectionSuite._initialize_models()
self.models['custom_damage_detector'] = CustomDamageDetector(
    damage_sensitivity=0.05,
    feature_focus=['fractal', 'edge']  # Focus on damage-relevant features
)
```

### Adding Custom Visualizations
```python
# In ResearchVisualizer
def plot_fabric_damage_analysis(self, data, results):
    # Custom plotting for damage visualization
    plt.savefig(self.output_dir / 'damage_analysis.png')
```

## 📊 Performance Considerations

- **Memory Usage**: Large datasets may require sampling for visualization
- **Processing Time**: Anomaly detection with autoencoders may take longer
- **Model Selection**: Random Forest and XGBoost typically perform well for fabric classification
- **Feature Scaling**: Always applied automatically for anomaly detection methods

## 🎯 Best Practices

### For Fabric Classification:
- Use stratified sampling to maintain class balance
- Consider ensemble methods (Random Forest, XGBoost) for robust performance
- Use cross-validation to assess model stability
- Analyze feature importance to understand fabric characteristics

### For Damage Detection:
- Start with low contamination rates (0.02-0.05) for subtle damage
- Use multiple anomaly detection methods and compare results
- Focus on fractal and edge features which are sensitive to damage
- Validate anomaly predictions with visual inspection when possible

### For Feature Analysis:
- Remove highly correlated features (>0.95 correlation)
- Focus on feature groups (fractal, texture, statistical) for interpretation
- Use dimensionality reduction to visualize fabric relationships

## 🚀 Advanced Usage

### Batch Processing Multiple Datasets
```python
import glob

# Process multiple fabric datasets
for csv_file in glob.glob("data/fabric_*.csv"):
    output_dir = f"analysis_{csv_file.split('_')[1]}"
    analyzer = ResearchAnalyzer(csv_file, output_dir)
    analyzer.run_complete_analysis()
```

### Custom Feature Selection
```python
# Focus analysis on specific feature types
analyzer = ResearchAnalyzer("data/features.csv", "fractal_analysis")
analyzer.load_and_prepare_data()

# Filter to fractal features only
fractal_features = [f for f in analyzer.feature_names if 'fractal' in f.lower()]
fractal_indices = [analyzer.feature_names.index(f) for f in fractal_features]
X_fractal = analyzer.X[:, fractal_indices]

# Run analysis on fractal features only
analyzer.run_anomaly_detection_custom(X_fractal, fractal_features)
```

## 📞 Support

For questions about fabric analysis applications:
- Check the comprehensive research report generated by the framework
- Review feature importance plots to understand model decisions
- Use dimensionality reduction plots to visualize fabric relationships
- Examine anomaly detection results for damage identification

The framework is designed to be self-documenting through its extensive visualization and reporting capabilities. 