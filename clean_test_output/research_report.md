# Micro-Texture Fabric Analysis Research Report

Generated on: 2025-07-23 15:26:21

## Data Summary
- Total samples: 50
- Features extracted: 44
- Target classes: 3

## Supervised Learning Results

### Logistic Regression
- Test Accuracy: 0.3000
- CV Mean Accuracy: 0.3250 ± 0.1696

### Svm Linear
- Test Accuracy: 0.4000
- CV Mean Accuracy: 0.3000 ± 0.1696

### Svm Rbf
- Test Accuracy: 0.2000
- CV Mean Accuracy: 0.2500 ± 0.1369

### Random Forest
- Test Accuracy: 0.2000
- CV Mean Accuracy: 0.2500 ± 0.1369

### Gradient Boosting
- Test Accuracy: 0.2000
- CV Mean Accuracy: 0.2750 ± 0.1458

### Mlp Classifier
- Test Accuracy: 0.2000
- CV Mean Accuracy: 0.3500 ± 0.0935

## Anomaly Detection Results

### One Class Svm
- Anomalies detected: 22
- Anomaly percentage: 44.00%

### Isolation Forest
- Anomalies detected: 5
- Anomaly percentage: 10.00%

### Elliptic Envelope
- Anomalies detected: 5
- Anomaly percentage: 10.00%

### Local Outlier Factor
- Anomalies detected: 5
- Anomaly percentage: 10.00%

### Dbscan
- Anomalies detected: 50
- Anomaly percentage: 100.00%

## Statistical Analysis

- Features with significant class differences (p < 0.05): 0

## Files Generated

- Model comparison plots in `visualizations/`
- Feature analysis plots in `visualizations/`
- Dimensionality reduction plots in `visualizations/`
- Evaluation metrics in `evaluation/`
- Statistical analysis results in `statistical_analysis/`
