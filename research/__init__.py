"""
Comprehensive Research Analysis Framework for Micro-Texture Fabric Analysis

This package provides end-to-end machine learning and statistical analysis tools
for fabric classification and micro-damage detection in textile materials.

Main Components:
- ResearchAnalyzer: Main orchestrator for all analysis tasks
- SupervisedModelSuite: Classification models for fabric type and quality assessment (LR, SVM, RF, XGBoost, MLP)
- AnomalyDetectionSuite: Anomaly detection models for micro-damage and tear detection (One-Class SVM, Isolation Forest, Autoencoders)
- DataProcessor: Data preparation and statistical analysis utilities
- ResearchVisualizer: Comprehensive visualization tools
- ModelEvaluator: Model evaluation and explainability tools

Usage:
------
```python
from research import ResearchAnalyzer, run_quick_analysis

# Quick analysis for fabric classification and damage detection
analyzer = run_quick_analysis("data/features.csv")

# Or manual control
analyzer = ResearchAnalyzer("data/features.csv", "output_dir")
analyzer.run_complete_analysis()
```
"""

from .research_analysis import ResearchAnalyzer, run_quick_analysis
from .models.supervised_models import SupervisedModelSuite
from .models.anomaly_detection import AnomalyDetectionSuite
from .utils.data_utils import DataProcessor
from .visualization.visualizers import ResearchVisualizer
from .evaluation.evaluators import ModelEvaluator

__version__ = "1.0.0"
__author__ = "Micro-Texture Research Team"

__all__ = [
    'ResearchAnalyzer',
    'run_quick_analysis',
    'SupervisedModelSuite',
    'AnomalyDetectionSuite', 
    'DataProcessor',
    'ResearchVisualizer',
    'ModelEvaluator'
] 