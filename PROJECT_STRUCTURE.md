# Project Structure & Organization

This project maintains **complete separation** between data collection and research analysis pipelines.

## 📁 **Data Collection Pipeline**

### **Core Files:**
```
main.py                          # Main data collection orchestrator
preprocess/
  └── preprocess.py             # Image preprocessing
capture/
  ├── capture_image.py          # Image capture functionality
  └── images/                   # 🔄 OUTPUT: Captured images
features/
  └── extract_features.py       # Feature extraction
analysis/
  ├── feature_extractor.py      # Core feature extraction
  ├── fractal_analysis.py       # Fractal dimension analysis
  └── fractal_fitting.py        # Fractal surface fitting
```

### **Data Collection Outputs:**
```
data/
  └── features.csv              # 🔄 OUTPUT: Extracted features
capture/images/                 # 🔄 OUTPUT: Raw captured images
fractal_overlays/               # 🔄 OUTPUT: Fractal visualization overlays
```

---

## 🔬 **Research Analysis Pipeline**

### **Core Files:**
```
research/
  ├── __init__.py               # Main research package
  ├── research_analysis.py      # Main research orchestrator
  ├── models/
  │   ├── supervised_models.py  # Classification models
  │   └── anomaly_detection.py  # Anomaly detection models
  ├── utils/
  │   └── data_utils.py         # Data processing utilities
  ├── visualization/
  │   └── visualizers.py        # Visualization tools
  ├── evaluation/
  │   └── evaluators.py         # Model evaluation tools
  └── README.md                 # Research framework documentation

run_research_analysis.py       # CLI interface for research analysis
```

### **Research Analysis Outputs:**
```
research_output/                # 🔄 OUTPUT: All research results
  ├── research_report.md        # Comprehensive analysis report
  ├── visualizations/           # All plots and charts
  │   ├── model_comparison.png
  │   ├── feature_importance.png
  │   ├── correlation_analysis.png
  │   ├── dimensionality_reduction.png
  │   ├── clustering_results.png
  │   ├── anomaly_detection.png
  │   └── ...
  ├── evaluation/               # Detailed evaluation metrics
  ├── models/                   # Trained model files
  └── statistical_analysis/     # Statistical test results
```

---

## 🔗 **Pipeline Independence**

### **✅ Complete Separation:**
- **No cross-dependencies** between pipelines
- **Independent execution** - can run either without the other
- **Separate output folders** - no conflicts
- **Modular design** - easy to extend either pipeline

### **🔄 Data Flow:**
```
1. Data Collection: main.py → data/features.csv
                             ↓
2. Research Analysis: run_research_analysis.py → research_output/
```

---

## 🚀 **Usage**

### **Data Collection Only:**
```bash
# Collect data and extract features
python main.py

# Outputs:
# - data/features.csv
# - capture/images/*.jpg  
# - fractal_overlays/*.png
```

### **Research Analysis Only:**
```bash
# Analyze existing feature data
python run_research_analysis.py

# Or programmatically:
from research import run_quick_analysis
analyzer = run_quick_analysis("data/features.csv")

# Outputs:
# - research_output/ (complete analysis results)
```

### **Complete Workflow:**
```bash
# Step 1: Collect data
python main.py

# Step 2: Analyze data  
python run_research_analysis.py

# Result: Both data/ and research_output/ directories populated
```

---

## 📦 **Dependencies**

### **Data Collection Dependencies:**
```
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-image>=0.20.0
scipy>=1.10.0
matplotlib>=3.7.0
picamera2>=0.3.12    # For Raspberry Pi camera
tqdm>=4.65.0
PyWavelets>=1.4.0
mahotas>=1.4.0
```

### **Research Analysis Dependencies:**
```
# Core (required)
scikit-learn>=1.0.0
seaborn>=0.12.0
umap-learn>=0.5.0
joblib>=1.2.0

# Optional (enhanced functionality)
xgboost>=1.7.0       # For XGBoost classifier
lightgbm>=3.3.0      # For LightGBM classifier
tensorflow>=2.10.0   # For autoencoder anomaly detection
shap>=0.41.0         # For SHAP explainability
lime>=0.2.0          # For LIME explainability
```

---

## 🔧 **Configuration**

Both pipelines are independently configurable:

- **Data Collection**: Modify parameters in `main.py` and individual modules
- **Research Analysis**: Use CLI flags or modify `ResearchAnalyzer` parameters

No shared configuration files = no conflicts between pipelines. 