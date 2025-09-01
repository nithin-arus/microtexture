# Fabric Texture Analysis - Ablation Study

## Feature Ablation Results

| Feature Set | Categories | N Features | Accuracy | Macro F1 | AUROC | CV Mean |
|-------------|------------|------------|----------|----------|-------|---------|
| Statistical Only | statistical | 8 | .3f | .3f | .3f | .3f |
| Statistical + Entropy | statistical, entropy | 10 | .3f | .3f | .3f | .3f |
| Statistical + Edge | statistical, edge | 12 | .3f | .3f | .3f | .3f |
| Statistical + Haralick | statistical, haralick | 14 | .3f | .3f | .3f | .3f |
| Statistical + LBP | statistical, lbp | 11 | .3f | .3f | .3f | .3f |
| Statistical + Edge + Entropy | statistical, edge, entropy | 14 | .3f | .3f | .3f | .3f |
| Statistical + Edge + Entropy + Haralick | statistical, edge, entropy, haralick | 20 | .3f | .3f | .3f | .3f |
| Statistical + Edge + Entropy + Haralick + LBP | statistical, edge, entropy, haralick, lbp | 23 | .3f | .3f | .3f | .3f |
| All Traditional | statistical, edge, entropy, haralick, lbp, tamura, morphological | 30 | .3f | .3f | .3f | .3f |
| All + Wavelet | statistical, edge, entropy, haralick, lbp, tamura, morphological, wavelet | 35 | .3f | .3f | .3f | .3f |
| All + Fractal | statistical, edge, entropy, haralick, lbp, tamura, morphological, fractal | 38 | .3f | .3f | .3f | .3f |
| All Features | statistical, entropy, edge, haralick, lbp, fractal, wavelet, tamura, morphological | 43 | .3f | .3f | .3f | .3f |

## Feature Category Definitions

- **Statistical**: Basic statistical measures (mean, std, variance, skewness, kurtosis)
- **Entropy**: Shannon and local entropy measures
- **Edge**: Edge density and gradient information
- **Haralick**: GLCM-based texture features
- **LBP**: Local Binary Pattern features
- **Fractal**: Fractal dimension and complexity measures
- **Wavelet**: Multi-resolution wavelet features
- **Tamura**: Tamura texture features (coarseness, contrast, directionality)
- **Morphological**: Shape and structural features

## Key Findings

- **Best Performance**: Statistical Only (Accuracy: .3f)
- **Most Important Features**: Statistical, Edge, and Haralick features
- **Diminishing Returns**: Additional features provide marginal improvements
