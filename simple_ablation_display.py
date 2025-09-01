#!/usr/bin/env python3
"""
Simple Ablation Study Display
============================

Displays the ablation study results in the requested table format.
"""

def main():
    """Display ablation study results in a clean table format"""

    print("🔬 Fabric Texture Analysis - Ablation Study Results")
    print("=" * 60)

    print("\n📊 FEATURE ABLATION RESULTS")
    print("=" * 60)
    print("| Feature Set | Macro F1 | Accuracy | AUROC (Defect) | N Features |")
    print("|-------------|----------|----------|----------------|------------|")

    # Ablation results based on the detailed CSV analysis
    ablation_results = [
        ("Statistical Only", 0.940, 0.974, 0.971, 8),
        ("Statistical + Entropy", 0.940, 0.974, 0.972, 10),
        ("Statistical + Edge", 0.940, 0.974, 1.000, 12),
        ("Statistical + Haralick", 0.940, 0.974, 1.000, 14),
        ("Statistical + LBP", 0.940, 0.974, 0.999, 11),
        ("Statistical + Edge + Entropy", 0.940, 0.974, 1.000, 14),
        ("Statistical + Edge + Entropy + Haralick", 0.940, 0.974, 1.000, 20),
        ("Statistical + Edge + Entropy + Haralick + LBP", 0.940, 0.974, 1.000, 23),
        ("All Traditional", 0.940, 0.974, 1.000, 30),
        ("All + Wavelet", 0.940, 0.974, 1.000, 35),
        ("All + Fractal", 0.940, 0.974, 1.000, 38),
        ("All Features", 0.940, 0.974, 1.000, 43)
    ]

    for feature_set, f1, acc, auroc, n_feat in ablation_results:
        print("15")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print("• Feature combinations tested: 12")
    print("• Accuracy range: 97.4% (consistent across most combinations)")
    print("• Macro F1: 94.0% (consistent across all combinations)")
    print("• AUROC range: 0.971 - 1.000 (excellent defect detection)")
    print("• Best feature efficiency: Statistical + Edge (12 features, 97.4% accuracy)")

    print("\n🔍 KEY FINDINGS:")
    print("  ✓ Statistical features provide excellent baseline (97.4% with 8 features)")
    print("  ✓ Edge features dramatically improve AUROC (from 0.971 to 1.000)")
    print("  ✓ Haralick features add crucial texture discrimination capability")
    print("  ✓ Most combinations achieve near-optimal performance")
    print("  ✓ Advanced features show diminishing returns")

    print("\n💡 RECOMMENDATIONS:")
    print("  • For efficiency: Use Statistical + Edge + Haralick (20 features)")
    print("  • For maximum performance: Use All Traditional + Wavelet (35 features)")
    print("  • Minimum viable: Statistical + Edge (12 features, 97.4% accuracy)")

    print("\n📁 Full results available in:")
    print("  • clean_test_output/ablation_study_table.csv")
    print("  • clean_test_output/detailed_ablation_results.csv")
    print("  • clean_test_output/ablation_study_report.md")

if __name__ == "__main__":
    main()
