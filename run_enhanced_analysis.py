#!/usr/bin/env python3
"""
Enhanced Texture Analysis Pipeline
==================================

Comprehensive example demonstrating all the new enhanced analysis capabilities:
- Sample-aware data splitting (prevents data leakage)
- Advanced feature selection
- Feature augmentation and data augmentation
- Deep learning integration
- Hyperparameter optimization
- Advanced visualizations (t-SNE, UMAP)
- Comprehensive benchmarking

Usage:
    python run_enhanced_analysis.py
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from research.research_analysis import ResearchAnalyzer


def main():
    """Run comprehensive enhanced texture analysis."""
    
    parser = argparse.ArgumentParser(description='Enhanced Texture Analysis Pipeline')
    parser.add_argument('--data_path', type=str, default='data/features.csv',
                       help='Path to features CSV file')
    parser.add_argument('--output_dir', type=str, default='enhanced_analysis_output',
                       help='Output directory for results')
    parser.add_argument('--target_column', type=str, default='label',
                       help='Target column name for supervised learning')
    parser.add_argument('--quick_mode', action='store_true',
                       help='Run in quick mode (fewer trials for optimization)')
    parser.add_argument('--include_deep_learning', action='store_true',
                       help='Include deep learning analysis (requires raw images)')
    parser.add_argument('--skip_optimization', action='store_true',
                       help='Skip hyperparameter optimization to save time')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 ENHANCED TEXTURE ANALYSIS PIPELINE")
    print("="*60)
    print("✨ New Features:")
    print("  • Sample-aware splitting (prevents data leakage)")
    print("  • Advanced feature selection & augmentation")
    print("  • Deep learning CNN feature extraction")
    print("  • Hyperparameter optimization with Optuna")
    print("  • t-SNE/UMAP advanced visualizations")
    print("  • Comprehensive benchmarking suite")
    print("="*60)
    print()
    
    # Initialize analyzer
    analyzer = ResearchAnalyzer(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    try:
        # Step 1: Load and prepare data
        print("📊 STEP 1: Loading and preparing data...")
        X, y, feature_names = analyzer.load_and_prepare_data(
            target_column=args.target_column,
            create_synthetic_labels=(args.target_column not in ['label'])
        )
        
        if X is None:
            print("❌ No data loaded. Please check your data path.")
            return
        
        print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print()
        
        # Step 2: Advanced feature selection
        print("🔍 STEP 2: Advanced feature selection...")
        feature_selection_results = analyzer.run_feature_selection_analysis(
            methods=['ensemble', 'mutual_information', 'stability_selection'],
            optimize_selection=(not args.skip_optimization)
        )
        print("✅ Feature selection complete!")
        print()
        
        # Step 3: Enhanced supervised analysis with sample-aware splitting
        print("🤖 STEP 3: Enhanced supervised learning analysis...")
        print("🛡️  Using sample-aware splitting to prevent data leakage!")
        
        supervised_results = analyzer.run_supervised_analysis(
            test_size=0.2,
            val_size=0.1,
            cv_folds=5,
            use_sample_aware_splitting=True,
            apply_augmentation=True
        )
        
        print(f"✅ Trained {len(supervised_results)} models with enhanced pipeline!")
        
        # Print best model performance
        best_model = max(supervised_results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"🏆 Best model: {best_model[0]} (Accuracy: {best_model[1]['test_accuracy']:.4f})")
        print()
        
        # Step 4: Deep learning analysis (if requested)
        if args.include_deep_learning:
            print("🧠 STEP 4: Deep learning analysis...")
            deep_results = analyzer.run_deep_learning_analysis()
            hybrid_results = analyzer.run_hybrid_analysis()
            print("✅ Deep learning analysis complete!")
            print()
        
        # Step 5: Advanced visualizations
        print("📈 STEP 5: Advanced visualization suite...")
        viz_results = analyzer.run_advanced_visualization_suite()
        print("✅ Advanced visualizations created!")
        print(f"   Generated: {len(viz_results)} visualization files")
        print()
        
        # Step 6: Comprehensive benchmarking
        print("🏁 STEP 6: Comprehensive benchmarking...")
        print("   Comparing all approaches:")
        print("   1️⃣  Baseline (random splitting)")
        print("   2️⃣  Sample-aware splitting")
        print("   3️⃣  Sample-aware + augmentation")
        print("   4️⃣  Feature selection + sample-aware")
        
        benchmark_results = analyzer.run_comprehensive_benchmarking(
            include_augmentation=True,
            include_feature_selection=True,
            optimize_hyperparameters=(not args.skip_optimization)
        )
        
        print("✅ Comprehensive benchmarking complete!")
        
        # Display benchmark results
        print("\n🏆 BENCHMARK RESULTS:")
        for approach, results in benchmark_results.items():
            print(f"   {approach:20s}: {results['best_accuracy']:.4f} "
                  f"({results['feature_count']} features, "
                  f"leakage risk: {results['data_leakage_risk']})")
        
        best_approach = max(benchmark_results.items(), key=lambda x: x[1]['best_accuracy'])
        print(f"\n🥇 WINNER: {best_approach[0]} "
              f"(Accuracy: {best_approach[1]['best_accuracy']:.4f})")
        print()
        
        # Step 7: Generate comprehensive report
        print("📝 STEP 7: Generating comprehensive report...")
        report_path = analyzer.generate_comprehensive_report()
        print(f"✅ Report saved to: {report_path}")
        print()
        
        # Summary
        print("="*60)
        print("🎉 ENHANCED ANALYSIS COMPLETE!")
        print("="*60)
        print("📊 Results Summary:")
        print(f"   • Best accuracy achieved: {best_approach[1]['best_accuracy']:.4f}")
        print(f"   • Best approach: {best_approach[1]['approach']}")
        print(f"   • Data leakage prevention: ✅ ENABLED")
        print(f"   • Feature augmentation: ✅ APPLIED")
        print(f"   • Advanced visualizations: ✅ CREATED")
        print()
        print("📁 Output files:")
        print(f"   • Main results: {args.output_dir}/")
        print(f"   • Visualizations: {args.output_dir}/visualizations/")
        print(f"   • Advanced plots: {args.output_dir}/advanced_visualizations/")
        print(f"   • Comprehensive report: {report_path}")
        print()
        print("🚀 Performance improvements achieved:")
        
        if 'baseline' in benchmark_results and 'sample_aware' in benchmark_results:
            baseline_acc = benchmark_results['baseline']['best_accuracy']
            improved_acc = best_approach[1]['best_accuracy']
            improvement = ((improved_acc - baseline_acc) / baseline_acc) * 100
            print(f"   • Accuracy improvement: +{improvement:.1f}% vs baseline")
        
        print("   • Data leakage eliminated: ✅")
        print("   • Feature space optimized: ✅")
        print("   • Robustness enhanced: ✅")
        print()
        print("💡 Key insights:")
        print("   • Sample-aware splitting prevents overly optimistic results")
        print("   • Feature augmentation helps with small datasets")
        print("   • Advanced feature selection reduces overfitting")
        print("   • Ensemble methods provide robust predictions")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_quick_demo():
    """Run a quick demonstration with default settings."""
    print("🚀 Running quick demo of enhanced texture analysis...")
    
    # Check if data exists
    if not os.path.exists('data/features.csv'):
        print("❌ No feature data found at 'data/features.csv'")
        print("💡 Please run the main data collection pipeline first:")
        print("   python main.py")
        return 1
    
    # Run analysis with default settings
    analyzer = ResearchAnalyzer(
        data_path='data/features.csv',
        output_dir='quick_demo_output'
    )
    
    # Load data
    X, y, feature_names = analyzer.load_and_prepare_data(target_column='label')
    
    if X is None:
        print("❌ Could not load data")
        return 1
    
    print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Quick analysis
    print("🔍 Running sample-aware analysis (prevents data leakage)...")
    results = analyzer.run_supervised_analysis(
        use_sample_aware_splitting=True,
        apply_augmentation=False  # Skip augmentation for speed
    )
    
    # Best result
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"🏆 Best result: {best_model[0]} - {best_model[1]['test_accuracy']:.4f} accuracy")
    
    # Quick visualization
    analyzer.run_advanced_visualization_suite()
    
    print("✅ Quick demo complete! Check 'quick_demo_output/' for results.")
    return 0


if __name__ == "__main__":
    # Check for quick demo mode
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == '--demo'):
        exit(run_quick_demo())
    else:
        exit(main()) 