#!/usr/bin/env python3
"""
CLI Interface for Fabric Analysis Research Framework

Comprehensive command-line tool for fabric classification and micro-damage detection
using machine learning analysis of micro-texture features.

Usage Examples:
--------------
# Run complete fabric analysis
python run_research_analysis.py

# Fabric classification with known labels  
python run_research_analysis.py --target fabric_type

# Focus on damage detection only
python run_research_analysis.py --damage-detection-only

# Quick analysis for large datasets
python run_research_analysis.py --quick
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from research import ResearchAnalyzer

def main():
    parser = argparse.ArgumentParser(
        description="Fabric Analysis Research Framework - ML analysis for fabric classification and damage detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Complete fabric analysis pipeline
  %(prog)s --target fabric_type               # Fabric classification with labels
  %(prog)s --damage-detection-only            # Focus on micro-damage detection
  %(prog)s --quick                           # Fast analysis for large datasets
  %(prog)s --contamination 0.05               # Expect 5%% damaged samples
  %(prog)s --output fabric_analysis_results   # Custom output directory
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/features.csv',
        help='Path to CSV file containing extracted fabric features (default: data/features.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='fabric_analysis_output',
        help='Output directory for analysis results (default: fabric_analysis_output)'
    )
    
    parser.add_argument(
        '--manifest', '-m',
        type=str,
        default=None,
        help='Path to manifest CSV with filename,label columns for label integrity'
    )
    
    parser.add_argument(
        '--deep-features',
        type=str,
        default=None,
        help='Path to deep features CSV (required for deep-only or hybrid modes)'
    )
    
    # Analysis type arguments
    parser.add_argument(
        '--target', '-t',
        type=str,
        default=None,
        help='Target column name for fabric classification (e.g., fabric_type, quality_grade)'
    )
    
    parser.add_argument(
        '--feature-mode',
        type=str,
        choices=['handcrafted', 'deep_only', 'hybrid'],
        default='handcrafted',
        help='Feature mode: handcrafted (default), deep_only, or hybrid'
    )
    
    parser.add_argument(
        '--fusion',
        type=str,
        choices=['concatenate', 'weighted', 'attention'],
        default='concatenate',
        help='Fusion strategy for hybrid mode (default: concatenate)'
    )
    
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=1,
        help='Number of random seeds for multi-seed evaluation (default: 1, use 5 for multi-seed)'
    )
    
    parser.add_argument(
        '--save-splits-dir',
        type=str,
        default=None,
        help='Directory to save/load sample-aware splits (enables split persistence)'
    )
    
    parser.add_argument(
        '--use-saved-splits',
        action='store_true',
        help='Load saved splits from save-splits-dir instead of creating new ones'
    )
    
    parser.add_argument(
        '--damage-detection-only',
        action='store_true',
        help='Run only anomaly detection for micro-damage identification (skip supervised learning)'
    )
    
    parser.add_argument(
        '--fabric-classification-only',
        action='store_true',
        help='Run only supervised fabric classification (skip anomaly detection)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--contamination', '-c',
        type=float,
        default=0.1,
        help='Expected proportion of damaged/defective samples (default: 0.1 for 10%%)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing in supervised learning (default: 0.2)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    # Performance options
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick analysis mode (skip robustness testing and detailed visualizations)'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation (faster for large datasets)'
    )
    
    # Feature analysis options
    parser.add_argument(
        '--focus-features',
        type=str,
        choices=['fractal', 'texture', 'statistical', 'edge', 'all'],
        default='all',
        help='Focus analysis on specific feature types (default: all)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        print("Make sure to run the data collection pipeline first:")
        print("  python main.py")
        sys.exit(1)
    
    if args.contamination < 0.01 or args.contamination > 0.5:
        print(f"Warning: Contamination rate {args.contamination} seems unusual.")
        print("Typical values: 0.02-0.15 (2-15% damaged samples)")
    
    # Create analyzer
    print("🔬 Fabric Analysis Research Framework")
    print("=" * 50)
    print(f"Data source: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Target column: {args.target or 'None (will create synthetic labels)'}")
    print(f"Expected damage rate: {args.contamination:.1%}")
    print()
    
    analyzer = ResearchAnalyzer(args.data, args.output, 
                               manifest_path=args.manifest,
                               deep_features_path=args.deep_features)
    
    try:
        # Load and prepare data
        print("📊 Loading and preparing fabric feature data...")
        analyzer.load_and_prepare_data(target_column=args.target, 
                                      use_manifest=(args.manifest is not None),
                                      migrate_schema=True)
        
        # Determine if we should use multi-seed evaluation
        use_multiseed = args.n_seeds > 1
        
        # Set splits directory
        splits_dir = args.save_splits_dir if args.save_splits_dir else str(analyzer.output_dir / "splits")
        
        # Run analysis based on selected mode
        if args.damage_detection_only:
            print("🔍 Running micro-damage detection analysis...")
            analyzer.run_anomaly_detection(contamination=args.contamination)
            
        elif args.fabric_classification_only:
            if analyzer.y_encoded is not None:
                print("🏷️  Running fabric classification analysis...")
                if use_multiseed:
                    print(f"Using multi-seed evaluation with {args.n_seeds} seeds...")
                    analyzer.run_supervised_analysis_multiseed(
                        test_size=args.test_size,
                        val_size=0.1,
                        cv_folds=args.cv_folds,
                        n_seeds=args.n_seeds,
                        feature_mode=args.feature_mode,
                        fusion_strategy=args.fusion,
                        save_splits_dir=splits_dir,
                        use_saved_splits=args.use_saved_splits
                    )
                else:
                    analyzer.run_supervised_analysis(
                        test_size=args.test_size, 
                        cv_folds=args.cv_folds
                    )
            else:
                print("Warning: No target column found for fabric classification.")
                print("Running with synthetic labels for demonstration...")
                analyzer.run_supervised_analysis(
                    test_size=args.test_size, 
                    cv_folds=args.cv_folds
                )
                
        else:
            # Complete analysis pipeline
            print("🚀 Running complete fabric analysis pipeline...")
            
            # Fabric classification (if labels available)
            if analyzer.y_encoded is not None:
                print("\n1️⃣  Fabric Classification Analysis")
                if use_multiseed:
                    print(f"Using multi-seed evaluation with {args.n_seeds} seeds...")
                    analyzer.run_supervised_analysis_multiseed(
                        test_size=args.test_size,
                        val_size=0.1,
                        cv_folds=args.cv_folds,
                        n_seeds=args.n_seeds,
                        feature_mode=args.feature_mode,
                        fusion_strategy=args.fusion,
                        save_splits_dir=splits_dir,
                        use_saved_splits=args.use_saved_splits
                    )
                else:
                    analyzer.run_supervised_analysis(
                        test_size=args.test_size, 
                        cv_folds=args.cv_folds
                    )
            
            # Micro-damage detection
            print("\n2️⃣  Micro-Damage Detection Analysis")
            analyzer.run_anomaly_detection(contamination=args.contamination)
            
            # Feature analysis
            print("\n3️⃣  Feature Analysis")
            analyzer.run_feature_analysis()
            
            # Dimensionality reduction
            print("\n4️⃣  Dimensionality Reduction")
            analyzer.run_dimensionality_reduction()
            
            # Clustering analysis
            print("\n5️⃣  Fabric Clustering Analysis")
            analyzer.run_clustering_analysis()
            
            # Robustness testing (unless quick mode)
            if not args.quick and analyzer.y_encoded is not None:
                print("\n6️⃣  Model Robustness Testing")
                analyzer.run_robustness_testing()
        
        # Generate comprehensive report
        print("\n📋 Generating Research Report...")
        analyzer.generate_research_report()
        
        # Success message
        print("\n🎉 Fabric Analysis Complete!")
        print(f"✅ Results saved to: {args.output}/")
        print(f"📊 View report: {args.output}/research_report.md")
        
        # Summary of key findings
        if hasattr(analyzer, 'results'):
            print("\n📈 Quick Summary:")
            
            # Supervised results summary
            if 'supervised_results' in analyzer.results and analyzer.results['supervised_results']:
                results = analyzer.results['supervised_results']
                
                # Check if multi-seed results
                if isinstance(results, dict) and 'aggregated' in results:
                    # Multi-seed results
                    aggregated = results['aggregated']
                    if aggregated:
                        # Find best model by mean accuracy
                        best_model_name = max(aggregated.items(), 
                                            key=lambda x: x[1]['accuracy']['mean'] if x[1]['accuracy'] else 0)[0]
                        best_metrics = aggregated[best_model_name]
                        acc_mean = best_metrics['accuracy']['mean']
                        acc_std = best_metrics['accuracy']['std']
                        print(f"   🏆 Best fabric classifier: {best_model_name}")
                        print(f"      Accuracy: {acc_mean:.3f} ± {acc_std:.3f} (across {args.n_seeds} seeds)")
                        if best_metrics.get('macro_f1'):
                            f1_mean = best_metrics['macro_f1']['mean']
                            print(f"      Macro F1: {f1_mean:.3f}")
                else:
                    # Single-seed results
                    if results:
                        best_model = max(results.items(), 
                                       key=lambda x: x[1].get('test_accuracy', 0))
                        acc = best_model[1].get('test_accuracy', 0)
                        print(f"   🏆 Best fabric classifier: {best_model[0]} ({acc:.3f} accuracy)")
            
            # Anomaly detection summary  
            if 'anomaly_results' in analyzer.results and analyzer.results['anomaly_results']:
                anomaly_results = analyzer.results['anomaly_results']
                if 'isolation_forest' in anomaly_results:
                    damage_detected = anomaly_results['isolation_forest']['n_anomalies']
                    damage_percent = anomaly_results['isolation_forest']['anomaly_percentage']
                    print(f"   🔍 Potential damage detected: {damage_detected} samples ({damage_percent:.1f}%)")
            
        print(f"\n📁 Explore visualizations in: {args.output}/visualizations/")
        print(f"📊 Check detailed metrics in: {args.output}/evaluation/")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the feature extraction has been completed:")
        print("  python main.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        print("Check your data format and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 