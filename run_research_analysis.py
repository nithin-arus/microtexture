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
    
    # Analysis type arguments
    parser.add_argument(
        '--target', '-t',
        type=str,
        default=None,
        help='Target column name for fabric classification (e.g., fabric_type, quality_grade)'
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
    
    analyzer = ResearchAnalyzer(args.data, args.output)
    
    try:
        # Load and prepare data
        print("📊 Loading and preparing fabric feature data...")
        analyzer.load_and_prepare_data(target_column=args.target)
        
        # Run analysis based on selected mode
        if args.damage_detection_only:
            print("🔍 Running micro-damage detection analysis...")
            analyzer.run_anomaly_detection(contamination=args.contamination)
            
        elif args.fabric_classification_only:
            if analyzer.y is not None:
                print("🏷️  Running fabric classification analysis...")
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
            if analyzer.y is not None:
                print("\n1️⃣  Fabric Classification Analysis")
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
            if not args.quick and analyzer.y is not None:
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
                best_model = max(analyzer.results['supervised_results'].items(), 
                               key=lambda x: x[1]['test_accuracy'])
                print(f"   🏆 Best fabric classifier: {best_model[0]} ({best_model[1]['test_accuracy']:.3f} accuracy)")
            
            # Anomaly detection summary  
            if 'anomaly_results' in analyzer.results and analyzer.results['anomaly_results']:
                damage_detected = analyzer.results['anomaly_results']['isolation_forest']['n_anomalies']
                damage_percent = analyzer.results['anomaly_results']['isolation_forest']['anomaly_percentage']
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