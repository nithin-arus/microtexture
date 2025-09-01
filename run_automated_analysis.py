#!/usr/bin/env python3
"""
Automated Fabric Analysis Script
Runs the complete analysis pipeline programmatically
"""

import sys
import os
sys.path.append('.')

from research.research_analysis import ResearchAnalyzer
import pandas as pd

def main():
    print('🔬 Starting automated fabric analysis...')
    print('=' * 50)

    # Initialize analyzer
    analyzer = ResearchAnalyzer('data/features.csv', 'automated_analysis_output')

    try:
        # Load data
        print('📊 Loading data...')
        X, y, feature_names = analyzer.load_and_prepare_data(target_column='label')
        print(f'✅ Loaded {X.shape[0]} samples with {X.shape[1]} features')
        print()

        # Run supervised analysis
        print('🤖 Running supervised learning analysis...')
        results = analyzer.run_supervised_analysis(use_sample_aware_splitting=True)
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f'🏆 Best model: {best_model[0]} - {best_model[1]["test_accuracy"]:.4f} accuracy')
        print()

        # Run anomaly detection
        print('🔍 Running anomaly detection...')
        analyzer.run_anomaly_detection(contamination=0.1)
        print('✅ Anomaly detection complete')
        print()

        # Generate report
        print('📋 Generating research report...')
        analyzer.generate_research_report()
        print('✅ Report generated')
        print()

        print('🎉 Automated analysis complete!')
        print('📁 Results saved to: automated_analysis_output/')

        # Summary
        print('\n📈 SUMMARY:')
        print(f'   • Dataset: {X.shape[0]} samples, {X.shape[1]} features')
        print(f'   • Best classifier: {best_model[0]} ({best_model[1]["test_accuracy"]:.4f})')
        print(f'   • Anomaly detection: ✅ Enabled (10% contamination expected)')
        print(f'   • Report: automated_analysis_output/research_report.md')

    except Exception as e:
        print(f'❌ Error during analysis: {e}')
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())

