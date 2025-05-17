import os
import pandas as pd
from typing import Dict, List, Optional

from .coverage_evaluator import SubtopicCoverageEvaluator
from .gcd_evaluator import GCDSubtopicEvaluator

class CitationEvaluationReporter:
    """
    Combines both evaluation approaches and generates overall reports.
    """
    
    def __init__(
        self, 
        hierarchy: Dict[int, Dict[str, List[str]]], 
        citation_data: pd.DataFrame
    ):
        """
        Initialize reporter with hierarchy and citation data.
        """
        self.coverage_evaluator = SubtopicCoverageEvaluator(hierarchy, citation_data)
        self.gcd_evaluator = GCDSubtopicEvaluator(hierarchy, citation_data)
        
    def generate_full_report(
        self, 
        output_path: Optional[str] = None,
        sample_size: int = 1000,
        n_random_pairs: int = 1000
    ) -> Dict:
        """
        Generates comprehensive report with both methods.
        """
        # Run evaluations
        coverage_df = self.coverage_evaluator.evaluate_coverage(sample_size)
        coverage_analysis = self.coverage_evaluator.analyze_coverage_trend(coverage_df)
        
        gcd_df = self.gcd_evaluator.evaluate_gcd(n_random_pairs)
        gcd_analysis = self.gcd_evaluator.analyze_gcd_results(gcd_df)
        
        # Generate visualizations
        coverage_fig = self.coverage_evaluator.visualize_coverage_trend(coverage_df)
        gcd_fig = self.gcd_evaluator.visualize_gcd_comparison(gcd_df)
        
        results = {
            'coverage_analysis': coverage_analysis,
            'gcd_analysis': gcd_analysis,
            'overall_quality': self._calculate_overall_quality(coverage_analysis, gcd_analysis)
        }
        
        # Save results if path provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            
            coverage_df.to_csv(f"{output_path}/subtopic_coverage.csv", index=False)
            gcd_df.to_csv(f"{output_path}/gcd_subtopics.csv", index=False)
            
            coverage_fig.savefig(f"{output_path}/coverage_trend.png", dpi=300, bbox_inches='tight')
            gcd_fig.savefig(f"{output_path}/gcd_comparison.png", dpi=300, bbox_inches='tight')
            
            self._write_report_summary(results, f"{output_path}/evaluation_summary.txt")
        
        return results
    
    def _calculate_overall_quality(self, coverage_analysis, gcd_analysis):
        """Calculates overall quality metrics from both analyses."""
        quality = {}
        
        # Check if coverage decreases with broader topics
        quality['coverage_trend_valid'] = coverage_analysis['is_decreasing']
        
        # Check if citation pairs have more specific GCD than random pairs
        quality['gcd_comparison_valid'] = gcd_analysis['gcd_difference'] > 0
        quality['gcd_significant'] = gcd_analysis['is_significant']
        
        # Overall assessment
        quality['overall_valid'] = (quality['coverage_trend_valid'] and 
                                  quality['gcd_comparison_valid'])
        
        return quality
    
    def _write_report_summary(self, results, filepath):
        """Writes a human-readable summary report."""
        with open(filepath, 'w') as f:
            f.write("Hierarchical Citation Evaluation Summary\n")
            f.write("======================================\n\n")
            
            # Coverage evaluation
            f.write("1. Subtopic Coverage Evaluation\n")
            f.write("------------------------------\n")
            f.write("Subtopic Coverage by Level:\n")
            for level, coverage in results['coverage_analysis']['coverage_by_level'].items():
                f.write(f"Level {level}: {coverage:.4f}\n")
            
            f.write(f"\nCoverage Monotonically Increasing: {results['coverage_analysis']['is_decreasing']}\n")
            
            # GCD evaluation
            f.write("\n2. GCD Subtopic Evaluation\n")
            f.write("-------------------------\n")
            f.write(f"Citation GCD Mean: {results['gcd_analysis']['citation_gcd_mean']:.4f}\n")
            f.write(f"Random GCD Mean: {results['gcd_analysis']['random_gcd_mean']:.4f}\n")
            f.write(f"GCD Difference: {results['gcd_analysis']['gcd_difference']:.4f}\n")
            
            # Overall assessment
            f.write("\n3. Overall Assessment\n")
            f.write("--------------------\n")
            f.write(f"Coverage Trend Valid: {results['overall_quality']['coverage_trend_valid']}\n")
            f.write(f"GCD Comparison Valid: {results['overall_quality']['gcd_comparison_valid']}\n")
            f.write(f"GCD Comparison Statistically Significant: {results['overall_quality']['gcd_significant']}\n")
            f.write(f"Overall Hierarchy Validity: {results['overall_quality']['overall_valid']}\n")