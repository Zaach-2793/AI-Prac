import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

from .coverage_evaluator import SubtopicCoverageEvaluator
from .gcd_evaluator import GCDSubtopicEvaluator
from .reporter import CitationEvaluationReporter

def load_data(data_dir='hierarchy_data'):
    """Load data for evaluation."""
    data_path = Path(data_dir)
    
    # Load hierarchy
    with open(data_path / "hierarchy.json", 'r') as f:
        hierarchy_str = json.load(f)
        
    # Convert string keys to integers for levels
    hierarchy = {int(k): v for k, v in hierarchy_str.items()}
    
    # Load citations
    citations = pd.read_csv(data_path / "citations.csv")
    
    return hierarchy, citations

def run_evaluation():
    """Run both evaluation methods and display results."""
    # Load data
    hierarchy, citations = load_data()
    
    print(f"Loaded hierarchy with {len(hierarchy)} levels")
    print(f"Loaded {len(citations)} citation relationships")
    
    # Evaluate using SubtopicCoverage method
    print("\n--- Subtopic Coverage Evaluation ---")
    coverage_eval = SubtopicCoverageEvaluator(hierarchy, citations)
    coverage_df = coverage_eval.evaluate_coverage()
    
    # Display results
    level_coverage = coverage_df.groupby('level')['coverage'].mean()
    for level, coverage in level_coverage.items():
        print(f"Level {level} average coverage: {coverage:.4f}")
    
    analysis = coverage_eval.analyze_coverage_trend()
    print(f"Coverage decreasing with broader topics: {analysis['is_decreasing']}")    
    fig1 = coverage_eval.visualize_coverage_trend()
    
    # Evaluate using GCD Subtopic method
    print("\n--- GCD Subtopic Evaluation ---")
    gcd_eval = GCDSubtopicEvaluator(hierarchy, citations)
    gcd_df = gcd_eval.evaluate_gcd(n_random_pairs=20)
    
    # Display results
    citation_gcd = gcd_df[gcd_df['is_citation'] == 1]['gcd_level'].mean()
    random_gcd = gcd_df[gcd_df['is_citation'] == 0]['gcd_level'].mean()
    
    print(f"Citation GCD mean: {citation_gcd:.2f}")
    print(f"Random GCD mean: {random_gcd:.2f}")
    print(f"Difference: {citation_gcd - random_gcd:.2f}")    
    fig2 = gcd_eval.visualize_gcd_comparison()
    
    # Generate combined report
    print("\n--- Generating Full Report ---")
    reporter = CitationEvaluationReporter(hierarchy, citations)
    results = reporter.generate_full_report(output_path="evaluation_results")
    
    print(f"Report saved to 'evaluation_results' directory")
    print(f"Overall hierarchy validity: {results['overall_quality']['overall_valid']}")
    
    plt.show()

if __name__ == "__main__":
    run_evaluation()