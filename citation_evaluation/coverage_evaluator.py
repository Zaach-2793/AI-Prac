import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from .base_evaluator import BaseCitationEvaluator

class SubtopicCoverageEvaluator(BaseCitationEvaluator):
    """
    Evaluates hierarchical clustering using the subtopic coverage approach.
    
    For each paper, calculates the percentage of papers in its cluster at each level
    that are cited by the paper. Ideally, this percentage should decrease as we move
    from specific to broader topics.
    """
    
    def evaluate_coverage(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluates the "subtopic coverage" metric for papers across hierarchy levels.
        """
        # Select papers that have citations
        citing_papers = list(self.citations.keys())
        
        # Sample papers if requested
        if sample_size and len(citing_papers) > sample_size:
            np.random.shuffle(citing_papers)
            citing_papers = citing_papers[:sample_size]
            
        results = []
        
        for paper_id in citing_papers:
            cited_papers = self.citations[paper_id]
            
            for level in sorted(self.hierarchy.keys()):
                # Skip if paper not found at this level
                if paper_id not in self.paper_clusters[level]:
                    continue
                    
                cluster_id = self.paper_clusters[level][paper_id]
                cluster_papers = set(self.hierarchy[level][cluster_id])
                
                # Remove self from consideration
                cluster_papers.discard(paper_id)
                
                if not cluster_papers:
                    continue
                
                # Calculate coverage
                cited_in_cluster = cited_papers.intersection(cluster_papers)
                coverage = len(cited_in_cluster) / len(cluster_papers)
                
                results.append({
                    'paper_id': paper_id,
                    'level': level,
                    'cluster_id': cluster_id,
                    'cluster_size': len(cluster_papers),
                    'cited_count': len(cited_in_cluster),
                    'coverage': coverage
                })
                
        return pd.DataFrame(results)
    
    def analyze_coverage_trend(self, coverage_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyzes coverage trends across hierarchy levels.
        """
        if coverage_df is None:
            coverage_df = self.evaluate_coverage()
            
        # Calculate mean coverage by level
        level_coverage = coverage_df.groupby('level')['coverage'].agg(['mean', 'std']).reset_index()
        
        # Check if coverage decreases as we move up
        is_decreasing = level_coverage['mean'].sort_index(ascending=False).is_monotonic_decreasing
        
        return {
            'coverage_by_level': dict(zip(level_coverage['level'], level_coverage['mean'])),
            'is_decreasing': is_decreasing,
            'level_differences': [level_coverage.loc[i, 'mean'] - level_coverage.loc[i+1, 'mean'] 
                                  for i in range(len(level_coverage)-1)]
        }
        
    def visualize_coverage_trend(
        self, 
        coverage_df: Optional[pd.DataFrame] = None,
        title: str = 'Subtopic Coverage Trend Across Hierarchy Levels'
    ) -> plt.Figure:
        """
        Visualizes the subtopic coverage trend across hierarchy levels.
        """
        if coverage_df is None:
            coverage_df = self.evaluate_coverage()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean coverage by level
        level_coverage = coverage_df.groupby('level')['coverage'].agg(['mean', 'std']).reset_index()
        
        # Plot mean coverage with error bars
        ax.errorbar(
            level_coverage['level'], 
            level_coverage['mean'] * 100, 
            marker='o',
            linestyle='-',
            capsize=5
        )
        
        ax.set_xlabel('Hierarchy Level (0 = broadest)')
        ax.set_ylabel('Mean Coverage (% of cluster papers cited)')
        ax.set_xticks([0,1,2,3])
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig