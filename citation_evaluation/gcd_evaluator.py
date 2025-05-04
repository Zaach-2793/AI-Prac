import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from .base_evaluator import BaseCitationEvaluator

class GCDSubtopicEvaluator(BaseCitationEvaluator):
    """
    Evaluates hierarchical clustering using the Greatest Common Denominator (GCD) subtopic approach.
    
    For citation pairs, calculates the most specific subtopic level that contains both papers,
    and compares with random paper pairs. Cited papers should share more specific subtopics
    with the citing papers than random pairs would.
    """
    
    def find_gcd_subtopic(self, paper1: str, paper2: str) -> Optional[int]:
        """
        Finds the Greatest Common Denominator (GCD) subtopic level for two papers.
        
        The GCD subtopic is the most specific (highest-level) subtopic that contains
        both papers.
        """
        # Start from the most specific level and move up
        for level in sorted(self.hierarchy.keys(), reverse=True):
            # Skip if either paper not found at this level
            if (paper1 not in self.paper_clusters[level] or 
                paper2 not in self.paper_clusters[level]):
                continue
                
            # Check if papers are in the same cluster at this level
            cluster1 = self.paper_clusters[level][paper1]
            cluster2 = self.paper_clusters[level][paper2]
            
            if cluster1 == cluster2:
                return level
                
        return None
    
    def evaluate_gcd(
        self, 
        n_random_pairs: int = 1000, 
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Evaluates the GCD subtopic metric for citation pairs vs. random pairs.
        """
        np.random.seed(random_seed)
        
        all_papers = self.get_all_papers()
        results = []
        
        # Process citation pairs
        for citing, cited_set in self.citations.items():
            for cited in cited_set:
                # Skip if either paper not in hierarchy
                if citing not in all_papers or cited not in all_papers:
                    continue
                    
                gcd_level = self.find_gcd_subtopic(citing, cited)
                
                if gcd_level is not None:
                    results.append({
                        'paper1': citing,
                        'paper2': cited,
                        'is_citation': 1,
                        'gcd_level': gcd_level
                    })
        
        # Sample random pairs
        processed_pairs = set((r['paper1'], r['paper2']) for r in results)
        
        random_pairs_added = 0
        attempts = 0
        max_attempts = n_random_pairs * 10  # Avoid infinite loop
        
        while random_pairs_added < n_random_pairs and attempts < max_attempts:
            attempts += 1
            
            paper1 = np.random.choice(all_papers)
            paper2 = np.random.choice(all_papers)
            
            # Skip self-pairs or existing citation pairs
            if paper1 == paper2 or (paper1, paper2) in processed_pairs:
                continue
                
            gcd_level = self.find_gcd_subtopic(paper1, paper2)
            
            if gcd_level is not None:
                results.append({
                    'paper1': paper1,
                    'paper2': paper2,
                    'is_citation': 0,
                    'gcd_level': gcd_level
                })
                processed_pairs.add((paper1, paper2))
                random_pairs_added += 1
        
        return pd.DataFrame(results)
    
    def analyze_gcd_results(self, gcd_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyzes GCD results and computes key metrics.
        """
        if gcd_df is None:
            gcd_df = self.evaluate_gcd()
            
        # Calculate statistics for citation and random pairs
        citation_gcds = gcd_df[gcd_df['is_citation'] == 1]['gcd_level']
        random_gcds = gcd_df[gcd_df['is_citation'] == 0]['gcd_level']
        
        citation_mean = citation_gcds.mean()
        random_mean = random_gcds.mean()
        difference = citation_mean - random_mean
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(citation_gcds, random_gcds, equal_var=False)
        
        return {
            'citation_gcd_mean': citation_mean,
            'random_gcd_mean': random_mean,
            'gcd_difference': difference,
            'is_significant': p_value < 0.05,
            'p_value': p_value,
            't_statistic': t_stat
        }
        
    def visualize_gcd_comparison(
        self, 
        gcd_df: Optional[pd.DataFrame] = None,
        title: str = 'GCD Subtopic Levels: Citation Pairs vs. Random Pairs'
    ) -> plt.Figure:
        """
        Visualizes the comparison of GCD subtopic levels.
        """
        if gcd_df is None:
            gcd_df = self.evaluate_gcd()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate GCD level distributions
        citation_gcds = gcd_df[gcd_df['is_citation'] == 1]['gcd_level']
        random_gcds = gcd_df[gcd_df['is_citation'] == 0]['gcd_level']
        
        # Create histograms
        bins = range(min(gcd_df['gcd_level']) - 1, max(gcd_df['gcd_level']) + 2)
        
        ax.hist(
            [citation_gcds, random_gcds],
            bins=bins,
            alpha=0.7,
            label=['Citation Pairs', 'Random Pairs']
        )
        
        # Calculate and display means
        citation_mean = citation_gcds.mean()
        random_mean = random_gcds.mean()
        
        ax.axvline(citation_mean, color='blue', linestyle='--', linewidth=2)
        ax.axvline(random_mean, color='orange', linestyle='--', linewidth=2)
        
        ax.set_xlabel('GCD Subtopic Level (higher = more specific)')
        ax.set_ylabel('Number of Paper Pairs')
        ax.set_title(title)
        ax.legend()
        ax.text(
            0.05, 0.95, 
            f'Citation Mean: {citation_mean:.2f}\nRandom Mean: {random_mean:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
        )
        
        return fig