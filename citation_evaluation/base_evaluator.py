import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set

class BaseCitationEvaluator:
    """Base class for citation-based evaluation of hierarchical clustering."""
    
    def __init__(
        self, 
        hierarchy: Dict[int, Dict[str, List[str]]], 
        citation_data: pd.DataFrame
    ):
        """Initialize with hierarchical clustering and citation data."""
        self.hierarchy = hierarchy
        self.citation_data = citation_data
        self.paper_clusters = self._build_paper_cluster_mapping()
        self.citations = self._build_citation_graph()
        
    def _build_paper_cluster_mapping(self) -> Dict[int, Dict[str, str]]:
        """Create mapping from paper_id to cluster_id for each hierarchy level."""
        paper_clusters = {}
        
        for level, clusters in self.hierarchy.items():
            paper_clusters[level] = {}
            for cluster_id, papers in clusters.items():
                for paper_id in papers:
                    paper_clusters[level][paper_id] = cluster_id
                    
        return paper_clusters
    
    def _build_citation_graph(self) -> Dict[str, Set[str]]:
        """Build citation graph as an adjacency list."""
        citations = defaultdict(set)
        
        for _, row in self.citation_data.iterrows():
            citing = row['paper_id']
            cited = row['cited_paper_id']
            citations[citing].add(cited)
            
        return citations
    
    def get_all_papers(self) -> List[str]:
        """Return a list of all paper IDs in the hierarchy."""
        all_papers = set()
        for level in self.hierarchy.keys():
            for cluster in self.hierarchy[level].values():
                all_papers.update(cluster)
        return list(all_papers)