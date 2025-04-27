import pandas as pd
import json
from pathlib import Path
import random

def generate_test_hierarchy(
    num_papers=100,
    levels=3,
    branching_factor=2,
    output_dir='generated_test_data'
):
    """
    Generate a hierarchical clustering structure with papers.
    
    Args:
        num_papers: Number of papers to include
        levels: Number of hierarchy levels
        branching_factor: Number of child clusters per parent
        output_dir: Where to save the data
        
    Returns:
        Tuple of (hierarchy dict, paper_ids list)
    """
    paper_ids = [f"paper{i}" for i in range(1, num_papers + 1)]
    
    # Create hierarchy (top-down approach)
    hierarchy = {0: {'0': paper_ids.copy()}}  # Root contains all papers
    
    # For each level below root
    for level in range(1, levels):
        hierarchy[level] = {}
        
        # For each parent cluster in previous level
        for parent_id, parent_papers in hierarchy[level-1].items():
            # Skip if cluster too small to split
            if len(parent_papers) < branching_factor:
                continue
                
            # Shuffle papers for random assignment
            papers_to_assign = parent_papers.copy()
            random.shuffle(papers_to_assign)
            
            # Split into child clusters
            papers_per_child = len(papers_to_assign) // branching_factor
            remainder = len(papers_to_assign) % branching_factor
            
            for i in range(branching_factor):
                # Calculate papers for this child
                if i < remainder:
                    n_papers = papers_per_child + 1
                else:
                    n_papers = papers_per_child
                
                # Create cluster if it would have papers
                if n_papers > 0:
                    child_id = f"{parent_id}.{i}"
                    child_papers = papers_to_assign[:n_papers]
                    hierarchy[level][child_id] = child_papers
                    papers_to_assign = papers_to_assign[n_papers:]
    
    return hierarchy, paper_ids

def generate_citation_network(
    hierarchy,
    paper_ids,
    same_cluster_citation_prob=0.3,
    parent_cluster_citation_prob=0.1,
    random_citation_prob=0.01,
    output_dir='generated_test_data'
):
    """
    Generate citation relationships between papers based on cluster membership.
    
    Args:
        hierarchy: Hierarchical clustering structure
        paper_ids: List of paper IDs
        same_cluster_citation_prob: Probability of citing papers in same finest cluster
        parent_cluster_citation_prob: Probability of citing papers in parent cluster
        random_citation_prob: Probability of citing random papers
        output_dir: Where to save the data
        
    Returns:
        DataFrame with citation relationships
    """
    # Find finest level (most specific clusters)
    max_level = max(hierarchy.keys())
    
    # Build paper to cluster mapping
    paper_to_cluster = {}
    for level in range(max_level + 1):
        for cluster_id, papers in hierarchy[level].items():
            for paper in papers:
                if paper not in paper_to_cluster:
                    paper_to_cluster[paper] = {}
                paper_to_cluster[paper][level] = cluster_id
    
    citations = []
    
    # Generate citations for each paper
    for paper in paper_ids:
        # Skip papers not assigned to clusters at all levels
        if paper not in paper_to_cluster or max_level not in paper_to_cluster[paper]:
            continue
            
        # Get finest cluster for this paper
        finest_cluster = paper_to_cluster[paper][max_level]
        
        # Get papers in same finest cluster
        same_cluster_papers = [p for p in hierarchy[max_level][finest_cluster] if p != paper]
        
        # Cite papers in same cluster with high probability
        for potential_citation in same_cluster_papers:
            if random.random() < same_cluster_citation_prob:
                citations.append({
                    'paper_id': paper, 
                    'cited_paper_id': potential_citation
                })
        
        # For each intermediate level, cite papers in same cluster but different sub-cluster
        for level in range(1, max_level):
            if level not in paper_to_cluster[paper]:
                continue
                
            parent_cluster = paper_to_cluster[paper][level]
            
            # Get papers in same parent cluster but different sub-clusters
            papers_in_parent = set(hierarchy[level][parent_cluster])
            if max_level in paper_to_cluster[paper]:
                papers_in_sub = set(hierarchy[max_level][paper_to_cluster[paper][max_level]])
                parent_only_papers = [p for p in papers_in_parent - papers_in_sub if p != paper]
                
                # Cite with medium probability
                for potential_citation in parent_only_papers:
                    if random.random() < parent_cluster_citation_prob:
                        citations.append({
                            'paper_id': paper, 
                            'cited_paper_id': potential_citation
                        })
        
        # Cite random papers with low probability
        other_papers = [p for p in paper_ids if p != paper and 
                        (p not in same_cluster_papers) and
                        (max_level in paper_to_cluster[paper])]
        
        # Sample subset for efficiency
        sample_size = min(50, len(other_papers))
        if sample_size > 0:
            sample_papers = random.sample(other_papers, sample_size)
            
            for potential_citation in sample_papers:
                if random.random() < random_citation_prob:
                    citations.append({
                        'paper_id': paper, 
                        'cited_paper_id': potential_citation
                    })
    
    citation_df = pd.DataFrame(citations)
    
    # Save data
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    with open(output_path / "hierarchy.json", 'w') as f:
        json.dump(hierarchy, f)
    
    citation_df.to_csv(output_path / "citations.csv", index=False)
    
    return citation_df

# Generate both small and large test datasets if script is run directly
if __name__ == "__main__":
    # Small dataset for quick tests
    print("Generating small test dataset...")
    small_h, small_papers = generate_test_hierarchy(
        num_papers=30,
        levels=3,
        output_dir="small_test_data"
    )
    small_c = generate_citation_network(
        small_h,
        small_papers,
        output_dir="small_test_data"
    )
    print(f"Created {len(small_papers)} papers with {len(small_c)} citations")
    
    # Larger dataset for more realistic testing
    print("Generating large test dataset...")
    large_h, large_papers = generate_test_hierarchy(
        num_papers=200,
        levels=4,
        branching_factor=3,
        output_dir="large_test_data"
    )
    large_c = generate_citation_network(
        large_h,
        large_papers,
        output_dir="large_test_data"
    )
    print(f"Created {len(large_papers)} papers with {len(large_c)} citations")