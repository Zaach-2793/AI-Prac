import pandas as pd
import re

def normalize_id_for_comparison(paper_id):
    """
    Normalize paper IDs: remove version numbers, convert to lowercase, etc.
    """
    if not isinstance(paper_id, str):
        return ''
    
    paper_id = paper_id.lower()
    paper_id = paper_id.replace('https://', 'http://')
    
    # Remove trailing slashes
    paper_id = paper_id.rstrip('/')
    
    # Remove version numbers (v1, v2, etc.) for comparison only
    paper_id = re.sub(r'v\d+$', '', paper_id)
    
    return paper_id

def get_direct_version_replacement(citation_file, arxiv_file, output_file):
    """
    Matches and replaces cited_paper_id with the versioned equivalent from dataset.
    
    Args:
        citation_file: Path to the citation data CSV file
        arxiv_file: Path to the arxiv papers CSV file
        output_file: Path to save the filtered citation data
    """
    print("Starting citation filtering process with direct version replacement...")
    
    # Read the CSV files
    citation_data = pd.read_csv(citation_file)
    arxiv_papers = pd.read_csv(arxiv_file)
    
    print(f"Citation data loaded: {len(citation_data)} rows")
    print(f"Arxiv papers loaded: {len(arxiv_papers)} rows")
    
    # List of normalized arxiv IDs (without version numbers)
    normalized_arxiv_ids = [normalize_id_for_comparison(paper_id) for paper_id in arxiv_papers['id']]
    
    # Dictionary mapping normalized IDs to versioned IDs
    arxiv_id_map = {normalize_id_for_comparison(row['id']): row['id'] 
                    for _, row in arxiv_papers.iterrows() 
                    if isinstance(row['id'], str)}
    
    print(f"Created map of {len(arxiv_id_map)} normalized arxiv IDs to versioned IDs")
    print("Sample mappings:")
    sample_count = min(3, len(arxiv_id_map))
    for i, (key, value) in enumerate(arxiv_id_map.items()):
        if i < sample_count:
            print(f"  {key} -> {value}")
    
    citation_data['normalized_cited_id'] = citation_data['cited_paper_id'].apply(normalize_id_for_comparison)
    
    # Filter rows where normalized_cited_id is in the normalized arxiv IDs set
    filtered_data = citation_data[citation_data['normalized_cited_id'].isin(normalized_arxiv_ids)].copy()
    
    print(f"Filtered citation data: {len(filtered_data)} rows match arxiv papers")
    
    for index, row in filtered_data.iterrows():
        norm_id = row['normalized_cited_id']
        if norm_id in arxiv_id_map:
            # Replace with the versioned ID from arxiv_papers
            filtered_data.at[index, 'cited_paper_id'] = arxiv_id_map[norm_id]
    
    # Convert any remaining https to http in cited_paper_id
    filtered_data['cited_paper_id'] = filtered_data['cited_paper_id'].apply(
        lambda x: x.replace('https://', 'http://') if isinstance(x, str) else x
    )
    
    filtered_data = filtered_data.drop(columns=['normalized_cited_id'])
    
    # Check that version numbers are present in the output
    version_pattern = r'v\d+$'
    rows_with_versions = filtered_data['cited_paper_id'].str.contains(version_pattern, regex=True, na=False)
    version_count = rows_with_versions.sum()
    
    print(f"Output rows with version numbers: {version_count} out of {len(filtered_data)}")
    
    filtered_data.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
    
    print("\nSample output rows:")
    print(filtered_data.head(3))

if __name__ == "__main__":
    # Input and output files
    citation_file = "citation_evaluation/data_for_eval/citation_data.csv"
    arxiv_file = "arxiv_papers_cleaned.csv"
    output_file = "citation_evaluation/data_for_eval/filtered_citations.csv"
    
    get_direct_version_replacement(citation_file, arxiv_file, output_file)