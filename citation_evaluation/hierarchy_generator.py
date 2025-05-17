import pandas as pd
import json
import os
from pathlib import Path

def gen_hierarchy(csv_path, output_path):
    """
    Generate a four-level hierarchical taxonomy from papers_with_clusters.csv
    in the format expected by the citation evaluation.
    
    Args:
        csv_path: Path to the papers_with_clusters.csv file
        output_path: Path to save the output JSON file
    """
    # Load the papers CSV
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required_columns = ['category', 'subcategory', 'cluster', 'id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        if 'title' in df.columns and 'id' not in df.columns:
            # Use title as id if id is missing
            df['id'] = df['title'].apply(lambda x: x.replace(' ', '_').lower()[:50])
            print("Using 'title' field to generate id values")
        else:
            print(f"Missing required columns: {missing_columns}")
            return None
    
    # Initialize four-level hierarchy structure
    hierarchy = {
        0: {},  # Root level (all papers)
        1: {},  # Category level
        2: {},  # Subcategory level
        3: {}   # Fine-grained cluster level
    }
    
    # Level 0: All papers grouped together
    all_papers = df['id'].tolist()
    hierarchy[0] = {'0': all_papers}
    
    # Level 1: Group by category
    for category in df['category'].unique():
        if pd.isna(category):
            continue
            
        papers_in_category = df[df['category'] == category]['id'].tolist()
        category_id = str(category).replace(' ', '_').lower()
        hierarchy[1][category_id] = papers_in_category
    
    # Level 2: Group by subcategory
    for subcategory in df['subcategory'].unique():
        if pd.isna(subcategory):
            continue
            
        papers_in_subcategory = df[df['subcategory'] == subcategory]['id'].tolist()
        subcategory_id = str(subcategory).replace(' ', '_').lower()
        hierarchy[2][subcategory_id] = papers_in_subcategory
    
    # Level 3: Group by fine-grained cluster
    for cluster in df['cluster'].unique():
        if pd.isna(cluster):
            continue
            
        papers_in_cluster = df[df['cluster'] == cluster]['id'].tolist()
        cluster_id = str(cluster)
        hierarchy[3][cluster_id] = papers_in_cluster
    
    # Save the hierarchy to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_categories = len(hierarchy[1])
    total_subcategories = len(hierarchy[2])
    total_clusters = len(hierarchy[3])
    
    print(f"Successfully created four-level hierarchy in the correct format")
    print(f"Statistics:")
    print(f"- Total papers: {len(all_papers)}")
    print(f"- Categories (Level 1): {total_categories}")
    print(f"- Subcategories (Level 2): {total_subcategories}")
    print(f"- Fine-grained clusters (Level 3): {total_clusters}")
    
    return hierarchy

if __name__ == "__main__":
    csv_path = "clustering_results/papers_with_clusters.csv"
    output_path = "citation_evaluation/data_for_eval/hierarchy.json"
    
    hierarchy = gen_hierarchy(csv_path, output_path)