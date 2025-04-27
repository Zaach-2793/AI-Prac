import pandas as pd
from typing import Dict, List

def load_hierarchy(file_path: str) -> Dict[int, Dict[str, List[str]]]:
    """
    Load hierarchical clustering from a JSON file.
    
    Expected format: {
        "0": {"0": ["paper1", "paper2", ...]},
        "1": {"1.0": ["paper1", ...], "1.1": [...]}
    }
    
    Args:
        file_path: Path to hierarchy JSON file
        
    Returns:
        Dictionary representation of hierarchy
    """
    import json
    
    with open(file_path, 'r') as f:
        raw_hierarchy = json.load(f)
    
    # Convert string keys to integers for level keys
    hierarchy = {int(level): clusters for level, clusters in raw_hierarchy.items()}
    
    return hierarchy

def load_citation_data(file_path: str) -> pd.DataFrame:
    """
    Load citation data from CSV file.
    
    Expected format: paper_id, cited_paper_id
    
    Args:
        file_path: Path to citation data CSV
        
    Returns:
        DataFrame with citation relationships
    """
    return pd.read_csv(file_path)