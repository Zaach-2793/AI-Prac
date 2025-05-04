import argparse
import pandas as pd
from topic_modeler import TopicModeler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load research paper data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the research papers
    """
    logger.info(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def main():
    parser = argparse.ArgumentParser(description="Research Paper Topic Modeling Tool")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file containing research papers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Target number of clusters"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum size of clusters"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples in neighborhood for core points"
    )
    parser.add_argument(
        "--top-keywords",
        type=int,
        default=5,
        help="Number of keywords to extract per cluster"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    df = load_data(args.input)
    
    modeler = TopicModeler(
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )
    
    modeler.fit_transform(df['cleaned_abstract'].tolist())
    
    modeler.extract_cluster_keywords(
        df['cleaned_abstract'].tolist(),
        top_n=args.top_keywords
    )
    
    fig = modeler.visualize_clusters(
        titles=df['title'].tolist(),
        save_path=str(output_dir / "clusters.html")
    )
    
    summary = modeler.get_cluster_summary()
    
    summary_df = pd.DataFrame([
        {
            'cluster_id': cluster_id,
            'size': info['size'],
            'keywords': ', '.join(info['keywords'])
        }
        for cluster_id, info in summary.items()
    ])
    summary_df.to_csv(output_dir / "cluster_summary.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Found {len(summary)} clusters")
    
    print("\nCluster Summary:")
    print("===============")
    for cluster_id, info in summary.items():
        print(f"\nCluster {cluster_id} (Size: {info['size']})")
        print(f"Keywords: {', '.join(info['keywords'])}")

if __name__ == "__main__":
    main() 