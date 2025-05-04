import pandas as pd
import os
from topic_modeler import TopicModeler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    output_dir = Path("clustering_results")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Loading research papers data...")
    data_path = "arxiv_papers_cleaned.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} papers")
    
    required_columns = ['title', 'abstract', 'cleaned_abstract']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    logger.info("Initializing topic modeler...")
    modeler = TopicModeler(
        n_clusters=15,  
        min_cluster_size=5,
        min_samples=3,
        n_components=2,
        random_state=42
    )
    
    logger.info("Processing papers and creating clusters...")
    modeler.fit_transform(df['cleaned_abstract'].tolist())
    
    logger.info("Extracting keywords for each cluster...")
    modeler.extract_cluster_keywords(df['cleaned_abstract'].tolist(), top_n=5)
    
    logger.info("Creating visualization...")
    fig = modeler.visualize_clusters(
        titles=df['title'].tolist(),
        save_path=str(output_dir / "paper_clusters.html")
    )
    
    logger.info("Generating cluster summary...")
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
    
    print("\nCluster Summary:")
    print("===============")
    for cluster_id, info in summary.items():
        print(f"\nCluster {cluster_id} (Size: {info['size']})")
        print(f"Keywords: {', '.join(info['keywords'])}")
    
    df['cluster'] = modeler.clusters
    df.to_csv(output_dir / "papers_with_clusters.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Found {len(summary)} clusters")
    logger.info(f"Visualization saved to {output_dir}/paper_clusters.html")
    logger.info(f"Cluster summary saved to {output_dir}/cluster_summary.csv")
    logger.info(f"Papers with cluster assignments saved to {output_dir}/papers_with_clusters.csv")

if __name__ == "__main__":
    main() 