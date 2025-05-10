import os
import pandas as pd
from pathlib import Path
import logging
from .clustering.topic_modeler import TopicModeler
from .label_creator import LabelCreatorTogether

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_clustering_pipeline(data_path, output_dir):
    logger.info("Loading research papers data...")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} papers")

    required_columns = ['title', 'abstract', 'cleaned_abstract']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False

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
    modeler.visualize_clusters(
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

    df['cluster'] = modeler.clusters
    df.to_csv(output_dir / "papers_with_clusters.csv", index=False)

    logger.info(f"Saved clustering output to: {output_dir}")
    return True

def run_labeling_pipeline(cluster_path, papers_path, taxonomy_path, api_key):
    if not cluster_path.exists() or not papers_path.exists():
        logger.error("Required input files not found for labeling step.")
        return

    logger.info("Initializing Label Creator...")
    label_creator = LabelCreatorTogether(
        input_cluster_csv=str(cluster_path),
        input_papers_csv=str(papers_path),
        output_taxonomy_path=str(taxonomy_path), 
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        api_key=api_key,
    )

    logger.info("Running label generation and merging into original papers file...")
    label_creator.run()
    logger.info("Finished labeling.")

def main():
    output_dir = Path("clustering_results")
    output_dir.mkdir(exist_ok=True)
    data_path = "arxiv_papers_cleaned.csv"
    cluster_path = output_dir / "cluster_summary.csv"
    papers_path = output_dir / "papers_with_clusters.csv"
    taxonomy_path = Path("constants") / "taxonomy_tree.py"
    api_key = os.environ.get("TOGETHER_API_KEY")

    if run_clustering_pipeline(data_path, output_dir):
        run_labeling_pipeline(cluster_path, papers_path, taxonomy_path, api_key)

if __name__ == "__main__":
    main()
