import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from .clustering.topic_modeler import TopicModeler
from .label_creator import LabelCreatorTogetherEnhanced

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_clustering_by_category_subcategory(data_path, output_dir, target_cluster_count=248, min_group_threshold=5, use_kmeans_fallback=True):
    """
    Run the clustering pipeline, but perform clustering within each (category, subcategory) pair.
    
    Args:
        data_path: Path to the CSV file containing paper data
        output_dir: Directory to save outputs
        target_cluster_count: Approximate target for total number of clusters across all groups
        min_group_threshold: Minimum number of papers required to attempt clustering
        use_kmeans_fallback: Whether to use K-means when HDBSCAN fails
    """
    logger.info("Loading research papers data...")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} papers")

    required_columns = ['title', 'abstract', 'cleaned_abstract', 'category', 'subcategory']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Group papers by (category, subcategory)
    cat_subcat_groups = df.groupby(['category', 'subcategory'])
    logger.info(f"Found {len(cat_subcat_groups)} category-subcategory pairs")
    
    # Calculate a global scaling factor to achieve the target cluster count
    # Estimate an average of 2-3 clusters per group
    total_paper_count = len(df)
    total_group_count = len(cat_subcat_groups)
    avg_clusters_per_group = max(1, target_cluster_count / total_group_count)
    
    # Scale min_cluster_size to achieve target cluster count
    # For 6000 papers and 200 clusters, we'd want ~30 papers per cluster on average
    target_papers_per_cluster = total_paper_count / target_cluster_count
    logger.info(f"Targeting approximately {target_cluster_count} total clusters "
               f"with {target_papers_per_cluster:.1f} papers per cluster on average")
    
    # To store all results
    all_clustered_papers = []
    all_cluster_summaries = []
    
    # Track clustering statistics
    total_groups = 0
    successful_groups = 0
    kmeans_fallback_groups = 0
    skipped_groups = 0
    all_noise_groups = 0
    
    # Generate a global cluster ID that won't conflict across groups
    next_global_cluster_id = 0
    
    # Process each (category, subcategory) group
    for (cat, subcat), group_df in cat_subcat_groups:
        group_size = len(group_df)
        total_groups += 1
        logger.info(f"Processing {cat}/{subcat} with {group_size} papers")
        
        # Skip groups that are too small for meaningful clustering
        if group_size < min_group_threshold:
            logger.warning(f"Skipping {cat}/{subcat} - too few papers ({group_size})")
            # Assign these papers to cluster -1 (unclustered)
            group_df['cluster'] = -1
            all_clustered_papers.append(group_df)
            skipped_groups += 1
            continue
            
        # Initialize a topic modeler for this group
        # Calculate min_cluster_size based on target papers per cluster
        # Scale by group size relative to total papers
        scale_factor = (group_size / total_paper_count) * total_group_count
        target_clusters_for_group = max(1, min(5, avg_clusters_per_group * scale_factor))
        min_cluster_size = max(15, min(100, int(group_size / target_clusters_for_group)))
        
        logger.info(f"Targeting ~{target_clusters_for_group:.1f} clusters for this group with min_cluster_size={min_cluster_size}")
        
        modeler = TopicModeler(
            n_clusters=min(8, max(2, int(group_size / min_cluster_size))),
            min_cluster_size=min_cluster_size,
            min_samples=min(5, max(2, int(min_cluster_size / 3))),  # 1/3 of min_cluster_size, at least 2
            n_components=2,
            random_state=42
        )
        
        # Run clustering on this group
        modeler.fit_transform(group_df['cleaned_abstract'].tolist())
        
        # Extract keywords for clusters
        modeler.extract_cluster_keywords(group_df['cleaned_abstract'].tolist(), top_n=5)
        
        # Create visualization for this group
        group_output_dir = output_dir / f"{cat}_{subcat}"
        group_output_dir.mkdir(exist_ok=True)
        
        modeler.visualize_clusters(
            titles=group_df['title'].tolist(),
            save_path=str(group_output_dir / "paper_clusters.html")
        )
        
        # Get cluster summary for this group
        summary = modeler.get_cluster_summary()
        
        # Check how many real clusters we found
        unique_clusters = set(modeler.clusters)
        num_real_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        num_noise = np.sum(modeler.clusters == -1)
        
        if num_real_clusters == 0:
            # All points are noise
            logger.warning(f"All points in {cat}/{subcat} were classified as noise.")
            all_noise_groups += 1
            
            if use_kmeans_fallback and 'KMeans' in str(type(modeler.clusterer)):
                # We already used K-means as a fallback
                kmeans_fallback_groups += 1
                logger.info(f"K-means fallback was used for {cat}/{subcat}")
            else:
                # No clustering was successful
                group_df['cluster'] = -1
                all_clustered_papers.append(group_df)
                continue
        else:
            successful_groups += 1
        
        # Handle edge case: if all points are noise (summary is empty)
        if not summary:
            logger.warning(f"No clusters found in {cat}/{subcat}. All points will be treated as noise.")
            group_df['cluster'] = -1  # Assign all to noise
            all_clustered_papers.append(group_df)
            continue
        
        # Map local cluster IDs to global cluster IDs
        cluster_id_map = {-1: -1}  # Always include -1 mapping for noise points
        for local_id in summary.keys():
            if local_id == -1:  # Keep noise as -1 (already handled)
                continue
            else:
                cluster_id_map[local_id] = next_global_cluster_id
                next_global_cluster_id += 1
        
        # Update cluster summaries with global IDs and category/subcategory info
        for local_id, info in summary.items():
            global_id = cluster_id_map[local_id]
            if global_id == -1:
                continue  # Skip noise clusters in summary
                
            all_cluster_summaries.append({
                'cluster_id': global_id,
                'category': cat,
                'subcategory': subcat,
                'size': info['size'],
                'keywords': ', '.join(info['keywords'])
            })
        
        # Map the cluster IDs in the group dataframe
        group_df['cluster'] = [cluster_id_map[c] for c in modeler.clusters]
        
        # Add to our collection
        all_clustered_papers.append(group_df)
    
    # Combine all results
    result_df = pd.concat(all_clustered_papers, ignore_index=True)
    
    # Save the combined results
    summary_df = pd.DataFrame(all_cluster_summaries)
    summary_df.to_csv(output_dir / "cluster_summary.csv", index=False)
    
    result_df.to_csv(output_dir / "papers_with_clusters.csv", index=False)
    
    logger.info(f"Saved clustering output to: {output_dir}")
    logger.info(f"Created {next_global_cluster_id} clusters across all category-subcategory pairs")
    
    # Report clustering statistics
    logger.info(f"Clustering statistics:")
    logger.info(f"  Total groups: {total_groups}")
    logger.info(f"  Successful groups with HDBSCAN: {successful_groups}")
    logger.info(f"  Groups using K-means fallback: {kmeans_fallback_groups}")
    logger.info(f"  Groups with all points as noise: {all_noise_groups}")
    logger.info(f"  Skipped groups (too small): {skipped_groups}")
    
    return True

def run_labeling_pipeline(cluster_path, papers_path, taxonomy_path, api_key):
    """Run the labeling pipeline to generate fine-grained topic labels."""
    if not cluster_path.exists() or not papers_path.exists():
        logger.error("Required input files not found for labeling step.")
        return

    logger.info("Initializing Label Creator...")
    label_creator = LabelCreatorTogetherEnhanced(
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
    taxonomy_path = Path("constants") / "taxonomy_tree.json"
    api_key = "e9f81f80dcc23956c59b52785e7ceb2c50c66c519ce20ec61cf56272ee99b694"

    if run_clustering_by_category_subcategory(
        data_path, 
        output_dir, 
        target_cluster_count=360, 
        min_group_threshold=5, 
        use_kmeans_fallback=True
    ):
        run_labeling_pipeline(cluster_path, papers_path, taxonomy_path, api_key)

    from .label_analysis import TaxonomyVisualizer
    visualizer = TaxonomyVisualizer(
        papers_csv_path="clustering_results/papers_with_clusters.csv",
        taxonomy_json_path="constants/taxonomy_tree.json"
    )
    visualizer.run()

if __name__ == "__main__":
    main()