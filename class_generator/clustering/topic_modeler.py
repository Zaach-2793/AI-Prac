import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeler:
    def __init__(
        self,
        n_clusters: int = 10,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        n_components: int = 2,
        random_state: int = 42,
        max_features: int = 1000
    ):
        """
        Initialize the TopicModeler with clustering parameters.
        
        Args:
            n_clusters: Target number of clusters (approximate)
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood for core points
            n_components: Number of components for dimensionality reduction
            random_state: Random state for reproducibility
            max_features: Maximum number of features for TF-IDF vectorization
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_components = n_components
        self.random_state = random_state
        
        # Adjust max_features based on dataset size
        self.max_features = max_features
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=max(5, min(15, min_cluster_size))  # Adjust n_neighbors based on min_cluster_size
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True  # enables soft prediction
        )

        self.embeddings = None
        self.clusters = None
        self.feature_names = None
        self.cluster_keywords = {}
        
    def fit_transform(self, texts: List[str], reassign_noise: bool = True, confidence_threshold: float = 0.1) -> np.ndarray:
        """
        Process texts, reduce dimensions, cluster, and optionally reassign noise points.

        Args:
            texts: List of text documents to process
            reassign_noise: Whether to reassign documents originally labeled -1
            confidence_threshold: Soft assignment confidence threshold (0-1)

        Returns:
            Reduced 2D embeddings
        """
        if len(texts) < self.min_cluster_size:
            logger.warning(f"Dataset size ({len(texts)}) is smaller than min_cluster_size ({self.min_cluster_size}). Adjusting parameters.")
            self.min_cluster_size = max(2, len(texts) // 5)  # At least 2, or 20% of dataset
            self.min_samples = max(1, self.min_cluster_size // 2)  # At least 1
            
            # Recreate clusterer with new parameters
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            logger.info(f"Adjusted clustering parameters: min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")

        # Also adjust max_features if the corpus is small
        adjusted_max_features = min(self.max_features, max(100, len(texts) * 10))  # 10 features per document, at least 100
        if adjusted_max_features != self.max_features:
            logger.info(f"Adjusting max_features from {self.max_features} to {adjusted_max_features} for small corpus")
            self.vectorizer.max_features = adjusted_max_features

        logger.info("Vectorizing texts...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        logger.info("Reducing dimensionality with UMAP...")
        self.embeddings = self.reducer.fit_transform(tfidf_matrix)

        # First try with HDBSCAN
        logger.info("Fitting HDBSCAN clustering...")
        success = False
        attempts = 0
        max_attempts = 3
        fallback_to_kmeans = False
        
        while not success and attempts < max_attempts:
            try:
                # Try progressively more relaxed parameters
                if attempts > 0:
                    new_min_cluster_size = max(2, self.min_cluster_size // (attempts + 1))
                    new_min_samples = max(1, self.min_samples // (attempts + 1))
                    logger.info(f"Retry attempt {attempts} with min_cluster_size={new_min_cluster_size}, min_samples={new_min_samples}")
                    
                    self.clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=new_min_cluster_size,
                        min_samples=new_min_samples,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True
                    )
                
                self.clusterer.fit(self.embeddings)  
                raw_clusters = self.clusterer.labels_
                
                # Check if all points are classified as noise
                if np.all(raw_clusters == -1):
                    if attempts == max_attempts - 1:
                        logger.warning("All points classified as noise after maximum attempts. Falling back to K-means.")
                        fallback_to_kmeans = True
                    else:
                        logger.warning(f"All points classified as noise in attempt {attempts+1}. Relaxing parameters further.")
                        attempts += 1
                        continue
                else:
                    # Found at least some non-noise clusters
                    success = True
                    logger.info(f"Found {len(set(raw_clusters)) - (1 if -1 in raw_clusters else 0)} clusters with {np.sum(raw_clusters == -1)} noise points.")
            
            except Exception as e:
                logger.error(f"Error in HDBSCAN clustering: {e}")
                if attempts == max_attempts - 1:
                    logger.warning("Failed after maximum attempts. Falling back to K-means.")
                    fallback_to_kmeans = True
                else:
                    logger.warning(f"Failed in attempt {attempts+1}. Retrying with relaxed parameters.")
                    attempts += 1
                    continue
        
            if fallback_to_kmeans:
                # Fall back to K-means which will always assign clusters
                from sklearn.cluster import KMeans
                
                # Determine a reasonable number of clusters based on data size
                n_kmeans_clusters = max(2, min(10, len(texts) // 20))  # 1 cluster per ~20 documents, between 2-10
                logger.info(f"Using K-means fallback with {n_kmeans_clusters} clusters")
                
                kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=self.random_state)
                raw_clusters = kmeans.fit_predict(self.embeddings)
                
                # K-means doesn't produce noise points (-1), all points are assigned to a cluster
                logger.info(f"K-means completed with {len(set(raw_clusters))} clusters and no noise points.")
                break
        
        # At this point, we either have HDBSCAN clusters with possible noise points,
        # or K-means clusters with no noise points
        if not fallback_to_kmeans and reassign_noise and np.any(raw_clusters == -1):
            logger.info("Reassigning noise points using soft prediction...")
            from hdbscan import prediction

            # Identify which points are noise
            noise_mask = raw_clusters == -1
            non_noise_mask = ~noise_mask
            
            # Only attempt reassignment if we have non-noise points to learn from
            if np.any(non_noise_mask):
                # Build new clusterer trained ONLY on confident points
                clean_embeddings = self.embeddings[non_noise_mask]
                self.clusterer.fit(clean_embeddings)

                # Predict cluster labels for all points (including old -1s)
                predicted_labels, strengths = prediction.approximate_predict(self.clusterer, self.embeddings)

                # Replace only raw -1s with predicted if confident
                reassigned = 0
                self.clusters = []
                for raw, pred, conf in zip(raw_clusters, predicted_labels, strengths):
                    if raw == -1 and pred != -1 and conf > confidence_threshold:
                        self.clusters.append(pred)
                        reassigned += 1
                    else:
                        self.clusters.append(raw)

                self.clusters = np.array(self.clusters)
                logger.info(f"Reassigned {reassigned} noisy points out of {(raw_clusters == -1).sum()} total noise.")
            else:
                logger.warning("All points were classified as noise. No reassignment possible.")
                self.clusters = raw_clusters
        else:
            self.clusters = raw_clusters

        # Add post-processing to merge small clusters
        # Set a higher threshold for small clusters
        min_cluster_threshold = max(10, int(len(texts) * 0.05))  # At least 5% of the dataset
        if len(set(self.clusters)) > 2:  # Only if we have more than one real cluster
            self.merge_small_clusters(min_size_threshold=min_cluster_threshold, similarity_threshold=0.4)

        return self.embeddings

    def extract_cluster_keywords(self, texts: List[str], top_n: int = 5) -> Dict[int, List[str]]:
        """
        Extract representative keywords for each cluster.
        
        Args:
            texts: List of text documents
            top_n: Number of keywords to extract per cluster
            
        Returns:
            Dictionary mapping cluster IDs to lists of keywords
        """
        if self.clusters is None:
            raise ValueError("Must fit the model before extracting keywords")
            
        logger.info("Extracting cluster keywords...")
        cluster_keywords = {}
        
        tfidf_matrix = self.vectorizer.transform(texts)
        
        for cluster_id in set(self.clusters):
            if cluster_id == -1:  
                continue
                
            cluster_mask = self.clusters == cluster_id
            cluster_docs = tfidf_matrix[cluster_mask]
            
            mean_tfidf = cluster_docs.mean(axis=0).A1
            
            top_indices = mean_tfidf.argsort()[-top_n:][::-1]
            keywords = [self.feature_names[i] for i in top_indices]
            
            cluster_keywords[cluster_id] = keywords
            
        self.cluster_keywords = cluster_keywords
        return cluster_keywords
    
    def merge_small_clusters(self, 
                         min_size_threshold: int = 10, 
                         similarity_threshold: float = 0.5) -> None:
        """
        Merge small clusters that are similar to reduce the total number of clusters.
        
        Args:
            min_size_threshold: Clusters smaller than this will be considered for merging
            similarity_threshold: Minimum cosine similarity for clusters to be merged
        """
        if self.clusters is None or self.embeddings is None:
            raise ValueError("Must fit the model before merging clusters")
            
        logger.info(f"Merging small clusters (threshold: {min_size_threshold})...")
        
        # Count cluster sizes and identify small clusters
        unique_clusters = list(set(self.clusters))
        if len(unique_clusters) <= 2:  # Just one real cluster and possibly noise
            logger.info("Too few clusters to merge. Skipping.")
            return
            
        cluster_sizes = {c: np.sum(self.clusters == c) for c in unique_clusters if c != -1}
        small_clusters = [c for c, size in cluster_sizes.items() if size < min_size_threshold]
        
        if not small_clusters:
            logger.info("No small clusters to merge. Skipping.")
            return
            
        logger.info(f"Found {len(small_clusters)} small clusters to potentially merge")
        
        # Calculate cluster centroids
        cluster_centroids = {}
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue
            mask = self.clusters == cluster_id
            cluster_centroids[cluster_id] = np.mean(self.embeddings[mask], axis=0)
        
        # Calculate similarities between small clusters and all other clusters
        merges = {}  # Small cluster -> target cluster to merge into
        
        for small_cluster in small_clusters:
            small_centroid = cluster_centroids[small_cluster]
            best_similarity = -1
            best_target = None
            
            # Find most similar larger cluster
            for target_cluster in unique_clusters:
                if target_cluster == -1 or target_cluster == small_cluster:
                    continue
                    
                # Don't merge small clusters into other small clusters
                if target_cluster in small_clusters:
                    continue
                    
                target_centroid = cluster_centroids[target_cluster]
                similarity = np.dot(small_centroid, target_centroid) / (
                    np.linalg.norm(small_centroid) * np.linalg.norm(target_centroid)
                )
                
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_target = target_cluster
            
            if best_target is not None:
                merges[small_cluster] = best_target
                logger.info(f"Will merge cluster {small_cluster} (size {cluster_sizes[small_cluster]}) "
                           f"into cluster {best_target} (size {cluster_sizes[best_target]}) "
                           f"with similarity {best_similarity:.3f}")
                
        # Apply the merges
        if merges:
            new_clusters = np.copy(self.clusters)
            for source, target in merges.items():
                new_clusters[new_clusters == source] = target
                # Also update keywords
                if source in self.cluster_keywords and target in self.cluster_keywords:
                    # Combine keywords, keep unique
                    combined = list(set(self.cluster_keywords[source] + self.cluster_keywords[target]))
                    self.cluster_keywords[target] = combined[:5]  # Keep top 5
                    del self.cluster_keywords[source]
                    
            self.clusters = new_clusters
            logger.info(f"Merged {len(merges)} small clusters. New cluster count: {len(set(self.clusters)) - (1 if -1 in self.clusters else 0)}")
        else:
            logger.info("No suitable merges found based on similarity threshold.")
    
    def visualize_clusters(
        self,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive visualization of the clusters.
        
        Args:
            titles: Optional list of document titles for hover text
            save_path: Optional path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if self.embeddings is None or self.clusters is None:
            raise ValueError("Must fit the model before visualization")
            
        logger.info("Creating visualization...")
        
        plot_data = pd.DataFrame(
            self.embeddings,
            columns=[f'Component_{i+1}' for i in range(self.n_components)]
        )
        plot_data['Cluster'] = self.clusters
        
        if titles:
            plot_data['Title'] = titles
            
        hover_text = []
        for idx, row in plot_data.iterrows():
            text = f"Cluster: {row['Cluster']}"
            if titles:
                text += f"<br>Title: {row['Title']}"
            if row['Cluster'] in self.cluster_keywords:
                text += f"<br>Keywords: {', '.join(self.cluster_keywords[row['Cluster']])}"
            hover_text.append(text)
            
        fig = px.scatter(
            plot_data,
            x='Component_1',
            y='Component_2',
            color='Cluster',
            hover_data=['Title'] if titles else None,
            title='Research Paper Clusters',
            labels={'Component_1': 'UMAP 1', 'Component_2': 'UMAP 2'}
        )
        
        fig.update_traces(hovertemplate='%{text}<extra></extra>', text=hover_text)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def get_cluster_summary(self) -> Dict[int, Dict]:
        """
        Get a summary of each cluster including size and keywords.
        
        Returns:
            Dictionary with cluster summaries
        """
        if self.clusters is None:
            raise ValueError("Must fit the model before getting cluster summary")
            
        summary = {}
        for cluster_id in set(self.clusters):
            if cluster_id == -1:  
                continue
                
            cluster_size = np.sum(self.clusters == cluster_id)
            keywords = self.cluster_keywords.get(cluster_id, [])
            
            summary[cluster_id] = {
                'size': cluster_size,
                'keywords': keywords
            }
            
        return summary