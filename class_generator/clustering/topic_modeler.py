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
        random_state: int = 42
    ):
        """
        Initialize the TopicModeler with clustering parameters.
        
        Args:
            n_clusters: Target number of clusters (approximate)
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood for core points
            n_components: Number of components for dimensionality reduction
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_components = n_components
        self.random_state = random_state
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        self.embeddings = None
        self.clusters = None
        self.feature_names = None
        self.cluster_keywords = {}
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Process the texts and return the reduced embeddings.
        
        Args:
            texts: List of text documents to process
            
        Returns:
            Reduced embeddings as numpy array
        """
        logger.info("Vectorizing texts...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info("Reducing dimensionality...")
        self.embeddings = self.reducer.fit_transform(tfidf_matrix)
        
        logger.info("Clustering documents...")
        self.clusters = self.clusterer.fit_predict(self.embeddings)
        
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