import unittest
import numpy as np
import pandas as pd
from topic_modeler import TopicModeler

class TestTopicModeler(unittest.TestCase):
    def setUp(self):
        """Set up test data and model."""
        self.sample_texts = [
            "Machine learning algorithms for natural language processing",
            "Deep learning approaches in computer vision",
            "Neural networks for image recognition",
            "Statistical methods in data analysis",
            "Regression analysis and predictive modeling",
            "Time series forecasting with ARIMA",
            "Quantum computing applications",
            "Quantum entanglement and information",
            "Quantum cryptography protocols",
            "Classical mechanics principles",
            "Newton's laws of motion",
            "Einstein's theory of relativity"
        ]
        
        self.model = TopicModeler(
            n_clusters=3,
            min_cluster_size=2,
            min_samples=1,
            n_components=2
        )
        
    def test_fit_transform(self):
        """Test the fit_transform method."""
        embeddings = self.model.fit_transform(self.sample_texts)
        
        self.assertEqual(embeddings.shape[0], len(self.sample_texts))
        self.assertEqual(embeddings.shape[1], 2)
        
        self.assertIsNotNone(self.model.clusters)
        self.assertEqual(len(self.model.clusters), len(self.sample_texts))
        
    def test_extract_cluster_keywords(self):
        """Test keyword extraction."""
        self.model.fit_transform(self.sample_texts)
        
        keywords = self.model.extract_cluster_keywords(self.sample_texts, top_n=3)
        
        self.assertIsInstance(keywords, dict)
        self.assertTrue(len(keywords) > 0)
        
        for cluster_keywords in keywords.values():
            self.assertLessEqual(len(cluster_keywords), 3)
            
    def test_visualize_clusters(self):
        """Test cluster visualization."""
        self.model.fit_transform(self.sample_texts)
        self.model.extract_cluster_keywords(self.sample_texts)
        
        fig = self.model.visualize_clusters(titles=self.sample_texts)
        
        self.assertIsNotNone(fig)
        
    def test_get_cluster_summary(self):
        """Test cluster summary generation."""
        self.model.fit_transform(self.sample_texts)
        self.model.extract_cluster_keywords(self.sample_texts)
        
        summary = self.model.get_cluster_summary()
        
        self.assertIsInstance(summary, dict)
        for cluster_id, info in summary.items():
            self.assertIn('size', info)
            self.assertIn('keywords', info)
            self.assertIsInstance(info['size'], (int, np.integer))
            self.assertIsInstance(info['keywords'], list)
            
    def test_invalid_operations(self):
        """Test that invalid operations raise appropriate errors."""
        with self.assertRaises(ValueError):
            self.model.extract_cluster_keywords(self.sample_texts)
            
        with self.assertRaises(ValueError):
            self.model.visualize_clusters()
            
        with self.assertRaises(ValueError):
            self.model.get_cluster_summary()

if __name__ == '__main__':
    unittest.main() 