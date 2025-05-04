import unittest
import pandas as pd
import os
import json
from pathlib import Path

from citation_evaluation.base_evaluator import BaseCitationEvaluator
from citation_evaluation.coverage_evaluator import SubtopicCoverageEvaluator
from citation_evaluation.gcd_evaluator import GCDSubtopicEvaluator
from citation_evaluation.reporter import CitationEvaluationReporter


class TestCitationEvaluation(unittest.TestCase):
    """Test cases for citation evaluation modules."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all test cases."""
        # Create test directory
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Define hierarchy for testing (to test both evaluation approaches)
        cls.hierarchy = {
            0: {'0': ['paper1', 'paper2', 'paper3', 'paper4', 'paper5', 'paper6', 'paper7', 'paper8']},
            1: {
                '1.0': ['paper1', 'paper2', 'paper3', 'paper4'], 
                '1.1': ['paper5', 'paper6', 'paper7', 'paper8']
            },
            2: {
                '2.0': ['paper1', 'paper2'], 
                '2.1': ['paper3', 'paper4'], 
                '2.2': ['paper5', 'paper6'], 
                '2.3': ['paper7', 'paper8']
            }
        }
        
        # Create citation data with specific patterns to validate evaluation metrics
        cls.citation_data = pd.DataFrame([
            # Citations within finest clusters (level 2) - highest probability
            {'paper_id': 'paper1', 'cited_paper_id': 'paper2'},
            {'paper_id': 'paper2', 'cited_paper_id': 'paper1'},
            {'paper_id': 'paper3', 'cited_paper_id': 'paper4'},
            {'paper_id': 'paper4', 'cited_paper_id': 'paper3'},
            {'paper_id': 'paper5', 'cited_paper_id': 'paper6'},
            {'paper_id': 'paper6', 'cited_paper_id': 'paper5'},
            {'paper_id': 'paper7', 'cited_paper_id': 'paper8'},
            {'paper_id': 'paper8', 'cited_paper_id': 'paper7'},
            
            # Citations within level 1 clusters but across level 2 clusters - medium probability
            {'paper_id': 'paper1', 'cited_paper_id': 'paper3'},
            {'paper_id': 'paper3', 'cited_paper_id': 'paper1'},
            {'paper_id': 'paper2', 'cited_paper_id': 'paper4'},
            {'paper_id': 'paper5', 'cited_paper_id': 'paper7'},
            {'paper_id': 'paper7', 'cited_paper_id': 'paper5'},
            
            # Cross-cluster citations (across level 1) - low probability
            {'paper_id': 'paper1', 'cited_paper_id': 'paper5'},
            {'paper_id': 'paper8', 'cited_paper_id': 'paper4'}
        ])
        
        # Save test data to files
        cls.hierarchy_file = cls.test_dir / "test_hierarchy.json"
        cls.citation_file = cls.test_dir / "test_citations.csv"
        
        with open(cls.hierarchy_file, 'w') as f:
            json.dump(cls.hierarchy, f)
        
        cls.citation_data.to_csv(cls.citation_file, index=False)
    
    def test_base_evaluator_initialization(self):
        """Test initialization and core functionality of BaseCitationEvaluator."""
        evaluator = BaseCitationEvaluator(self.hierarchy, self.citation_data)
        
        # Check paper clusters mapping
        self.assertIn(0, evaluator.paper_clusters)
        self.assertIn(1, evaluator.paper_clusters)
        self.assertIn(2, evaluator.paper_clusters)
        
        # Check specific mappings
        self.assertEqual(evaluator.paper_clusters[2]['paper1'], '2.0')
        self.assertEqual(evaluator.paper_clusters[1]['paper3'], '1.0')
        
        # Check citation graph
        self.assertIn('paper1', evaluator.citations)
        self.assertIn('paper2', evaluator.citations['paper1'])
        self.assertEqual(len(evaluator.citations['paper1']), 3)  # paper1 cites paper2, paper3, paper5
        
        # Check get_all_papers
        all_papers = evaluator.get_all_papers()
        self.assertEqual(len(all_papers), 8)
        self.assertIn('paper1', all_papers)
        self.assertIn('paper8', all_papers)
    
    def test_subtopic_coverage_evaluation(self):
        """Test SubtopicCoverageEvaluator functionality."""
        evaluator = SubtopicCoverageEvaluator(self.hierarchy, self.citation_data)
        
        # Test evaluate_coverage
        coverage_df = evaluator.evaluate_coverage()
        
        # Verify structure
        self.assertIsInstance(coverage_df, pd.DataFrame)
        self.assertIn('paper_id', coverage_df.columns)
        self.assertIn('level', coverage_df.columns)
        self.assertIn('coverage', coverage_df.columns)
        
        # Check specific coverage patterns - get coverage for paper1 at different levels
        paper1_coverage = coverage_df[coverage_df['paper_id'] == 'paper1']
        paper1_by_level = paper1_coverage.groupby('level')['coverage'].mean()
        
        if len(paper1_by_level) >= 3:  # Check coverage trend (should be higher at more specific levels)
            self.assertGreaterEqual(paper1_by_level[2], paper1_by_level[1])
            self.assertGreaterEqual(paper1_by_level[1], paper1_by_level[0])
        
        # Test analyze_coverage_trend
        trend = evaluator.analyze_coverage_trend(coverage_df)
        self.assertIn('coverage_by_level', trend)
        self.assertIn('is_decreasing', trend)
        
        # Test visualization
        fig = evaluator.visualize_coverage_trend(coverage_df)
        self.assertIsNotNone(fig)
    
    def test_gcd_evaluator(self):
        """Test GCDSubtopicEvaluator functionality."""
        evaluator = GCDSubtopicEvaluator(self.hierarchy, self.citation_data)
        
        # Test find_gcd_subtopic function
        gcd_level = evaluator.find_gcd_subtopic('paper1', 'paper2')
        self.assertEqual(gcd_level, 2)  # Should be most specific level
        
        gcd_level = evaluator.find_gcd_subtopic('paper1', 'paper3')
        self.assertEqual(gcd_level, 1)  # One level up
        
        gcd_level = evaluator.find_gcd_subtopic('paper1', 'paper8')
        self.assertEqual(gcd_level, 0)  # Root level only
        
        # Test evaluate_gcd
        gcd_df = evaluator.evaluate_gcd(n_random_pairs=10)
        
        # Verify structure
        self.assertIsInstance(gcd_df, pd.DataFrame)
        self.assertIn('paper1', gcd_df.columns)
        self.assertIn('paper2', gcd_df.columns)
        self.assertIn('is_citation', gcd_df.columns)
        self.assertIn('gcd_level', gcd_df.columns)
        
        # Test analyze_gcd_results
        analysis = evaluator.analyze_gcd_results(gcd_df)
        self.assertIn('citation_gcd_mean', analysis)
        self.assertIn('random_gcd_mean', analysis)
        self.assertIn('gcd_difference', analysis)
        
        # Test visualization
        fig = evaluator.visualize_gcd_comparison(gcd_df)
        self.assertIsNotNone(fig)
    
    def test_reporter(self):
        """Test CitationEvaluationReporter functionality."""
        reporter = CitationEvaluationReporter(self.hierarchy, self.citation_data)
        
        # Test report generation
        report_output_dir = self.test_dir / "report_test"
        report = reporter.generate_full_report(
            output_path=str(report_output_dir),
            sample_size=8,  # Use all papers
            n_random_pairs=10
        )
        
        # Check report structure
        self.assertIn('coverage_analysis', report)
        self.assertIn('gcd_analysis', report)
        self.assertIn('overall_quality', report)
        
        # Check output files
        self.assertTrue(os.path.exists(report_output_dir / "subtopic_coverage.csv"))
        self.assertTrue(os.path.exists(report_output_dir / "gcd_subtopics.csv"))
        self.assertTrue(os.path.exists(report_output_dir / "coverage_trend.png"))
        self.assertTrue(os.path.exists(report_output_dir / "gcd_comparison.png"))
        self.assertTrue(os.path.exists(report_output_dir / "evaluation_summary.txt"))


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_dir = Path("test_utils_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Simple test hierarchy
        self.hierarchy = {
            0: {'0': ['paper1', 'paper2', 'paper3']},
            1: {'1.0': ['paper1', 'paper2'], '1.1': ['paper3']}
        }
        
        # Simple test citations
        self.citation_data = pd.DataFrame([
            {'paper_id': 'paper1', 'cited_paper_id': 'paper2'},
            {'paper_id': 'paper2', 'cited_paper_id': 'paper3'}
        ])
        
        # Save test data
        self.hierarchy_file = self.test_dir / "test_hierarchy.json"
        self.citation_file = self.test_dir / "test_citations.csv"
        
        with open(self.hierarchy_file, 'w') as f:
            json.dump(self.hierarchy, f)
        
        self.citation_data.to_csv(self.citation_file, index=False)
    
    def test_load_functions(self):
        """Test data loading functions."""
        from citation_evaluation.utils import load_citation_data, load_hierarchy
        
        loaded_citations = load_citation_data(str(self.citation_file))
        self.assertEqual(len(loaded_citations), 2)
        
        loaded_hierarchy = load_hierarchy(str(self.hierarchy_file))
        self.assertEqual(len(loaded_hierarchy), 2)


if __name__ == "__main__":
    unittest.main()