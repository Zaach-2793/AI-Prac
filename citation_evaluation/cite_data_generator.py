import csv
import requests
import time
import pandas as pd
import xml.etree.ElementTree as ET
import re
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import logging
from queue import Queue
import random

class CitationGenerator:
    """
    Citation generator, records only citations that have ArXiv IDs
    """
    
    def __init__(self, concurrency=5, rate_limit=20, min_delay=0.5):
        self.arxiv_api_url = "http://export.arxiv.org/api/query"
        self.s2_api_url = "https://api.semanticscholar.org/v1/paper/arXiv:"
        self.s2_graph_api_url = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"
        
        self.concurrency = concurrency  # Number of concurrent API requests
        self.rate_limit = rate_limit  # Maximum requests per minute
        self.min_delay = min_delay  # Minimum delay between requests
        
        # Rate limiting
        self.request_times = []
        self.request_count_lock = threading.Lock()
        
        # Cache for API responses
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.log_file = f"citation_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('citation_generator')
        self.logger.info(f"Initialized real-time citation generator with concurrency={concurrency}, rate_limit={rate_limit}")
        
        # Set up namespaces for ArXiv XML parsing
        self.namespaces = {
            'arxiv': 'http://arxiv.org/schemas/atom',
            'atom': 'http://www.w3.org/2005/Atom'
        }
        
        # Results collection
        self.results_lock = threading.Lock()
        self.citations_count = 0
        
        self.stats = {
            "papers_processed": 0,
            "citations_found": 0,
            "api_requests": 0,
            "api_errors": 0,
            "cache_hits": 0,
            "retry_attempts": 0
        }
        self.stats_lock = threading.Lock()
        
        # Process tracking
        self.paper_index_map = {}  # Maps arxiv_id to its index in the processing order
        self.total_papers = 0
        
        # Writer thread control
        self.active = True
    
    def update_stat(self, key, increment=1):
        """Thread-safe update of a statistic value"""
        with self.stats_lock:
            self.stats[key] = self.stats[key] + increment
    
    def extract_arxiv_id(self, url_or_id: str) -> str:
        """
        Extract ArXiv ID from a URL or ID string.
        
        Args:
            url_or_id: ArXiv URL or ID
            
        Returns:
            Clean ArXiv ID
        """
        if not url_or_id:
            return ""
            
        # Check if it's a URL
        if url_or_id.startswith('http'):
            # Extract the ID pattern from the URL
            match = re.search(r'arxiv.org/(?:abs|pdf)/([^/\s]+(?:/\d+)?)', url_or_id)
            if match:
                raw_id = match.group(1)
                # Remove version number if originally present
                return re.sub(r'v\d+$', '', raw_id)
        
        # If it's already an ID, not a URL
        elif re.match(r'^\d+\.\d+', url_or_id) or '/' in url_or_id:
            return re.sub(r'v\d+$', '', url_or_id.strip())
            
        # Return the original if we couldn't extract an ID
        return url_or_id
    
    def is_valid_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Check if a string is a valid ArXiv ID.
        
        Args:
            arxiv_id: String to check
            
        Returns:
            True if valid ArXiv ID, False otherwise
        """
        # Check for new format: NNNN.NNNNN
        if re.match(r'^\d+\.\d+$', arxiv_id):
            return True
            
        # Check for old format: category/YYMMNNN
        if re.match(r'^[a-z-]+(?:\.[a-z-]+)?/\d{4,7}$', arxiv_id):
            return True
            
        return False
    
    def check_rate_limit(self) -> float:
        """
        Check if we're approaching the rate limit and calculate delay if needed.
        
        Returns:
            Number of seconds to delay before next request (0 if no delay needed)
        """
        with self.request_count_lock:
            current_time = time.time()
            
            # Remove request times older than 1 minute
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # If we're below the rate limit, no delay needed
            if len(self.request_times) < self.rate_limit:
                self.request_times.append(current_time)
                return self.min_delay
            
            # Calculate when we can make the next request
            oldest_request = min(self.request_times)
            wait_time = max(60 - (current_time - oldest_request), self.min_delay)
            
            # Update the oldest request time to current time + wait
            self.request_times.remove(oldest_request)
            self.request_times.append(current_time + wait_time)
            
            return wait_time + self.min_delay
    
    def get_references_from_s2(self, arxiv_id: str, retry_attempts=3) -> List[Dict]:
        """
        Get reference data from Semantic Scholar API with adaptive rate limiting and retries.
        
        Args:
            arxiv_id: The ArXiv ID
            retry_attempts: Number of retry attempts if request fails
            
        Returns:
            List of reference data dictionaries
        """
        # Check cache first
        cache_key = arxiv_id.replace('/', '_')
        cache_file = os.path.join(self.cache_dir, f"s2_full_{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if cached_data and 'references' in cached_data:
                        self.update_stat("cache_hits")
                        return cached_data['references']
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Error reading cache for {arxiv_id}: {e}")
                try:
                    os.rename(cache_file, cache_file + '.corrupted')
                except:
                    pass
        
        # Try multiple API endpoints with retries
        for attempt in range(retry_attempts + 1):
            # On later attempts, try alternative API
            use_graph_api = (attempt > 1)
            
            try:
                # Determine which API to use
                if use_graph_api:
                    url = f"{self.s2_graph_api_url}{arxiv_id}?fields=references.externalIds,references.paperId,references.title,references.url,references.doi"
                    self.logger.info(f"Trying Graph API for {arxiv_id} (attempt {attempt})")
                else:
                    url = f"{self.s2_api_url}{arxiv_id}"
                    self.logger.info(f"Fetching references from S2 for {arxiv_id} (attempt {attempt})")
                
                # Check rate limit and wait if necessary
                wait_time = self.check_rate_limit()
                if wait_time > self.min_delay:
                    self.logger.info(f"Rate limit approaching, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                
                # Make the request
                headers = {
                    'User-Agent': f'Citation Generator/1.0 (Research Project; max-concurrency={self.concurrency})',
                    'Accept': 'application/json'
                }
                
                self.update_stat("api_requests")
                response = requests.get(url, headers=headers, timeout=30)
                
                # Handle different response statuses
                if response.status_code == 200:
                    # Success, process the response
                    data = response.json()
                    
                    if use_graph_api:
                        # Extract references from Graph API response
                        if 'references' in data:
                            references = data['references']
                            self.logger.info(f"Found {len(references)} references via Graph API for {arxiv_id}")
                            
                            # Cache the result
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump({'references': references}, f, indent=2)
                            
                            return references
                        else:
                            self.logger.warning(f"No references field in Graph API response for {arxiv_id}")
                    else:
                        # Extract references from v1 API response
                        if 'references' in data:
                            references = data['references']
                            self.logger.info(f"Found {len(references)} references for {arxiv_id}")
                            
                            # Cache the result
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump({'references': references}, f, indent=2)
                            
                            return references
                        else:
                            self.logger.warning(f"No references field in response for {arxiv_id}")
                
                elif response.status_code == 404:
                    self.logger.warning(f"Paper not found in Semantic Scholar: {arxiv_id}")
                    # Cache empty result
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump({'references': []}, f, indent=2)
                    return []
                
                elif response.status_code in (429, 409, 503):
                    self.update_stat("retry_attempts")
                    wait_time = min(30 * (2 ** attempt), 300)
                    self.logger.warning(f"Rate limit exceeded for {arxiv_id}. Status: {response.status_code}. Waiting {wait_time}s before retry.")
                    time.sleep(wait_time)
                    continue
                
                else:
                    self.logger.error(f"Unexpected status code: {response.status_code} for {arxiv_id}")
                    self.update_stat("api_errors")
                    # If not the last attempt, try again with exponential backoff
                    if attempt < retry_attempts:
                        wait_time = min(10 * (2 ** attempt), 120)
                        time.sleep(wait_time)
                        continue
            
            except Exception as e:
                self.logger.error(f"Error fetching references for {arxiv_id}: {str(e)}")
                self.update_stat("api_errors")
                # If not the last attempt, try again
                if attempt < retry_attempts:
                    wait_time = min(5 * (2 ** attempt), 60)
                    time.sleep(wait_time)
                    continue
        
        # If we reach here, all attempts failed
        # Cache empty result to avoid retrying immediately
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'references': []}, f, indent=2)
        
        return []
    
    def process_paper(self, paper_url: str, arxiv_id: str, writer_queue: Queue):
        """
        Process a single paper to get its citations.
        
        Args:
            paper_url: Original URL of the paper
            arxiv_id: ArXiv ID of the paper
            writer_queue: Queue to write citation results
        """
        try:
            # Get paper index for progress reporting
            paper_idx = self.paper_index_map.get(arxiv_id, -1)
            if paper_idx >= 0:
                self.logger.info(f"Processing paper {paper_idx+1}/{self.total_papers}: {arxiv_id}")
            
            # Get references from Semantic Scholar
            references = self.get_references_from_s2(arxiv_id)
            paper_citations = []
            
            # Process each reference
            for ref in references:
                # Try to extract ArXiv ID from reference
                ref_arxiv_id = None
                
                # Check for ArXiv ID in externalIds (Graph API format)
                if 'externalIds' in ref and ref['externalIds']:
                    external_ids = ref['externalIds']
                    if 'ArXiv' in external_ids:
                        ref_arxiv_id = external_ids['ArXiv']
                
                # Check for ArXiv ID in arxivId (v1 API format)
                elif 'arxivId' in ref and ref['arxivId']:
                    ref_arxiv_id = ref['arxivId']
                
                # If we found an ArXiv ID, add the citation
                if ref_arxiv_id:
                    cited_paper_url = f"https://arxiv.org/abs/{ref_arxiv_id}"
                    paper_citations.append((paper_url, cited_paper_url))
            
            if paper_citations:
                writer_queue.put(paper_citations)
                
            # Update statistics
            with self.results_lock:
                self.citations_count += len(paper_citations)
            
            self.update_stat("papers_processed")
            self.update_stat("citations_found", len(paper_citations))
            
            if paper_idx >= 0:
                self.logger.info(f"Paper {paper_idx+1}/{self.total_papers}: Found {len(paper_citations)} ArXiv citations for {arxiv_id}")
            else:
                self.logger.info(f"Found {len(paper_citations)} ArXiv citations for paper {arxiv_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing paper {arxiv_id}: {str(e)}")
    
    def writer_thread(self, output_csv: str, writer_queue: Queue):
        """
        Thread to handle writing citation data to the output file in real-time.
        
        Args:
            output_csv: Path to output CSV file
            writer_queue: Queue containing citation data to write
        """
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['paper_id', 'cited_paper_id'])
            csvfile.flush()  # Ensure header is written immediately
            
            flush_counter = 0
            
            while self.active or not writer_queue.empty():
                try:
                    # Get citations with a short timeout to remain responsive
                    citations = writer_queue.get(timeout=0.5)
                    
                    # Write citations to the file
                    for citing_url, cited_url in citations:
                        writer.writerow([citing_url, cited_url])
                    
                    # Increment flush counter
                    flush_counter += 1
                    
                    # Flush the file every few writes to ensure real-time updates
                    if flush_counter % 5 == 0:
                        csvfile.flush()
                    
                    writer_queue.task_done()
                    
                except Queue.Empty:
                    # Queue is empty, flush any pending writes and continue
                    csvfile.flush()
                    continue
    
    def generate_citations(self, input_csv: str, output_csv: str):
        """
        Generate citation data for papers with ArXiv IDs.
        
        Args:
            input_csv: Path to input CSV with paper data
            output_csv: Path to output CSV for citation data
        """
        try:
            # Read input CSV
            self.logger.info(f"Reading input CSV: {input_csv}")
            papers_df = pd.read_csv(input_csv)
            
            # Validate required columns
            if 'id' not in papers_df.columns:
                self.logger.error("Error: Input CSV must have an 'id' column")
                return
            
            # Create mappings
            url_to_arxiv_id = {}
            
            # Extract ArXiv IDs from URLs
            self.logger.info("Extracting ArXiv IDs from paper URLs...")
            for idx, row in papers_df.iterrows():
                paper_url = str(row['id'])
                arxiv_id = self.extract_arxiv_id(paper_url)
                
                if self.is_valid_arxiv_id(arxiv_id):
                    url_to_arxiv_id[paper_url] = arxiv_id
                    
            paper_items = list(url_to_arxiv_id.items())
            self.total_papers = len(paper_items)
            
            # Create index mapping for progress reporting
            for i, (_, arxiv_id) in enumerate(paper_items):
                self.paper_index_map[arxiv_id] = i
                
            self.logger.info(f"Found {self.total_papers} valid ArXiv papers to process")
            
            # Set up threading and processing
            writer_queue = Queue()
            self.active = True
            
            # Start writer thread for real-time updates
            writer_thread = threading.Thread(
                target=self.writer_thread,
                args=(output_csv, writer_queue)
            )
            writer_thread.daemon = True
            writer_thread.start()
            
            # Process papers with concurrent executor
            self.logger.info(f"Starting processing with concurrency={self.concurrency}")
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                # Submit all papers for processing
                futures = []
                for paper_url, arxiv_id in paper_items:
                    future = executor.submit(self.process_paper, paper_url, arxiv_id, writer_queue)
                    futures.append(future)
                
                # Wait for all tasks to complete
                for i, future in enumerate(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error in future {i}: {str(e)}")
                    
                    # Log progress every 100 papers
                    if (i + 1) % 100 == 0 or (i + 1) == self.total_papers:
                        self.logger.info(f"Processed {i + 1}/{self.total_papers} papers ({((i + 1) / self.total_papers) * 100:.1f}%)")
                        self.logger.info(f"Current stats: {json.dumps(self.stats)}")
            
            # Signal writer to finish and wait for completion
            self.active = False
            writer_queue.join()  # Wait for all writing tasks to complete
            
            self.logger.info(f"Completed processing {self.total_papers} papers")
            self.logger.info(f"Generated {self.citations_count} ArXiv citation relationships in {output_csv}")
            self.logger.info(f"Final statistics: {json.dumps(self.stats)}")
            
        except Exception as e:
            self.logger.error(f"Error generating citations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Ensure writer thread knows to terminate
            self.active = False

def main():
    print("Real-time ArXiv Citation Generator")
    print("---------------------------------")
    
    # Get concurrency setting
    try:
        concurrency = int(input("Enter concurrency level (1-10, default: 5): ") or "5")
        concurrency = max(1, min(10, concurrency))  # Limit to reasonable range
    except ValueError:
        print("Using default concurrency of 5")
        concurrency = 5
    
    # Get rate limit setting
    try:
        rate_limit = int(input("Enter rate limit per minute (10-120, default: 20): ") or "20")
        rate_limit = max(10, min(120, rate_limit))  # Limit to reasonable range
    except ValueError:
        print("Using default rate limit of 20 requests per minute")
        rate_limit = 20
    
    generator = CitationGenerator(
        concurrency=concurrency, 
        rate_limit=rate_limit
    )    
    input_csv = "arxiv_papers_cleaned.csv"
    output_csv = "citation_evaluation/data_for_eval/citation_data.csv"
    
    generator.generate_citations(input_csv, output_csv)
    
    print(f"Log has been saved to {generator.log_file}")

if __name__ == "__main__":
    main()