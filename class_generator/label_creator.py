import pandas as pd
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from constants.constants import subcat_fullname_mapping

class LabelCreatorTogether:
    def __init__(self, 
                 input_cluster_csv: str,
                 input_papers_csv: str,
                 output_taxonomy_path: str,
                 api_key: Optional[str] = None,
                 model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 temperature: float = 0.5,
                 retries: int = 2,
                 retry_delay: float = 1.0):

        self.input_cluster_csv = Path(input_cluster_csv)
        self.input_papers_csv = Path(input_papers_csv)
        self.output_taxonomy_path = Path(output_taxonomy_path)
        self.model = model
        self.temperature = temperature
        self.retries = retries
        self.retry_delay = retry_delay

        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if self.api_key is None:
            raise ValueError("Together.ai API key not provided or found in environment.")

        self.clusters_df = None
        self.papers_df = None
        self.generated_cache = {}  # Cache to avoid duplicate API calls
        self.taxonomy = {}

    def load_data(self):
        """
        Load the cluster summary and papers CSV files into memory.
        Raises:
            FileNotFoundError: If any of the input files are missing.
        """
        if not self.input_cluster_csv.exists():
            raise FileNotFoundError(f"Cluster input file not found: {self.input_cluster_csv}")
        if not self.input_papers_csv.exists():
            raise FileNotFoundError(f"Papers input file not found: {self.input_papers_csv}")

        self.clusters_df = pd.read_csv(self.input_cluster_csv)
        self.papers_df = pd.read_csv(self.input_papers_csv)
        print(f"Loaded {len(self.clusters_df)} clusters and {len(self.papers_df)} papers.")

    def simple_fallback_label(self, keywords: str, top_n: int = 3) -> str:
        """
        Generate a basic fallback label using top N keywords.

        Args:
            keywords (str): Comma-separated keyword string.
            top_n (int): Number of top keywords to use in label.

        Returns:
            str: Capitalized label formed from top keywords.
        """
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        return " ".join(keywords_list[:top_n]).title() or "Uncategorized Topic"

    def generate_label_from_keywords(self, keywords: str) -> str:
        """
        Query the Together.ai API to generate a fine-grained topic label
        based on a string of keywords. Uses a cache and retry logic.

        Args:
            keywords (str): A comma-separated string of topic keywords.

        Returns:
            str: Generated label or a fallback label on failure.
        """
        if keywords in self.generated_cache:
            return self.generated_cache[keywords]

        url = "https://api.together.xyz/inference"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "prompt": (
                f"Given the following keywords extracted from scientific research papers:\n\n"
                f"{keywords}\n\n"
                f"Suggest ONLY a short human-readable research topic name (no more than 5 words). "
                f"Respond ONLY with the topic name. DO NOT provide any explanation, notes, or extra text."
            ),
            "max_tokens": 30,
            "temperature": self.temperature
        }

        for attempt in range(self.retries):
            try:
                response = requests.post(url, headers=headers, json=data)
                result = response.json()
                output = result.get("output", {})
                if isinstance(output, dict):
                    choices = output.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        raw_text = choices[0].get("text", "").strip()
                        cleaned = raw_text.split("\n")[0].split("---")[0].split("Explanation")[0].strip()
                        if cleaned:
                            self.generated_cache[keywords] = cleaned
                            return cleaned
                # print(f"Warning: Unexpected API format. Retrying... (Attempt {attempt+1})")
            except Exception as e:
                print(f"Error: {e}. Retrying... (Attempt {attempt+1})")
            time.sleep(self.retry_delay)

        fallback = self.simple_fallback_label(keywords)
        self.generated_cache[keywords] = fallback
        return fallback

    def generate_and_merge_labels(self):
        """
        Generate fine-grained labels for each cluster using Together.ai,
        merge them into the original papers dataset, and apply fallback
        using full subcategory names when needed.
        """
        print("Generating fine-grained labels for clusters...")
        fine_labels = []

        for idx, row in tqdm(self.clusters_df.iterrows(), total=len(self.clusters_df)):
            keywords = row.get("keywords", "")
            cluster_id = row.get("cluster_id", -1)

            if cluster_id == -1:
                fine_label = "NA"  # Placeholder, will fill in after merge
            else:
                fine_label = self.generate_label_from_keywords(keywords)

            fine_labels.append(fine_label)

        self.clusters_df["fine_topic_label"] = fine_labels

        print("Merging fine-grained labels into papers dataset...")
        merged = self.papers_df.drop(columns=["fine_topic_label"], errors="ignore").merge(
            self.clusters_df[["cluster_id", "fine_topic_label"]],
            how="left",
            left_on="cluster",
            right_on="cluster_id"
        )

        # Now fill in any remaining NaNs using subcat mapping
        merged["fine_topic_label"] = merged.apply(
            lambda row: subcat_fullname_mapping.get(row["subcategory"], row["subcategory"])
            if pd.isna(row["fine_topic_label"]) else row["fine_topic_label"], axis=1
        )

        merged.drop(columns=["cluster_id"], inplace=True)
        self.papers_df = merged

    def build_taxonomy_tree(self):
        """
        Construct a nested dictionary of the taxonomy tree:
        category → subcategory → list of fine-grained topics.
        """
        print("Building taxonomy tree from labeled data...")
        taxonomy = {}

        for _, row in self.papers_df.iterrows():
            category = row.get("category")
            subcategory = row.get("subcategory")
            fine_topic = row.get("fine_topic_label")

            if not category or not subcategory or not fine_topic:
                continue

            if category not in taxonomy:
                taxonomy[category] = {}
            if subcategory not in taxonomy[category]:
                taxonomy[category][subcategory] = []
            if fine_topic not in taxonomy[category][subcategory]:
                taxonomy[category][subcategory].append(fine_topic)

        self.taxonomy = taxonomy

    def save_output(self):
        self.papers_df.to_csv(self.input_papers_csv, index=False)
        print(f"Updated papers saved to {self.input_papers_csv}")

        named_clusters_path = self.input_cluster_csv.parent / "cluster_summary_named.csv"
        self.clusters_df.to_csv(named_clusters_path, index=False)
        print(f"Named clusters saved to {named_clusters_path}")
        
        with open(self.output_taxonomy_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated taxonomy tree\n")
            f.write("taxonomy = ")
            f.write(repr(self.taxonomy))
        print(f"Taxonomy tree saved to {self.output_taxonomy_path}")

    def run(self):
        """
        Execute the full pipeline: load data, generate labels,
        build taxonomy, and write all results to disk.
        """
        self.load_data()
        self.generate_and_merge_labels()
        self.build_taxonomy_tree()
        self.save_output()