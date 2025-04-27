import pandas as pd
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
from typing import Optional

class ClusterNameGenerator:
    def __init__(self, 
                 input_csv_path: str,
                 output_csv_path: str,
                 api_key: Optional[str] = None,
                 model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 temperature: float = 0.5,
                 retries: int = 3,
                 retry_delay: float = 2.0):
        """
        Initialize the LabelCreator using Together.ai API.

        Args:
            input_csv_path: Path to input CSV with cluster keywords.
            output_csv_path: Path to save labeled output.
            api_key: Together.ai API key.
            model: Together.ai model to use.
            temperature: Sampling temperature.
            retries: Number of retries if request fails.
            retry_delay: Delay (seconds) between retries.
        """
        self.input_csv_path = Path(input_csv_path)
        self.output_csv_path = Path(output_csv_path)
        self.model = model
        self.temperature = temperature
        self.retries = retries
        self.retry_delay = retry_delay

        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if self.api_key is None:
            raise ValueError("Together.ai API key not provided or found in environment.")
        
        self.df = None

    def load_data(self):
        if not self.input_csv_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_csv_path}")

        self.df = pd.read_csv(self.input_csv_path)
        print(f"Loaded {len(self.df)} clusters from {self.input_csv_path}")

    def simple_fallback_label(self, keywords: str, top_n: int = 3) -> str:
        """
        Create a simple label by joining top keywords if model fails.

        Args:
            keywords: Comma-separated keywords string.
            top_n: Number of keywords to use.
        
        Returns:
            Simple human-readable label.
        """
        keywords_list = [k.strip() for k in keywords.split(",")]
        selected_keywords = keywords_list[:top_n]
        return " ".join(selected_keywords).title()

    def generate_label_from_keywords(self, keywords: str) -> str:
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
                f"Suggest a concise, human-readable research topic name (no more than 4 words). "
                f"Respond ONLY with the topic name. DO NOT provide any explanation, notes, or extra text."
            ),
            "max_tokens": 30,
            "temperature": self.temperature
        }

        attempt = 0
        while attempt < self.retries:
            try:
                response = requests.post(url, headers=headers, json=data)
                result = response.json()
                if "output" in result and "choices" in result["output"]:
                    raw_text = result["output"]["choices"][0]["text"].strip()

                    cleaned_label = raw_text.split("\n")[0] 
                    cleaned_label = cleaned_label.split("---")[0] 
                    cleaned_label = cleaned_label.split("Explanation")[0] 
                    cleaned_label = cleaned_label.strip()

                    if cleaned_label:
                        return cleaned_label
                else:
                    print(f"Warning: Unexpected API format. Retrying... (Attempt {attempt+1})")
            except Exception as e:
                print(f"Error: {e}. Retrying... (Attempt {attempt+1})")
            
            attempt += 1
            time.sleep(self.retry_delay)

        print(f"Failed to generate label for keywords: {keywords}. Using fallback.")
        return self.simple_fallback_label(keywords)

    def generate_all_labels(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Generating labels for clusters...")
        generated_labels = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            keywords = row["keywords"]
            label = self.generate_label_from_keywords(keywords)
            generated_labels.append(label)

        self.df["generated_label"] = generated_labels

    def save_output(self):
        self.df.to_csv(self.output_csv_path, index=False)
        print(f"Saved labeled clusters to {self.output_csv_path}")

    def run(self):
        self.load_data()
        self.generate_all_labels()
        self.save_output()
