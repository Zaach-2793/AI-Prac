import pandas as pd
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict
from constants.constants_map import subcat_fullname_mapping, cat_fullname_mapping
from sentence_transformers import SentenceTransformer, util
import torch
import re
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelCreatorTogetherEnhanced:
    def __init__(self,
                 input_cluster_csv: str,
                 input_papers_csv: str,
                 output_taxonomy_path: str,
                 api_key: Optional[str] = None,
                 model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 temperature: float = 0.7,
                 retries: int = 3,
                 retry_delay: float = 1.0,
                 rerank_model: str = "all-MiniLM-L6-v2"):

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
        self.generated_cache = {}
        self.taxonomy = {}
        self.embedder = SentenceTransformer(rerank_model)
        
        # New: track labels and their quality
        self.label_scores = {}
        self.fallback_count = 0

    def load_data(self):
        """Load cluster and papers data, ensure category/subcategory information is present."""
        self.clusters_df = pd.read_csv(self.input_cluster_csv)
        self.papers_df = pd.read_csv(self.input_papers_csv)
        logger.info(f"Loaded {len(self.clusters_df)} clusters and {len(self.papers_df)} papers.")

        # Ensure the cluster summary has category and subcategory columns
        if 'category' not in self.clusters_df.columns:
            logger.info("Adding category and subcategory information to clusters...")
            # Get a representative paper for each cluster to extract category/subcategory
            cluster_info = {}
            for _, row in self.papers_df.iterrows():
                cluster_id = row.get('cluster')
                if cluster_id not in cluster_info and cluster_id != -1:
                    cluster_info[cluster_id] = {
                        'category': row.get('category'),
                        'subcategory': row.get('subcategory')
                    }
            
            # Add this information to the clusters dataframe
            self.clusters_df['category'] = self.clusters_df['cluster_id'].map(
                lambda x: cluster_info.get(x, {}).get('category', 'Unknown')
            )
            self.clusters_df['subcategory'] = self.clusters_df['cluster_id'].map(
                lambda x: cluster_info.get(x, {}).get('subcategory', 'Unknown')
            )

    @staticmethod
    def clean_label(text: str) -> str:
        """
        Clean and normalize a label, remove metadata and formatting issues.
        
        Args:
            text: Raw label text from LLM
            
        Returns:
            Cleaned, normalized label
        """
        # Remove patterns that look like metadata or instructions
        text = re.sub(r"Category\s+[A-Za-z\s]+\s+Subcategory\s+[A-Za-z\s]+\s+Keywords.*", "", text)
        text = re.sub(r"Label:", "", text)
        text = re.sub(r"Topic:", "", text)
        
        # Remove any remaining special formatting or markdown
        text = re.sub(r"[\*\#\@\<\>\[\]\(\)\{\}\`\~\|\=\+]", "", text)
        
        # Clean punctuation, normalize spacing
        text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Title case
        text = text.title()
        
        # Additional sanitization: max 5 words and at least 2 words for a good label
        words = text.split()
        if len(words) > 5:
            text = " ".join(words[:5])
        
        return text.strip()

    def validate_label(self, label: str, keywords: List[str], category: str, subcategory: str) -> Dict:
        """
        Validate a generated label against quality criteria.
        
        Args:
            label: The label to validate
            keywords: List of keywords from the cluster
            category: The category name
            subcategory: The subcategory name
            
        Returns:
            Dict with validation results and score
        """
        word_count = len(label.split())
        
        # Get the full name of the subcategory from mapping
        subcategory_full = subcat_fullname_mapping.get(subcategory, subcategory)
        category_full = cat_fullname_mapping.get(category, category)
        
        # Check if label is the same as the subcategory (case insensitive comparison)
        is_same_as_subcategory = (label.lower() == subcategory_full.lower() or 
                                  label.lower() == subcategory.lower())
        
        # Basic validation criteria
        results = {
            "word_count": word_count,
            "too_short": word_count < 2,
            "too_long": word_count > 5,
            "contains_category": category_full.lower() in label.lower(),
            "contains_subcategory": subcategory_full.lower() in label.lower(),
            "is_same_as_subcategory": is_same_as_subcategory,
            "score": 0
        }
        
        # Calculate a quality score
        score = 10  # Start with perfect score
        
        if results["too_short"] or results["too_long"]:
            score -= 3
        
        # Penalize if the label is just the category or subcategory
        if results["contains_category"]:
            score -= 2
            
        if results["contains_subcategory"]:
            score -= 2
            
        # Heavily penalize if label is the same as subcategory
        if results["is_same_as_subcategory"]:
            score -= 8  # This should effectively disqualify the label
            
        # Count how many keywords are reflected in the label
        keyword_matches = sum(1 for kw in keywords if kw.lower() in label.lower())
        if keyword_matches == 0:
            score -= 3  # Penalize if no keywords are included
        else:
            score += min(keyword_matches, 2)  # Bonus for including keywords (up to 2)
            
        results["score"] = max(0, score)
        return results

    def create_improved_prompt(self, keywords: List[str], category: str, subcategory: str) -> str:
        """
        Create an improved prompt for label generation with strict formatting instructions.
        
        Args:
            keywords: List of keywords from the cluster
            category: The category name
            subcategory: The subcategory name
            
        Returns:
            A well-structured prompt for the LLM
        """
        category_full = cat_fullname_mapping.get(category, category)
        subcategory_full = subcat_fullname_mapping.get(subcategory, subcategory)
        
        prompt = f"""You are a scientific taxonomy expert. Your task is to create a brief, specific research topic label.

CONTEXT:
- Category: {category_full}
- Subcategory: {subcategory_full}
- Keywords: {', '.join(keywords)}

STRICT FORMAT RULES:
1. Output ONLY 2-5 words that form a cohesive research topic label
2. Do NOT include articles (the, a, an) at the end of the label
3. Do NOT include conjunctions (and, or) and prepositions (of, in, on, etc.) at the end of the label
4. Do NOT include any reference markers or citation text
5. Do NOT include words like "References", "Citation", or numbers
6. Output ONLY the label text - no explanation, no additional text

INSTRUCTIONS:
1. Create a concise, descriptive label for this research topic (2-5 words only)
2. Make it specific and distinctive - NEVER use generic terms like "{category_full}" or "{subcategory_full}"
3. NEVER output a label that is identical to the subcategory name "{subcategory_full}"
4. Focus on the technical content reflected in the keywords
5. Do NOT include words like "Category", "Subcategory", or "Keywords" in your response

EXAMPLES OF GOOD LABELS:
- For subcategory "Machine Learning": "Neural Architecture Search"
- For subcategory "Quantum Mechanics": "Entanglement Phase Transitions"
- For subcategory "Cancer Biology": "Tumor Microenvironment Analysis"
- For subcategory "NLP": "Conversation State Tracking"

EXAMPLES OF BAD LABELS:
- "{category_full}" (too generic)
- "{subcategory_full}" (same as subcategory name)
- "AI" (too short)
- "Problems Or" (ends with conjunction)
- "Portfolio Return Modeling References 1" (includes reference text)
- "Category: Physics - Topic: Magnetism" (wrong format)

Generate LABEL based on requirements above:
"""
        
        return prompt

    def get_candidate_labels(self, prompt: str, top_k: int = 5) -> List[str]:
        """
        Get candidate labels from the LLM with multiple retries.
        
        Args:
            prompt: The prompt to send to the LLM
            top_k: Number of alternatives to request
            
        Returns:
            List of candidate labels
        """
        url = "https://api.together.xyz/inference"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 25,  # Reduced to prevent long outputs
            "temperature": self.temperature,
            "top_k": top_k
        }

        for attempt in range(self.retries):
            try:
                response = requests.post(url, headers=headers, json=data)
                result = response.json()
                if "output" in result and isinstance(result["output"], dict):
                    choices = result["output"].get("choices", [])
                    labels = []
                    for c in choices:
                        if "text" in c:
                            # Apply label cleaning and validation
                            label = self.clean_label(c.get("text", ""))
                            if label and 2 <= len(label.split()) <= 5:  # Only accept if 2-5 words
                                labels.append(label)
                    return list(filter(None, set(labels)))  # Return unique, non-empty labels
            except Exception as e:
                logger.error(f"API error on attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)
        
        return []

    def select_best_label(self, 
                         candidates: List[str], 
                         keywords: List[str], 
                         category: str, 
                         subcategory: str) -> str:
        """
        Select the best label from candidates using multiple criteria.
        
        Args:
            candidates: List of candidate labels
            keywords: List of keywords from the cluster
            category: The category name
            subcategory: The subcategory name
            
        Returns:
            The best label or a fallback
        """
        if not candidates:
            return self.create_fallback_label(keywords, category, subcategory)
        
        # Get full names for better comparison
        subcategory_full = subcat_fullname_mapping.get(subcategory, subcategory)
        
        # Filter out labels that are the same as the subcategory name
        filtered_candidates = [
            label for label in candidates 
            if label.lower() != subcategory_full.lower() and label.lower() != subcategory.lower()
        ]
        
        # If filtering removed all candidates, try to create alternatives
        if not filtered_candidates and candidates:
            logger.warning(f"All candidate labels matched subcategory name '{subcategory_full}'. Creating alternatives.")
            return self.create_fallback_label(keywords, category, subcategory)
        
        # Validate and score each candidate
        scored_candidates = []
        for label in filtered_candidates:
            validation = self.validate_label(label, keywords, category, subcategory)
            scored_candidates.append((label, validation["score"]))
        
        # First try: select highest scoring candidate
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_candidates:
            return self.create_fallback_label(keywords, category, subcategory)
            
        best_label, best_score = scored_candidates[0]
        
        # If best score is too low, try semantic similarity as well
        if best_score < 7 and len(filtered_candidates) > 1:
            # Encode keywords and candidates
            keyword_text = ", ".join(keywords)
            keyword_embed = self.embedder.encode(keyword_text, convert_to_tensor=True)
            label_embeds = self.embedder.encode(filtered_candidates, convert_to_tensor=True)
            
            # Calculate similarity
            similarities = util.cos_sim(keyword_embed, label_embeds)[0]
            best_sim_idx = torch.argmax(similarities).item()
            
            # If semantic similarity winner is different and has reasonable score
            sim_winner = filtered_candidates[best_sim_idx]
            sim_winner_score = next(score for label, score in scored_candidates if label == sim_winner)
            
            if sim_winner != best_label and sim_winner_score >= 5:
                # Blend the decision: prefer semantic match if it's decent
                best_label = sim_winner
        
        # Track the quality for reporting
        self.label_scores[best_label] = best_score
        
        # If still low quality, fall back to keyword-based label
        if best_score < 5:
            logger.warning(f"Low quality label '{best_label}' (score {best_score}). Using fallback.")
            fallback = self.create_fallback_label(keywords, category, subcategory)
            self.fallback_count += 1
            return fallback
            
        return best_label

    def create_fallback_label(self, keywords: List[str], category: str = None, subcategory: str = None) -> str:
        """
        Create a fallback label based on keywords, ensuring it's not identical to subcategory.
        
        Args:
            keywords: List of keywords from the cluster
            category: Optional category name for context
            subcategory: Optional subcategory name to avoid duplication
            
        Returns:
            A label constructed from top keywords
        """
        # Get full subcategory name if provided
        subcategory_full = ""
        if subcategory:
            subcategory_full = subcat_fullname_mapping.get(subcategory, subcategory)
        
        # Sort keywords by length (prefer more specific terms)
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        # Get the most frequent words in keywords (might represent key concepts)
        word_counts = {}
        for kw in keywords:
            for word in kw.lower().split():
                if len(word) > 3:  # Only count meaningful words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort words by frequency
        frequent_words = [word for word, count in 
                         sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                         if count > 1]  # Only words that appear multiple times
        
        # Create candidate fallback labels
        candidates = []
        
        # Method 1: Top 1-3 keywords
        top_keywords = []
        for kw in sorted_keywords:
            if len(" ".join(top_keywords + [kw]).split()) <= 5:
                top_keywords.append(kw)
            if len(top_keywords) >= 3:
                break
                
        if top_keywords:
            candidates.append(" ".join(top_keywords).title())
            
        # Method 2: Use frequent words
        if frequent_words and len(frequent_words) >= 2:
            freq_label = " ".join(frequent_words[:3]).title()
            candidates.append(freq_label)
            
        # Method 3: Add a differentiator to subcategory if all else fails
        if subcategory_full:
            # Try to find a distinctive keyword not in the subcategory name
            distinctive_keywords = [k for k in sorted_keywords 
                                   if k.lower() not in subcategory_full.lower()]
            
            if distinctive_keywords:
                # Add a distinctive word to make the label different from subcategory
                candidates.append(f"{subcategory_full} {distinctive_keywords[0].title()}")
                candidates.append(f"{distinctive_keywords[0].title()} {subcategory_full}")
                
                # Try keyword combinations
                if len(distinctive_keywords) >= 2:
                    candidates.append(f"{distinctive_keywords[0].title()} {distinctive_keywords[1].title()}")
        
        # Clean and validate all candidates
        valid_candidates = []
        for candidate in candidates:
            label = self.clean_label(candidate)
            # Ensure it's not the same as subcategory
            if subcategory_full and (label.lower() == subcategory_full.lower() or 
                                    label.lower() == subcategory.lower()):
                continue
            valid_candidates.append(label)
            
        if valid_candidates:
            return valid_candidates[0]
            
        # Ultimate fallback if nothing else works
        return "Specialized Research Topic"

    def generate_and_merge_labels(self):
        """Generate high-quality labels for clusters and merge them into the papers dataset."""
        logger.info("Generating improved fine-grained labels...")
        fine_labels = []
        
        # Track subcategory duplication issues
        subcategory_matches = 0
        
        for _, row in tqdm(self.clusters_df.iterrows(), total=len(self.clusters_df)):
            cluster_id = row.get("cluster_id", -1)
            keywords_str = row.get("keywords", "")
            category = row.get("category", "Unknown")
            subcategory = row.get("subcategory", "Unknown")
            
            if cluster_id == -1:
                fine_labels.append("NA")
                continue

            # Create a unique cache key that includes category/subcategory
            cache_key = f"{category}|{subcategory}|{keywords_str}"
            
            if cache_key in self.generated_cache:
                fine_labels.append(self.generated_cache[cache_key])
                continue

            # Parse keywords into list
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            
            # Get full subcategory name
            subcategory_full = subcat_fullname_mapping.get(subcategory, subcategory)
            
            # Generate improved label
            prompt = self.create_improved_prompt(keywords, category, subcategory)
            candidates = self.get_candidate_labels(prompt, top_k=5)
            
            # Check if any candidates match the subcategory name
            matching_candidates = [c for c in candidates if c.lower() == subcategory_full.lower()]
            if matching_candidates:
                subcategory_matches += 1
                logger.warning(f"Generated label matches subcategory '{subcategory_full}'. Will use alternative.")
            
            # Select the best label
            label = self.select_best_label(candidates, keywords, category, subcategory)
            
            # Final check to ensure it's not the same as subcategory
            if label.lower() == subcategory_full.lower() or label.lower() == subcategory.lower():
                logger.warning(f"Final label still matches subcategory '{subcategory_full}'. Using special fallback.")
                label = self.create_fallback_label(keywords, category, subcategory)
                
                # If still matching, add a placeholder modifier
                if label.lower() == subcategory_full.lower() or label.lower() == subcategory.lower():
                    label = f"Advanced {subcategory_full}"
            
            # Cache and save
            self.generated_cache[cache_key] = label
            fine_labels.append(label)

        self.clusters_df["fine_topic_label"] = fine_labels

        logger.info("Merging fine-grained labels into paper dataset...")
        merged = self.papers_df.drop(columns=["fine_topic_label"], errors="ignore").merge(
            self.clusters_df[["cluster_id", "fine_topic_label"]],
            how="left",
            left_on="cluster",
            right_on="cluster_id"
        )

        # Apply fallback for any missing labels
        merged["fine_topic_label"] = merged.apply(
            lambda row: subcat_fullname_mapping.get(row["subcategory"], row["subcategory"])
            if pd.isna(row["fine_topic_label"]) else row["fine_topic_label"], axis=1
        )

        merged.drop(columns=["cluster_id"], inplace=True)
        self.papers_df = merged
        
        # Report label quality statistics
        logger.info(f"Generated {len(self.generated_cache)} unique labels")
        logger.info(f"Found {subcategory_matches} candidates matching subcategory names")
        logger.info(f"Used fallback labels for {self.fallback_count} clusters")
        
        scores = list(self.label_scores.values())
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info(f"Average label quality score: {avg_score:.2f}/10")

    def build_taxonomy_tree(self):
        """Build the hierarchical taxonomy tree from the labeled papers."""
        logger.info("Building taxonomy tree...")
        taxonomy = {}
        for _, row in self.papers_df.iterrows():
            cat = row.get("category")
            subcat = row.get("subcategory")
            fine = row.get("fine_topic_label")
            if not cat or not subcat or not fine:
                continue
            taxonomy.setdefault(cat, {}).setdefault(subcat, []).append(fine)

        # Deduplicate and sort topics within each subcategory
        for cat in taxonomy:
            for subcat in taxonomy[cat]:
                # Convert to set to remove duplicates, then back to sorted list
                taxonomy[cat][subcat] = sorted(list(set(taxonomy[cat][subcat])))

        self.taxonomy = taxonomy

    def save_output(self):
        """Save the results to files."""
        self.papers_df.to_csv(self.input_papers_csv, index=False)
        logger.info(f"Updated papers saved to {self.input_papers_csv}")

        named_clusters_path = self.input_cluster_csv.parent / "cluster_summary_named.csv"
        self.clusters_df.to_csv(named_clusters_path, index=False)
        logger.info(f"Named clusters saved to {named_clusters_path}")

        with open(self.output_taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(self.taxonomy, f, indent=2)
        logger.info(f"Taxonomy tree saved to {self.output_taxonomy_path}")
        
        # Save label quality report
        quality_report_path = self.input_cluster_csv.parent / "label_quality_report.csv"
        quality_df = pd.DataFrame({
            'label': list(self.label_scores.keys()),
            'quality_score': list(self.label_scores.values())
        })
        quality_df.sort_values('quality_score', ascending=False).to_csv(quality_report_path, index=False)
        logger.info(f"Label quality report saved to {quality_report_path}")

    def run(self):
        """Run the complete label generation pipeline."""
        self.load_data()
        self.generate_and_merge_labels()
        self.build_taxonomy_tree()
        self.save_output()