import os
import json
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from constants.constants_map import * 

# Download necessary resources for text cleaning
nltk.download('stopwords')
nltk.download('punkt')

# Stopwords list
STOP_WORDS = set(stopwords.words("english"))

# Directory to store API cache
CACHE_DIR = "arxiv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_arxiv_data(category, max_results=100, start_index=0):
    """
    Fetches research paper data from the arXiv API based on the given category.
    Implements caching to avoid redundant API calls.
    """
    cache_file = os.path.join(CACHE_DIR, f"{category}_{start_index}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=cat:{category}&start={start_index}&max_results={max_results}"
    url = base_url + query

    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read().decode("utf-8")

        with open(cache_file, "w") as f:
            json.dump(xml_data, f)

        return xml_data
    
    except Exception as e:
        print(f"Error fetching data for {category}: {e}")
        return None

def clean_text(text):
    """
    Cleans and normalizes text by:
    - Lowercasing
    - Removing punctuation, numbers, extra spaces
    - Removing stopwords
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenize words
    text = " ".join([word for word in tokens if word not in STOP_WORDS])  # Remove stopwords
    return text

def parse_arxiv_data(xml_data, category, subcategory):
    """
    Parses the XML data returned by the arXiv API and extracts relevant fields.
    Cleans abstracts for NLP processing.
    """
    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        # Extract paper metadata
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        created = entry.find("{http://www.w3.org/2005/Atom}published").text.strip()
        updated = entry.find("{http://www.w3.org/2005/Atom}updated").text.strip()
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text.strip()
                   for author in entry.findall("{http://www.w3.org/2005/Atom}author")]

        cleaned_abstract = clean_text(summary)

        papers.append({
            "id": paper_id,
            "title": title,
            "category": category,
            "subcategory": subcategory if subcategory else category,  # Use category if no subcategory
            "abstract": summary,
            "cleaned_abstract": cleaned_abstract,
            "created": created,
            "updated": updated,
            "authors": ", ".join(authors)
        })

    return papers

def get_arxiv_dataframe(categories_dict, max_results=100):
    """
    Retrieves research papers from arXiv for multiple categories and subcategories.
    Returns a structured pandas DataFrame with cleaned abstracts.
    """
    all_papers = []

    for category, subcategories in categories_dict.items():
        if not subcategories:  # If no subcategories, fetch papers for the main category
            print(f"Fetching data for category: {category}...")
            xml_data = fetch_arxiv_data(category, max_results)
            if xml_data:
                papers = parse_arxiv_data(xml_data, category, None)
                all_papers.extend(papers)
        else:
            for subcat in subcategories:
                print(f"Fetching data for subcategory: {subcat} under {category}...")
                xml_data = fetch_arxiv_data(subcat, max_results)
                if xml_data:
                    papers = parse_arxiv_data(xml_data, category, subcat)
                    all_papers.extend(papers)

    df = pd.DataFrame(all_papers, columns=["id", "title", "category", "subcategory", "abstract", "cleaned_abstract", "created", "updated", "authors"])
    return df


