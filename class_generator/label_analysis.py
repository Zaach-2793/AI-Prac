import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json

class TaxonomyVisualizer:
    def __init__(self, papers_csv_path: str, taxonomy_json_path: str):
        self.papers_csv_path = Path(papers_csv_path)
        self.taxonomy_json_path = Path(taxonomy_json_path)
        self.df = None
        self.taxonomy = None

    def load_data(self):
        if not self.papers_csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {self.papers_csv_path}")
        if not self.taxonomy_json_path.exists():
            raise FileNotFoundError(f"Missing taxonomy JSON: {self.taxonomy_json_path}")

        self.df = pd.read_csv(self.papers_csv_path)
        with open(self.taxonomy_json_path, 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)

    def plot_unique_label_counts(self, output_path: str = "taxonomy_unique_labels_heatmap.png"):
        records = []
        for category, subcats in self.taxonomy.items():
            for subcat, fine_labels in subcats.items():
                records.append({
                    "Category": category,
                    "Subcategory": subcat,
                    "Unique Labels": len(set(fine_labels))
                })

        df_plot = pd.DataFrame(records)
        plt.figure(figsize=(16, 40))
        pivot = df_plot.pivot(index="Subcategory", columns="Category", values="Unique Labels").fillna(0)
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title("Number of Unique Fine-Grained Labels per Category and Subcategory")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved heatmap to {output_path}")

    def barplot_per_category(self, output_path="taxonomy_labels_per_category.png"):
        records = []
        for category, subcats in self.taxonomy.items():
            for subcat, fine_labels in subcats.items():
                records.append({
                    "Category": category,
                    "Subcategory": subcat,
                    "Unique Labels": len(set(fine_labels))
                })

        df_plot = pd.DataFrame(records)
        category_sums = df_plot.groupby("Category")["Unique Labels"].sum().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=category_sums, x="Category", y="Unique Labels", palette="Set3")
        plt.title("Total Unique Fine-Grained Labels per Category")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved category barplot to {output_path}")

    def run(self):
        self.load_data()
        self.plot_unique_label_counts()
        self.barplot_per_category()

