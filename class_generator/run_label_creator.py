import logging
from label_creator import ClusterNameGenerator
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    input_path = Path("clustering_results/cluster_summary.csv")
    output_path = Path("clustering_results/cluster_summary_named.csv")
    
    if not input_path.exists():
        logger.error(f"Cluster summary file not found: {input_path}")
        return
    
    logger.info("Initializing cluster name generator...")
    
    generator = ClusterNameGenerator(
        input_csv_path=str(input_path),
        output_csv_path=str(output_path),
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        api_key = "e9f81f80dcc23956c59b52785e7ceb2c50c66c519ce20ec61cf56272ee99b694"
    )
    
    logger.info("Running topic name generation pipeline...")
    generator.run()
    
    logger.info(f"Generated topic names saved to {output_path}")

if __name__ == "__main__":
    main()
