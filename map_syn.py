import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_prompt_groups(directory: str) -> Dict[str, int]:
    """
    Creates a dictionary mapping each synonymous prompt to a unique group code using the
    directory and parquet file tree

    Parameters:
    directory (str): Path to the nested directory containing prompt directories.

    Returns:
    Dict[str, int]: A dictionary where keys are prompt names and values are group codes.
    """
    prompt_groups = {}
    group_code = 0

    for prompt_dir in Path(directory).iterdir():
        if prompt_dir.is_dir():
            group_code += 1
            for synonym_file in prompt_dir.glob('*.parquet'):
                synonym_name = synonym_file.stem
                if synonym_name not in prompt_groups:
                    prompt_groups[synonym_name] = group_code

    logging.info(f"Created {len(prompt_groups)} prompt groups.")
    return prompt_groups

def map_group_codes_to_dataset(dataset_path: str, prompt_groups: Dict[str, int]) -> pd.DataFrame:
    """
    Maps group codes to prompts in the dataset.

    Parameters:
    dataset_path (str): Path to the large dataset file.
    prompt_groups (Dict[str, int]): A dictionary of prompt to group code mappings.

    Returns:
    pd.DataFrame: The updated dataset with group codes mapped.
    """
    try:
        df = pd.read_parquet(dataset_path)
        logging.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

        if 'prompt' not in df.columns:
            raise KeyError("The dataset does not contain a 'prompt' column.")

        df['group_code'] = df['prompt'].map(prompt_groups)

        missing_prompts = df['prompt'].isna().sum()
        if missing_prompts > 0:
            logging.warning(f"{missing_prompts} prompts in the dataset could not be mapped to a group code.")

        return df

    except Exception as e:
        logging.error(f"Failed to map group codes to dataset: {e}")
        raise

# Example usage
if __name__ == "__main__":
    directory = r'C:\Users\MiM\Documents\Projects\Elsa\results\scores'
    dataset_path = r'C:\Users\MiM\Documents\Projects\Elsa\src\checkpoint.parquet'

    # Create prompt groups
    prompt_groups = create_prompt_groups(directory)

    # Map group codes to the dataset
    df_with_codes = map_group_codes_to_dataset(dataset_path, prompt_groups)

    # Save the updated dataset if needed
    output_path = r'C:\Users\MiM\Documents\Projects\Elsa\results\mapped.parquet'
    df_with_codes.to_parquet(output_path)
    logging.info(f"Updated dataset saved to {output_path}")

    # Print a sample of the updated dataframe
    logging.info(f"Sample of the updated dataframe:\n{df_with_codes.head()}")
