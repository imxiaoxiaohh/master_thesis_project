import pandas as pd
from utils import clean_latex
import re

def load_and_preprocess_data(input_file_path, output_file_path):
    """
    Clean LaTeX equations in the CSV file and remove empty rows.
    
    Returns: The cleaned dataframe
    """
    df = pd.read_csv(input_file_path)
    
    # Apply clean_latex function to both columns
    df['ground_truth_eq'] = df['ground_truth_eq'].apply(clean_latex)
    df['generated_equation'] = df['generated_equation'].apply(clean_latex)
    
    # Remove rows where both equation columns are empty
    df = df[~(df['ground_truth_eq'].str.strip() == '') & ~(df['generated_equation'].str.strip() == '')]
    
    # Drop the 'context' column if it exists
    if 'context' in df.columns:
        df = df.drop(columns=['context'])

    df.to_csv(output_file_path, index=False)
    
    return df





