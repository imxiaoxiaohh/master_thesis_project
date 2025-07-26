import argparse
import os
from config import RAW_CSV_DIR, PROCESSED_CSV_DIR, RESULTS_CSV_DIR
from preprocessing import load_and_preprocess_data
from static_metrics import texbleu, cal_levenshtein_distance, ratio, rouge_l_tokenized
import pandas as pd
from math_metrics import avg_tree_edit_distance  

def evaluation_pipeline(input_csv_filename):
    # Paths
    raw_csv_path = os.path.join(RAW_CSV_DIR, input_csv_filename)
    cleaned_csv_filename = input_csv_filename.replace('.csv', '_cleaned.csv')
    cleaned_csv_path = os.path.join(PROCESSED_CSV_DIR, cleaned_csv_filename)
    metrics_csv_filename = input_csv_filename.replace('.csv', '_metrics.csv')
    metrics_csv_path = os.path.join(RESULTS_CSV_DIR, metrics_csv_filename)

    # Ensure output directories exist
    os.makedirs(PROCESSED_CSV_DIR, exist_ok=True)
    os.makedirs(RESULTS_CSV_DIR, exist_ok=True)

    # Load and preprocess
    df = load_and_preprocess_data(raw_csv_path, cleaned_csv_path)
    print(f"\nTotal number of rows after cleaning: {len(df)}")
    
    results = []

    # Process all rows
    for idx, row in df.iterrows():
        print(f"Processing entry {idx + 1}/{len(df)}...")
        reference = row['ground_truth_eq']
        prediction = row['generated_equation']

        # Skip if prediction is missing or empty
        if pd.isna(prediction) or not str(prediction).strip():
            print(f"  Skipping entry {idx + 1} due to empty prediction.")
            continue

        texbleu_score, _, _ = texbleu(reference, prediction)
        lev_score = cal_levenshtein_distance(reference, prediction)
        seq_score  = ratio(reference, prediction)
        rouge_score = rouge_l_tokenized(reference, prediction)
        avg_ted, ted_scores = avg_tree_edit_distance(reference, prediction)

        result = {
            'paper_id': row.get('paper_id'),
            'equation_id': row.get('equation_id'),
            'texbleu': texbleu_score,
            'levenshtein_distance': lev_score,
            'sequence_similarity': seq_score,
            'rouge_l': rouge_score,
            'avg_tree_edit_distance': avg_ted,
            'individual_ted_scores': str(ted_scores)
        }
        results.append(result)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(metrics_csv_path, index=False)
        print(f"\nMetrics results saved to {metrics_csv_path}")

        # Print summary statistics
        print("\nSummary Statistics:")
        metric_cols = ['texbleu', 'levenshtein_distance', 'sequence_similarity', 'rouge_l', 'avg_tree_edit_distance']
        print(results_df[metric_cols].describe())
    else:
        print("No results to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a specific raw CSV file.")
    parser.add_argument('--input_csv', type=str, required=True, help='Name of the raw CSV file in data/raw_csv/')
    args = parser.parse_args()
    evaluation_pipeline(args.input_csv)