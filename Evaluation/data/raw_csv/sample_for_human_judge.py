import csv
import random

# # Number of samples to select
# NUM_SAMPLES = 100  # Change this as needed

# # Input and output file paths
# INPUT_CSV = 'deepseek-R1_results_evaluation_table.csv'
# OUTPUT_CSV = 'human_judge_deepseek.csv'

# # Read all rows from the input CSV
# with open(INPUT_CSV, 'r', encoding='utf-8') as infile:
#     reader = list(csv.reader(infile))
#     header = reader[0]
#     rows = reader[1:]

# # Randomly sample rows
# sampled_rows = random.sample(rows, min(NUM_SAMPLES, len(rows)))

# # Write sampled rows to the output CSV
# with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as outfile:
#     writer = csv.writer(outfile)
#     writer.writerow(header)
#     writer.writerows(sampled_rows)

# print(f"Sampled {len(sampled_rows)} rows to {OUTPUT_CSV}") 

import csv

# List of LLM result files and their output names
llm_files = [
    ("qwen3-235B_results_evaluation_table.csv", "human_judge_qwen3-235B.csv"),
    ("llama3.3-70B_results_evaluation_table.csv", "human_judge_llama3.3-70B.csv"),
    ("gpt-4o-mini_results_evaluation_table.csv", "human_judge_gpt-4o-mini.csv"),
    ("gpt-4.1_results_evaluation_table.csv", "human_judge_gpt-4.1.csv"),
]

# Load the reference (paper_id, equation_id) pairs from human_judge_deepseek.csv
with open("human_judge_deepseek.csv", "r", encoding="utf-8") as ref_file:
    reader = csv.DictReader(ref_file)
    key_pairs = {(row["paper_id"], row["equation_id"]) for row in reader}

# For each LLM result file, extract matching rows
for input_csv, output_csv in llm_files:
    with open(input_csv, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = [row for row in reader if (row["paper_id"], row["equation_id"]) in key_pairs]
        header = reader.fieldnames

    with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")