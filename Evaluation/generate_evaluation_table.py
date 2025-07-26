import os
import glob
import json
import csv

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_dir = os.path.join(base_dir, 'Generation', 'outputs')
raw_csv_dir = os.path.join(base_dir, 'Evaluation', 'data/raw_csv')
dataset_path = os.path.join(base_dir, 'Dataset', 'academic_dataset_Final.json')

# Ensure raw_csv directory exists
os.makedirs(raw_csv_dir, exist_ok=True)

# Load the academic dataset (ground truth)
with open(dataset_path, "r", encoding="utf-8") as f:
    academic_data = json.load(f)

# Create a dictionary to easily look up papers and equations in academic dataset
academic_dict = {}
for paper in academic_data:
    paper_id = paper.get("id", "")
    academic_dict[paper_id] = {}
    equations = paper.get("equations", [])
    for eq in equations:
        eq_id = eq.get("equation id", "")
        academic_dict[paper_id][eq_id] = {
            "ground_truth_eq": " || ".join(eq.get("EQ_latex", [])),
            "ground_truth_description": eq.get("description", ""),
            "context": eq.get("context", "")
        }

# Process each output JSON file in outputs_dir
json_files = glob.glob(os.path.join(outputs_dir, '*_results.json'))

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    rows = []
    for paper_id, paper_data in output_data.items():
        for eq_id, output_content in paper_data.items():
            # Skip if this paper_id or equation_id doesn't exist in the academic dataset
            if paper_id not in academic_dict or eq_id not in academic_dict[paper_id]:
                continue

            # Get ground truth data
            ground_truth_eq = academic_dict[paper_id][eq_id]["ground_truth_eq"]
            ground_truth_desc = academic_dict[paper_id][eq_id]["ground_truth_description"]
            context = academic_dict[paper_id][eq_id]["context"]

            # Extract generated equation and description from output data
            generated_eq = ""
            generated_desc = ""

            # Handle special case
            if isinstance(output_content, str) and output_content.strip() == "Agent stopped due to iteration limit or time limit.":
                generated_eq = output_content.strip()
            else:
                # Extract LaTeX from the output content
                if isinstance(output_content, str):
                    if "<latex>" in output_content:
                        latex_parts = output_content.split("<latex>")
                        if len(latex_parts) > 1:
                            latex_text = latex_parts[1].split("</latex>")[0].strip()
                            generated_eq = latex_text
                    elif "latex>" in output_content:
                        latex_parts = output_content.split("latex>")
                        if len(latex_parts) > 1:
                            latex_text = latex_parts[1].strip()
                            latex_text = latex_text.removesuffix("</")
                            generated_eq = latex_text

                    # Extract description from the output content
                    if "<description>" in output_content:
                        description_parts = output_content.split("<description>")
                        if len(description_parts) > 1:
                            description_text = description_parts[1].split("</description>")[0].strip()
                            generated_desc = description_text
                    elif "description>" in output_content:
                        description_parts = output_content.split("description>")
                        if len(description_parts) > 1:
                            description_text = description_parts[1].strip()
                            description_text = description_text.removesuffix("</")
                            generated_desc = description_text

            rows.append({
                "paper_id": paper_id,
                "equation_id": eq_id,
                "context": context,
                "ground_truth_eq": ground_truth_eq,
                "ground_truth_description": ground_truth_desc,
                "generated_equation": generated_eq,
                "generated_description": generated_desc
            })

    # Write out the results to a CSV file in raw_csv_dir
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    csv_path = os.path.join(raw_csv_dir, f"{base_name}_evaluation_table.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["paper_id", "equation_id", "context", "ground_truth_eq", "ground_truth_description", 
                      "generated_equation", "generated_description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV file '{csv_path}' has been created with {len(rows)} entries.")



