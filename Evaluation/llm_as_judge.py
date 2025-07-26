import os
import pandas as pd
from openai import OpenAI
import re
import argparse
from config import (
    SEMANTIC_ACCURACY_PROMPT,
    REASONING_PROMPT,
    COMPLETENESS_PROMPT,
    SYNTACTIC_CORRECTNESS_PROMPT,
    CONTEXUAL_APPROPRIATENESS_PROMPT,
    RAW_CSV_DIR,
    RESULTS_CSV_DIR
)


class LLMJudge:
    def __init__(self, model_name="gpt-4.1-mini", api_key=None, temperature=0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def evaluate(self, prompt):
        """Evaluate using standard chat completion"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a mathematical evaluation assistant. Respond only with the score and a one-sentence explanation in this format: 'Score: X' followed by 'Explanation: [your explanation]'"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=256
            )
            
            content = response.choices[0].message.content
            return self.parse_response(content)
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return None, f"Error occurred: {str(e)}"
    
    def parse_response(self, response):
        """Parse the LLM response to extract score and explanation"""
        # Look for patterns like "Score: 3" or "Score 3" or "3/5" etc.
        score_patterns = [
            r"Score\s*[:\-]?\s*(\d)",
            r"(\d)\s*[/]\s*[45]",  # matches "3/4" or "2/5"
            r"(?:give|rate|score).*?(\d)",
            r"^(\d)",  # number at start of response
        ]
        
        score = None
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                break
        
        # Extract explanation - everything after "Explanation:" or fallback to full response
        explanation_match = re.search(r"Explanation\s*[:\-]?\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            # If no explicit explanation format, use the whole response
            explanation = response.strip()
        
        # Clean up explanation by removing scores from both beginning and end
        explanation = re.sub(r"^Score\s*[:\-]?\s*\d+\s*", "", explanation, flags=re.IGNORECASE).strip()
        explanation = re.sub(r"\s*Score\s*[:\-]?\s*\d+\s*$", "", explanation, flags=re.IGNORECASE).strip()
        
        # Also remove any trailing standalone numbers that might be scores
        explanation = re.sub(r"\s*\n\s*\d+\s*$", "", explanation).strip()
        
        return score, explanation

def main():
    parser = argparse.ArgumentParser(description="LLM as Judge: Evaluate a specific raw CSV file.")
    parser.add_argument('--input_csv', type=str, required=True, help='Name of the raw CSV file in data/raw_csv/')
    args = parser.parse_args()

    input_csv_filename = args.input_csv
    input_csv_path = os.path.join(RAW_CSV_DIR, input_csv_filename)
    results_csv_filename = input_csv_filename.replace('.csv', '_llm_judge_results.csv')
    results_csv_path = os.path.join(RESULTS_CSV_DIR, results_csv_filename)
    # Ensure output directory exists
    os.makedirs(RESULTS_CSV_DIR, exist_ok=True)


    # Initialize the LLM judge
    judge = LLMJudge()
    
    # Load data
    df = pd.read_csv(input_csv_path)
    results = []
    save_every = 20  # Save every 5 results

    
    for idx, row in df.iterrows():
        print(f"\nProcessing entry {idx+1}...")
        
        # Prepare prompts for each evaluation dimension
        semantic_prompt = SEMANTIC_ACCURACY_PROMPT.format(
            context=row['context'],
            eq_gt=row['ground_truth_eq'],
            eq_gen=row['generated_equation'],
            description_gt=row['ground_truth_description'],
            description_gen=row['generated_description']
        )
        
        reasoning_prompt = REASONING_PROMPT.format(
            context=row['context'],
            eq_gt=row['ground_truth_eq'],
            eq_gen=row['generated_equation'],
            description_gt=row['ground_truth_description'],
            description_gen=row['generated_description']
        )
        
        completeness_prompt = COMPLETENESS_PROMPT.format(
            context=row['context'],
            eq_gen=row['generated_equation'],
            description_gen=row['generated_description']
        )

        syntactic_prompt = SYNTACTIC_CORRECTNESS_PROMPT.format(
            eq_gen=row['generated_equation']
        )

        contextual_prompt = CONTEXUAL_APPROPRIATENESS_PROMPT.format(
            context=row['context'],
            eq_gen=row['generated_equation'],
            description_gen=row['generated_description']
        )

        # Evaluate each dimension
        semantic_score, semantic_explanation = judge.evaluate(semantic_prompt)
        reasoning_score, reasoning_explanation = judge.evaluate(reasoning_prompt)
        completeness_score, completeness_explanation = judge.evaluate(completeness_prompt)
        syntactic_score, syntactic_explanation = judge.evaluate(syntactic_prompt)
        contextual_score, contextual_explanation = judge.evaluate(contextual_prompt)

        # Compile results
        result = {
            "paper_id": row['paper_id'],
            "equation_id": row['equation_id'],
            "semantic_score": semantic_score,
            # "semantic_explanation": semantic_explanation,
            "reasoning_score": reasoning_score,
            # "reasoning_explanation": reasoning_explanation,
            "completeness_score": completeness_score,
            # "completeness_explanation": completeness_explanation,
            "syntactic_score": syntactic_score,
            # "syntactic_explanation": syntactic_explanation,
            "contextual_score": contextual_score,
            # "contextual_explanation": contextual_explanation
        }
        results.append(result)
        # Save every 5 results
        if len(results) % save_every == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_csv_path, index=False)
            print(f"Saved {len(results)} results to {results_csv_path}")
    # # Save results
    if results and (len(results) % save_every != 0):
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to {results_csv_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for dimension in ['semantic', 'reasoning', 'completeness', 'syntactic', 'contextual']:
            scores = [r[f'{dimension}_score'] for r in results if r[f'{dimension}_score'] is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"Average {dimension} score: {avg_score:.2f}")
            else:
                print(f"No valid {dimension} scores")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()