# Paths
# DATA_DIR = "data/"
# RAW_CSV = DATA_DIR + "evaluation_table.csv"
# PROCESSED_CSV = DATA_DIR + "cleaned_data.csv"
# RESULTS_CSV = DATA_DIR + "test_llm_judge_results.csv"

RAW_CSV_DIR = "data/raw_csv/"
PROCESSED_CSV_DIR = "data/preprocessed_csv/"
RESULTS_CSV_DIR = "data/result_csv/"


SEMANTIC_ACCURACY_PROMPT = """
You are evaluating the SEMANTIC ACCURACY of a generated equation and variable description compared to the reference and promblem context provided below.

**Context**:{context}
**Ground Truth Equation (multi equations separated by `||`)**: {eq_gt}
**Generated Equation (multi equations separated by `||`)**: {eq_gen}
**Ground Truth Description**:{description_gt}
**Generated Description**: {description_gen}

**Task**:  
Measures whether the generated equation expresses the same mathematical relationships as the ground truth equation, allowing for equivalent rearrangements or variable renaming.
**Semantic Accuracy Rubric (1-5)**:
	- **5**: Exact same meaning; only trivial variations allowed (e.g., algebraic rearrangement, variable renaming).
	- **4**: Near-match; small semantic deviations but intent clearly preserved.
	- **3**: Core meaning largely correct but includes some secondary inaccuracies (e.g., incorrect constant or minor relation misinterpreted).
	- **2**: Partial overlap in meaning; significant misunderstanding or contradiction.
	- **1**: Completely unrelated; wrong understanding of task.

**IMPORTANT**  
Respond **only** in the following format (and nothing else):

Score: <1-5>  
Explanation: <brief justification>

"""

REASONING_PROMPT = """
You are evaluating the REASONING QUALITY of a generated equation and variable description compared to the reference and promblem context provided below.

**Context**:{context}
**Ground Truth Equation (multi equations separated by `||`)**: {eq_gt}
**Generated Equation (multi equations separated by `||`)**: {eq_gen}
**Ground Truth Description**:{description_gt}
**Generated Description**: {description_gen}

**Task:**  
Evaluates the logical clarity and correctness of the relationships implied or inferred by the generated equations and descriptions, as explicit reasoning steps are not available. 
You should mentally infer the reasoning process the model may have followed and evaluate its logical clarity.

**REASONING QUALITY Rubric (1-5):**  
- **5**: Clearly inferred logical relationships between variables and operations. Implied reasoning path fully logical and consistent.
- **4**: Generally logical inferred relationships; minor ambiguity or small logical gaps not significantly impacting clarity.
- **3**: Partially clear logic; noticeable gaps or ambiguity in inferred reasoning.
- **2**: Significant logical inconsistencies or confusion in inferred reasoning; inferred logic barely understandable.
- **1**: No coherent inferred logic; relationships confusing or nonsensical.

**IMPORTANT**  
Respond **only** in the following format (and nothing else):

Score: <1-5>  
Explanation: <brief justification>
"""

COMPLETENESS_PROMPT = """
You are evaluating the INFORMATIONAL COMPLETENESS of a generated equation and variable description according to the promblem context provided below.

**Context**:{context}

**Generated Equation**: {eq_gen}

**Generated Description**: {description_gen}

**Task:**  
Evaluates if the generated equations and descriptions provide a complete final solution that fully answers or resolves the problem scenario, considering intermediate steps are not explicitly generated or required.

**INFORMATIONAL COMPLETENESS Rubric (1-5):**  
- **5**: All necessary terms, variables, and constraints are present in the equation/block. No signs of under-specification.
- **4**: Minor omissions, e.g., one term or constraint missing, but the equation is still practically usable.
- **3**: Noticeable omissions of key components, but the overall structure is still interpretable as partially solving the problem.
- **2**: Several important components missing or ambiguous.
- **1**: Equation feels incomplete or disconnected from any meaningful solution.

**IMPORTANT**  
Respond **only** in the following format (and nothing else):

Score: <1-5>  
Explanation: <brief justification>
"""

SYNTACTIC_CORRECTNESS_PROMPT = """
You are evaluating the SYNTACTIC CORRECTNESS of a generated equation provided below.

**Generated Equation**: {eq_gen}


**Task:**  
Measures whether the equation is mathematically well-formed and syntactically valid (e.g., parsable LaTeX, balanced structure), independent of correctness of meaning.

**SYNTACTIC CORRECTNESS Rubric (1-5):**  
- **5**: Fully valid; no syntax, parsing, or formatting issues.
- **4**: Minor syntax issues (e.g., a bracket or LaTeX detail) but easily correctable.
- **3**: Noticeable formatting issues but still parseable and interpretable.
- **2**: Multiple syntax errors that hinder rendering or understanding.
- **1**: Completely ill-formed; not parseable or interpretable.

**IMPORTANT**  
Respond **only** in the following format (and nothing else):

Score: <1-5>  
Explanation: <brief justification>
"""

CONTEXUAL_APPROPRIATENESS_PROMPT = """
You are evaluating the CONTEXUAL APPROPRIATENESS of a generated equation and variable description according to the promblem context provided below.

**Context**:{context}

**Generated Equation**: {eq_gen}

**Generated Description**: {description_gen}

**Task:**  
Assesses whether the generated equation and description appropriately match the scenario, intent, or constraints of the original problem statement and reflect the specific scenario or problem context provided.

**CONTEXUAL APPROPRIATENESS (1-5):**  
- **5**: Perfectly matches and clearly addresses the described context.
- **4**: Strong alignment with minor ambiguity or weakly integrated detail.
- **3**: Partial relevance; some generic or incorrectly inferred parts.
- **2**: Loosely related; insufficient follow-through on context.
- **1**: Completely irrelevant or hallucinated content.


**IMPORTANT**  
Respond **only** in the following format (and nothing else):

Score: <1-5>  
Explanation: <brief justification>
"""