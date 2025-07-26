from collections import defaultdict

def build_combined_context(data_list):
    papers = defaultdict(list)
    for entry in data_list:
        papers[entry["paper_id"]].append(entry)

    blocks = []
    for paper_id, eq_list in papers.items():
        # sort by numeric eq id if you can
        try:
            eq_list.sort(key=lambda x: int(x["equation_id"]))
        except:
            eq_list.sort(key=lambda x: x["equation_id"])
        for i, eq in enumerate(eq_list):
            eq_id = eq["equation_id"]
            lines = [f"Combined context for paper {paper_id}, Equation {eq_id}", ""]
            for j in range(i+1):
                e = eq_list[j]
                lines.append(f"Context for Equation {e['equation_id']}:")
                lines.append(e["context"])
                # Only include description for prior equations (not current)
                if j < i and e.get("description"):
                    lines.append("Description:")
                    lines.append(e["description"])
                # only show prior LaTeX
                if j < i:
                    for tex in e.get("EQ_latex", []):
                        lines.append(f"<latex>{tex}</latex>")
                lines.append("")  # blank
            blocks.append((paper_id, eq_id, "\n".join(lines)))
    return blocks


def get_system_prompt():
    """
    Creates a strict system prompt to guide the LLM's behavior.
    """
    return """You are a scientific writing assistant trained to generate LaTeX equations for research papers.

Your task is to:

1.  Understand the context and flow of the document.
2.  Generate the LaTeX code for the next equation (Equation n) based on the provided context.

**Response Rules:**
- Respond with the LaTeX code and the description only.
- Wrap the LaTeX code in `<latex>...</latex>` tags.
- Wrap the variable description in `<description>...</description>` tags.
- **You MUST NOT include any other text, explanations, or thought processes.** Your response must only contain the requested tags and their content.
"""

def construct_final_prompt(combined_context):
    """
    Construct the final prompt for the agent using the combined context.
    """
    prompt = f"""
    You are a scientific writing assistant trained to generate LaTeX equations for research papers.

    For each equation (Equation n), you will receive a combined context which includes:

    - The surrounding natural language context for each equation up to Equation n.
    - Any prior equations (Equation 1 to Equation n–1) along with their LaTeX representations, if Equation n > 1.
    - Descriptions for prior equations (Equation 1 to Equation n–1), if available.

    Your task is to:

    1.Understand the context and flow of the document,
    2.Generate the LaTeX code for the next equation (Equation n) based on the combined context,

    Respond with the LaTeX code and the description only, wrap latex in <latex>...</latex> tags,
    
    wrap one sentence variable description in <description>...</description>. Do not include any explanation or extra commentary.
    {combined_context}
    """
    return prompt