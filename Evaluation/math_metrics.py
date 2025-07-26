

from sympy.parsing.latex import parse_latex
from sympy import Basic
from zss import Node, simple_distance

# Convert SymPy expression to zss.Node
def sympy_to_zss(expr):
    if not isinstance(expr, Basic):
        return Node(str(expr))
    label = type(expr).__name__
    children = [sympy_to_zss(arg) for arg in expr.args]
    return Node(label, children)

def tree_edit_distance_zss(latex1, latex2):
    try:
        expr1 = parse_latex(latex1)
        expr2 = parse_latex(latex2)
        tree1 = sympy_to_zss(expr1)
        tree2 = sympy_to_zss(expr2)
    except Exception as e:
        print(f"Parse error: {e}")
        return 1.0, {'parse_error': True}

    # zss.simple_distance returns the unnormalized edit distance
    dist = simple_distance(tree1, tree2)
    max_size = max(len(list(tree1.iter())), len(list(tree2.iter())))
    norm_dist = dist / max_size if max_size > 0 else 0.0
    return norm_dist, {
        'parse_error': False,
        'raw_distance': dist,
        'tree1_size': len(list(tree1.iter())),
        'tree2_size': len(list(tree2.iter()))
    }

import pandas as pd

def split_equations(equation_str):
    """Split a string by '||' and strip whitespace from each part."""
    if pd.isna(equation_str):
        return []
    return [eq.strip() for eq in equation_str.split('||') if eq.strip()]

def avg_tree_edit_distance(ref_str, pred_str):
    ref_eqs = split_equations(ref_str)
    pred_eqs = split_equations(pred_str)
    n = min(len(ref_eqs), len(pred_eqs))
    if n == 0:
        return 1.0, []  # No equations to compare

    ted_scores = []
    for i in range(n):
        ted, meta = tree_edit_distance_zss(ref_eqs[i], pred_eqs[i])
        ted_scores.append(ted)
    avg_ted = sum(ted_scores) / n
    return avg_ted, ted_scores

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")

    # Test with just a few examples (e.g., indices 0 to 10)
    for idx in range(0,11):
        row = df.iloc[idx]
        reference = row['ground_truth_eq']
        prediction = row['generated_equation']

        avg_ted, ted_scores = avg_tree_edit_distance(reference, prediction)
        print(f"\nExample {idx + 1}:")
        print(f"Reference: {reference}")
        print(f"Prediction: {prediction}")
        print(f"Avg Tree Edit Distance: {avg_ted:.4f}")
        print(f"Individual TEDs: {ted_scores:.4f}")
        print("-" * 60)