#This file contains code for cleaning the csv, latex string, NAN, load embeding model, tokenizarion
import re
import numpy as np
import pandas as pd
import torch
import numpy as np
from transformers import GPT2TokenizerFast, GPT2Model
from transformers import AutoTokenizer, AutoModel



def clean_latex(latex_str):
    """
    Clean and normalize LaTeX equations for better comparison.
    Handles LaTeX delimiters, whitespace, common operators, and common formatting artifacts.
    """
    if not isinstance(latex_str, str) or pd.isna(latex_str) or \
       latex_str == "Agent stopped due to iteration limit or time limit.":
        return ""
    
    latex_str = latex_str.strip()
    
    latex_str = re.sub(r'\\begin\{[^}]+\}', '', latex_str)
    latex_str = re.sub(r'\\end\{[^}]+\}', '', latex_str)
    
    # Replace multiple spaces with a single space
    latex_str = re.sub(r'\s+', ' ', latex_str)
    
    # Remove spaces around common operators
    latex_str = re.sub(r'\s*([=+\-*/\\])\s*', r'\1', latex_str)
    latex_str = latex_str.rstrip('.,;: ')
    
    # Remove font/format macros in one sweep
    font_pat = re.compile(
        r'\\(?:mathrm|mathtt|mathbf|mathit|mathsf|mathcal|mathbb|mathscr|'
        r'mathfrak|mathnormal|boldsymbol|bm|text|operatorname)'
        r'\{([^}]+)\}'
    )
    # repeat until nested macros all gone
    while font_pat.search(latex_str):
        latex_str = font_pat.sub(r'\1', latex_str)
    
    # Handle multiple equations separated by '||'
    if '||' in latex_str:
        parts = [clean_latex(part) for part in latex_str.split('||')]
        return ' || '.join(parts)

    return latex_str.strip()





'''Load the embeding model for tokenization'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

def load_GPT2_models():
    global gpt2_tokenizer, gpt2_model, gpt2_embedding_layer, gpt2_positional_embedding, new_embeddings
    
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    gpt2_model = GPT2Model.from_pretrained('gpt2')
    gpt2_embedding_layer = gpt2_model.wte.to(device)
    gpt2_positional_embedding = gpt2_model.wpe.to(device)

    try:
        if device == 'cuda':
            new_embeddings_state = torch.load('new_embeddings.pth')
        else:
            new_embeddings_state = torch.load('new_embeddings.pth', map_location=torch.device('cpu'))

        new_vocab_size, embedding_dim = new_embeddings_state['weight'].shape
        new_embeddings = torch.nn.Embedding(new_vocab_size, embedding_dim).to(device)
        new_embeddings.load_state_dict(new_embeddings_state)
    except FileNotFoundError:
        print("Warning: new_embeddings.pth not found. Using default embeddings.")
        new_embeddings = None


def spacing(text):
    result = []
    for i, char in enumerate(text):
        if char == "\\":
            if i == 0 or text[i-1] != " ":
                result.append(" \\")
            else:
                result.append("\\")
        else:
            result.append(char)
    return ''.join(result)

def get_gpt2_token_embeddings(sentence):
    sentence = spacing(sentence)
    tokens = gpt2_tokenizer.encode(sentence, truncation=True, max_length=512)
    decoded_tokens = [gpt2_tokenizer.decode([token]) for token in tokens]

    token_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    positions = torch.arange(0, token_ids.size(1)).unsqueeze(0).to(device)

    if new_embeddings is not None:
        token_embeddings = torch.cat([gpt2_embedding_layer.weight, new_embeddings.weight])[token_ids]
    else:
        token_embeddings = gpt2_embedding_layer(token_ids)

    pos_embeddings = gpt2_positional_embedding(positions) * 100

    embeddings = list(zip(token_embeddings[0], pos_embeddings[0]))

    return embeddings, decoded_tokens


# MathBERT for tokenization
def load_mathbert_models():
    """Load MathBERT tokenizer and model"""
    global mathbert_tokenizer, mathbert_model
    
    mathbert_tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
    mathbert_model = AutoModel.from_pretrained("tbs17/MathBERT").to(device)
    print("MathBERT model and tokenizer loaded successfully")

def get_mathbert_embeddings(formula):
    """
    Get contextual embeddings for a LaTeX formula using MathBERT
    Returns:
        embeddings: Tensor of shape (seq_len, hidden_size)
        tokens: List of decoded tokens
    """
    inputs = mathbert_tokenizer(
        formula,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding='max_length'
    ).to(device)
    
    with torch.no_grad():
        outputs = mathbert_model(**inputs)
    
    embeddings = outputs.last_hidden_state.squeeze(0)
    
    token_ids = inputs["input_ids"].squeeze(0).tolist()
    tokens = [mathbert_tokenizer.decode([tid]) for tid in token_ids]
    
    return embeddings, tokens