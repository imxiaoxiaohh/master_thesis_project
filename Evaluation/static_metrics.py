import torch
import math
import difflib
from utils import get_gpt2_token_embeddings, load_GPT2_models
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
from rouge_score import rouge_scorer

#Character‑level similarity (difflib ratio)
def ratio(reference, prediction):
    """
    Compute a SequenceMatcher-based similarity (0–1)
    """
    return SequenceMatcher(None, reference, prediction).ratio()

#Levenshtein distance
def cal_levenshtein_distance(reference, prediction):
    """
    Compute a normalized Levenshtein‐based similarity (0–1)
    between two LaTeX strings and return just the score.
    """
    dist = levenshtein_distance(reference, prediction)
    max_len = max(len(reference), len(prediction), 1)
    sim = 1 - (dist / max_len)
    return round(sim, 4)


## TexBLEU
def cosine_distance(emb1, emb2):
    return 1 - torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def token_distance(token1, token2, w_emb=0.5, w_pos=0.5, alpha=2, beta=0.1):
    emb1, pos1 = token1
    emb2, pos2 = token2
    
    emb_dist = cosine_distance(emb1, emb2) ** alpha

    pos_dist = math.tanh(beta * torch.abs(pos1 - pos2).float().mean().item())
    
    distance = w_emb * emb_dist + w_pos * pos_dist
    
    return distance

def n_gram_similarity(ref_tokens, pred_tokens, n, max_d=2.0):
    ref_ngrams = [ref_tokens[i:i+n] for i in range(len(ref_tokens)-n+1)]
    pred_ngrams = [pred_tokens[i:i+n] for i in range(len(pred_tokens)-n+1)]
    
    L_n = min(len(ref_ngrams), len(pred_ngrams))
    if L_n == 0:
        return 0
    
    # core part //author
    total_distance = sum(
        sum(token_distance(ref_token, pred_token) 
            for ref_token, pred_token in zip(ref_ngram, pred_ngram))
        for ref_ngram, pred_ngram in zip(ref_ngrams[:L_n], pred_ngrams[:L_n])
    )
    
    return 1 - (total_distance / (L_n * n)) #1 - (total_distance / (L_n * n * max_d))

load_GPT2_models()

def texbleu(reference, prediction, max_n=2, weights=None):
    ''' Computes the TexBLEU score, a BLEU-like metric for LaTeX equations using GPT-2 token embeddings.
        calculates n-gram similarities based on embedding and positional distances,
        and combines these with a brevity penalty to produce a similarity score between 0 and 1.
    '''    
    if weights is None:
        weights = [1/max_n] * max_n

    ref_embeddings, ref_decoded_tokens = get_gpt2_token_embeddings(reference)
    pred_embeddings, pred_decoded_tokens = get_gpt2_token_embeddings(prediction)

    # Handle empty embeddings case explicitly
    if len(ref_embeddings) == 0 or len(pred_embeddings) == 0:
        return 0.0, ref_decoded_tokens, pred_decoded_tokens

    n_gram_scores = [
        n_gram_similarity(ref_embeddings, pred_embeddings, n) 
        for n in range(1, max_n + 1)
    ]
    ref_length = len(ref_embeddings)
    pred_length = len(pred_embeddings)

    if pred_length == 0:
        bp = 0
    else:
        bp = 1 if pred_length > ref_length else math.exp(1 - ref_length / pred_length)
    # Avoid log(0) explicitly
    bleu_score = math.exp(sum(
        w * math.log(max(s, 1e-10)) for w, s in zip(weights, n_gram_scores)
    ))

    final_score = bleu_score * bp  # Apply brevity penalty

    return round(final_score, 4), ref_decoded_tokens, pred_decoded_tokens


# Rouge L

def rouge_l_tokenized(reference, prediction):
    """
    Tokenize both LaTeX strings with GPT-2, re-join with spaces,
    then compute the ROUGE-L F1 score on those token sequences.

    Returns:
      The ROUGE-L F1 score (0.0–1.0), rounded to 4 decimals.
    """
    # 1) Build a scorer for only ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # 2) Get GPT-2 decoded tokens
    _, ref_tokens  = get_gpt2_token_embeddings(reference)
    _, pred_tokens = get_gpt2_token_embeddings(prediction)

    # 3) Join into whitespace-delimited strings
    ref_for_rouge  = " ".join(ref_tokens)
    pred_for_rouge = " ".join(pred_tokens)

    # 4) Compute ROUGE-L
    scores = scorer.score(ref_for_rouge, pred_for_rouge)

    # 5) Return the F1 measure
    return round(scores["rougeL"].fmeasure, 4)

