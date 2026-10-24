import re
import numpy as np

def normalize_answer(text):
    """
    Lowercase, remove punctuation, remove articles (a, an, the)
    """
    text = str(text).lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def exact_match(prediction, gold_answer):
    """
    Strict equality after normalization
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold_answer)
    return 1.0 if pred_norm == gold_norm else 0.0

def f1_score(prediction, gold_answer):
    """
    Token-level precision and recall
    """
    prediction = normalize_answer(prediction)
    gold_answer = normalize_answer(gold_answer)
    
    pred_tokens = set(prediction.split())
    gold_tokens = set(gold_answer.split())
    
    if len(gold_tokens) == 0:
        return 1.0 if len(pred_tokens) == 0 else 0.0
    
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def extract_number(text):
    # Simple extraction of first number
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
    if matches:
        return float(matches[0])
    return None

def numerical_accuracy(prediction, gold_answer, tolerance=0.01):
    """
    For numerical answers, check if within tolerance
    
    Args:
        tolerance: Relative error tolerance (1% by default)
    """
    try:
        pred_value = extract_number(prediction)
        gold_value = extract_number(gold_answer)
        
        if pred_value is None or gold_value is None:
             return exact_match(prediction, gold_answer)

        if gold_value == 0:
            return 1.0 if pred_value == 0 else 0.0
        
        relative_error = abs(pred_value - gold_value) / abs(gold_value)
        return 1.0 if relative_error <= tolerance else 0.0
    except:
        # If not a numerical answer, fall back to EM
        return exact_match(prediction, gold_answer)

def scaled_f1(prediction, gold_answer):
    """
    For numerical answers, compute F1 based on relative error
    Proposed in FinQA paper
    """
    try:
        pred_value = extract_number(prediction)
        gold_value = extract_number(gold_answer)
        
        if pred_value is None or gold_value is None:
             return 0.0

        if gold_value == 0:
            return 1.0 if pred_value == 0 else 0.0
        
        relative_error = abs(pred_value - gold_value) / abs(gold_value)
        
        # F1 decreases linearly with relative error
        f1 = max(0.0, 1.0 - relative_error)
        return f1
    except:
        return 0.0

def retrieval_precision_at_k(retrieved_cells, gold_cells, k):
    """
    What fraction of top-k retrieved cells are in gold set?
    
    Args:
        retrieved_cells: List of retrieved (table_id, row, col)
        gold_cells: Set of gold (table_id, row, col)
        k: Cutoff
    """
    top_k_retrieved = set(retrieved_cells[:k])
    correct = top_k_retrieved & gold_cells
    
    return len(correct) / k if k > 0 else 0.0

def retrieval_recall_at_k(retrieved_cells, gold_cells, k):
    """
    What fraction of gold cells are in top-k retrieved?
    """
    top_k_retrieved = set(retrieved_cells[:k])
    correct = top_k_retrieved & gold_cells
    
    return len(correct) / len(gold_cells) if gold_cells else 0.0

def mean_reciprocal_rank(retrieved_cells, gold_cells):
    """
    Average reciprocal rank of first gold cell
    """
    for rank, cell in enumerate(retrieved_cells, 1):
        if cell in gold_cells:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_cells, gold_cells, k):
    """
    Ranking quality metric accounting for position
    """
    dcg = 0.0
    gold_cells_set = set(gold_cells)
    
    for i, cell in enumerate(retrieved_cells[:k], 1):
        relevance = 1.0 if cell in gold_cells_set else 0.0
        dcg += relevance / np.log2(i + 1)
    
    # Ideal DCG (all gold cells retrieved first)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_cells), k)))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def attribution_precision_cells(predicted_cells, gold_cells):
    """
    Of predicted supporting cells, how many are correct?
    """
    if not predicted_cells:
        return 0.0
    
    correct = set(predicted_cells) & set(gold_cells)
    return len(correct) / len(predicted_cells)

def attribution_recall_cells(predicted_cells, gold_cells):
    """
    Of gold supporting cells, how many were predicted?
    """
    if not gold_cells:
        return 1.0  # Vacuously true if no gold cells
    
    correct = set(predicted_cells) & set(gold_cells)
    return len(correct) / len(gold_cells)

def attribution_f1_cells(predicted_cells, gold_cells):
    """
    Harmonic mean of precision and recall
    """
    precision = attribution_precision_cells(predicted_cells, gold_cells)
    recall = attribution_recall_cells(predicted_cells, gold_cells)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def bootstrap_significance_test(model_a_results, model_b_results, n_bootstrap=1000):
    """
    Test if Model A significantly outperforms Model B
    
    Returns:
        p_value: Probability that difference is due to chance
        confidence_interval: 95% CI for performance difference
    """
    n = len(model_a_results)
    differences = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        sample_a = [model_a_results[i] for i in indices]
        sample_b = [model_b_results[i] for i in indices]
        
        diff = np.mean(sample_a) - np.mean(sample_b)
        differences.append(diff)
    
    # Compute p-value (two-tailed)
    p_value = np.mean(np.array(differences) <= 0) * 2
    
    # 95% confidence interval
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return p_value, (ci_lower, ci_upper)

