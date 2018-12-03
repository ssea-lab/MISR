import math
import numpy as np


def precisionCalculator(outputs, ground_truths):
    '''
    Compute Precision for one
    - outputs: Model output (item id) iterable
    - ground_truths: Ground-Truth iterable
    '''
    output_set, ground_truth_set = set(outputs), set(ground_truths)
    tp = len(output_set.intersection(ground_truth_set))
    return tp / len(output_set)


def recallCalculator(outputs, ground_truths):
    '''
    Compute Recall for one
    - outputs: Model output (item id) iterable
    - ground_truths: Ground-Truth iterable
    '''
    output_set, ground_truth_set = set(outputs), set(ground_truths)
    tp = len(output_set.intersection(ground_truth_set))
    return tp / len(ground_truth_set)


def precisionRecallAndF1Calculator(outputs, ground_truths):
    '''
    Compute precision, recall and F1 for one in the same time
    - outputs: Model output (item id) iterable
    - ground_truths: Ground-Truth iterable
    - return (precision, recall, f1)
    '''
    output_set, ground_truth_set = set(outputs), set(ground_truths)
    tp = len(output_set.intersection(ground_truth_set))
    precision = tp / len(output_set)
    recall = tp / len(ground_truth_set)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def ndcgCalculator(ranked_outputs, ground_truths):
    '''
    Compute NDCG for one
    - ranked_outputs: Ranked Model output (item id) iterable
    - ground_truths: Ground-Truth iterable
    '''
    correct_answers = set(ground_truths)
    ndcg, idcg = 0.0, 0.0
    for rank, answer in enumerate(ranked_outputs):
        if answer in correct_answers:  # 只要在就好，不分大小
            ndcg += 1.0 / math.log(rank + 2, 2)

    for i in range(len(correct_answers)):
        idcg += 1.0 / math.log(i + 2, 2)

    return ndcg / idcg


def meanAveragePrecisionCalculator(ranked_outputs, ground_truths):
    '''
    Compute MAP for one
    - ranked_outputs: Ranked Model output (item id) iterable
    - ground_truths: Ground-Truth iterable    
    '''
    ap, hit = 0.0, 0.0
    correct_answers = set(ground_truths)
    for rank, answer in enumerate(ranked_outputs):
        if answer in correct_answers:
            hit += 1.0
            ap += hit / (rank + 1)
    return ap / len(correct_answers)


# 最基本的评价实现
def evaluate(ranked_outputs, ground_truths, topK):
    '''
    Evaluate one
    - rankded_outpus: Ranked Model output (item id) iterable
    - gound_truths: Ground-Truth iterable   
    return (precision, recall, f1, ndcg, map)
    '''
    o, t = ranked_outputs[:topK], ground_truths
    return (*precisionRecallAndF1Calculator(o, t), ndcgCalculator(o, t), meanAveragePrecisionCalculator(o, t))


def batchEvaluate(ranked_outputs, ground_truths, topK):
    '''
    Evaluate a batch of outputs
    - rankded_outpus: Iterable of Iterable
    - gound_truths: Iterable of Iterable
    return average (precision, recall, f1, ndcg, map)
    '''
    result = []
    for o, t in zip(ranked_outputs, ground_truths):
        o = o[:topK]
        result.append(
            (*precisionRecallAndF1Calculator(o, t), ndcgCalculator(o, t), meanAveragePrecisionCalculator(o, t)))
    return np.average(result, axis=0)
