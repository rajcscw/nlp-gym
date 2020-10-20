from typing import List
import copy


def compute_match(predicted_labels: List[str], true_labels: List[str]) -> float:
    """Computes exact match for multi-label envs

    Args:
        predicted_labels (List[str]): predicted labels
        true_labels (List[str]): true labels

    Returns:
        float - match ratio
    """
    true_labels = copy.deepcopy(true_labels)
    predicted_labels = copy.deepcopy(predicted_labels)

    # remove "terminate" for match computation
    if "terminate" in true_labels:
        true_labels.remove("terminate")
    if "terminate" in predicted_labels:
        predicted_labels.remove("terminate")

    # bring both sequence to same length
    n_targets = len(true_labels)
    n_predicted = len(predicted_labels)
    max_seq_length = max(n_targets, n_predicted)

    # pad predicted labels
    filled_predicted_labels = ["none"] * max_seq_length
    filled_predicted_labels[:n_predicted] = predicted_labels

    # pad target labels
    filled_target_labels = ["none"] * max_seq_length
    filled_target_labels[:n_targets] = true_labels

    n_matched = 0
    for true, predicted in zip(filled_target_labels, filled_predicted_labels):
        if true == predicted:
            n_matched += 1
    match_ratio = n_matched / max_seq_length
    return match_ratio


def compute_hamming_score(predicted_labels: List[str], true_labels: List[str]) -> float:
    true_labels = copy.deepcopy(true_labels)
    predicted_labels = copy.deepcopy(predicted_labels)

    # remove "terminate" for match computation
    if "terminate" in true_labels:
        true_labels.remove("terminate")
    if "terminate" in predicted_labels:
        predicted_labels.remove("terminate")

    pred_labels = set(predicted_labels)
    true_labels = set(true_labels)
    n_inter = len(set(pred_labels).intersection(true_labels))

    # how many labels in true labels match the predicted labels
    # note: we do not use the unique ones here; so that we penalize if they ramble
    denominator = max(len(predicted_labels), len(true_labels))
    hamming_score = n_inter/denominator
    return hamming_score
