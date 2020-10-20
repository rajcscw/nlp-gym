from typing import List
from nlp_gym.data_pools.base import Sample
from collections import defaultdict


def init_multiproc():
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass


def get_sample_weights(samples: List[Sample]) -> List[float]:
    def _to_str(labels: List[str]):
        return ";".join(labels)

    # get label frequencies
    label_seq_frequencies = defaultdict(int)
    for sample in samples:
        label_seq_frequencies[_to_str(sample.oracle_label)] += 1

    # compute sample weights as inverse of label freq
    weights = []
    for sample in samples:
        weight = 1 / label_seq_frequencies[_to_str(sample.oracle_label)]
        weights.append(weight)
    return weights
