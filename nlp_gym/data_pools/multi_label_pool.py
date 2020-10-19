from sprl_package.data_pools.base import DataPool, Sample
from typing import Tuple, List
from abc import abstractclassmethod
import random


class MultiLabelPool(DataPool):
    def __init__(self, samples: List[Sample], possible_labels: List[str], sample_weights: List[float]):
        self._samples, self._labels, self._weights = samples, possible_labels, sample_weights
        self.pool_length = len(self._samples)

    def __len__(self):
        return self.pool_length

    def __getitem__(self, ix: int) -> Tuple[Sample, float]:
        if ix >= self.pool_length:
            raise StopIteration
        sample = self._samples[ix]
        weight = self._weights[ix]
        return sample, weight

    def sample(self) -> Tuple[str, Sample]:
        random_sample = random.choice(self._samples)
        return random_sample

    def labels(self) -> List[str]:
        return self._labels

    @abstractclassmethod
    def prepare(cls, **args) -> 'MultiLabelPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List['MultiLabelPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(self.pool_length * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix], self._labels, self._weights[start_ix: end_ix]))
            start_ix = end_ix
        return pools
