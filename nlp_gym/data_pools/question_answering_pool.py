import random
from abc import abstractclassmethod
from dataclasses import dataclass
from typing import Dict, List

from sprl_package.data_pools.base import DataPool


@dataclass(init=True)
class Sample:
    id: str
    question: str
    facts: List[str]
    choices: Dict[str, str]
    answer: str


class QADataPool(DataPool):
    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @abstractclassmethod
    def prepare(cls, **args) -> 'QADataPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError
    
    def split(self, split_ratios: List[float]) -> List['QADataPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix]))
            start_ix = end_ix
        return pools
