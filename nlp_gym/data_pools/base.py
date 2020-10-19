from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass(init=True)
class Sample:
    """
    Dataclass for holding datapoints

    Attributes:
        input_text - textual input
        oracle_label - true label for the given data point
    """
    input_text: str
    oracle_label: List[str]


class DataPool(ABC):
    """Abstract Data Pool containing samples
    """
    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the data pool
        """
        pass

    @abstractmethod
    def __getitem__(self, ix: int) -> Sample:
        """Returns one item in the data pool
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """Returns a random labeled data point (eg. for instance, playing an episode of RL)
        """
        raise NotImplementedError
