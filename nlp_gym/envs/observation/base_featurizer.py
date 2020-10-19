from abc import ABC, abstractmethod
from typing import List, Union
import torch


class ObservationFeaturizer(ABC):

    @abstractmethod
    def init_on_reset(self, input_text: Union[List[str], str]):
        """
        Takes an input text (sentence) or list of token strings and featurizes it or prepares it
        This function would be called in env.reset()
        """
        raise NotImplementedError

    @abstractmethod
    def featurize_input_on_step(self, input_index: int) -> torch.Tensor:
        """
        Featurizes the given input for the current input

        Args:
            observation (Observation): observation to be embedded

        Returns:
            Observation: featurized observation
        """
        raise NotImplementedError

    def featurize_context_on_step(self, context: List[str]) -> torch.Tensor:
        """
        Featuries the context labels and returns a vector representation

        Returns:
            torch.Tensor: context representation
        """
        raise NotImplementedError

    def get_observation_dim(self) -> int:
        """
        Returns the observation dim
        """
        return self.get_input_dim() + self.get_context_dim()

    @abstractmethod
    def get_input_dim(self) -> int:
        """
        Returns the dimension of the input vector representation
        """
        raise NotImplementedError

    @abstractmethod
    def get_context_dim(self) -> int:
        """
        Returns the dimension of the context vector representation
        """
        raise NotImplementedError
