from dataclasses import dataclass
from typing import List, Union
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer
from abc import abstractmethod
import torch
import copy


class ObservationFeaturizer(BaseObservationFeaturizer):

    @abstractmethod
    def init_on_reset(self, input_text: Union[List[str], str]):
        """
        Takes an input text (sentence) or list of token strings and featurizes it or prepares it
        This function would be called in env.reset()
        """
        raise NotImplementedError


@dataclass(init=True)
class Observation:
    current_input_str: str
    current_input_index: int
    current_action_history: List[str]

    def get_current_index(self):
        return self.current_input_index

    def get_current_input(self):
        return self.current_input_str

    def get_current_action_history(self) -> List[str]:
        return self.current_action_history

    def get_vector(self) -> torch.Tensor:
        return self.current_vector

    @classmethod
    def build(cls, input_index: int, input_str: str, action_history: List[str],
              observation_featurizer: ObservationFeaturizer) -> 'Observation':
        observation = Observation(input_str, input_index, action_history)
        observation.current_vector = observation_featurizer.featurize(observation)
        assert observation.get_vector().shape[0] == observation_featurizer.get_observation_dim()
        return observation

    def get_updated_observation(self, input_index: int, input_str: str, action: str, observation_featurizer: ObservationFeaturizer) -> 'Observation':
        updated_action_history = copy.deepcopy(self.current_action_history)
        updated_action_history.append(action)
        updated_observation = Observation.build(input_index, input_str, updated_action_history, observation_featurizer)
        return updated_observation
