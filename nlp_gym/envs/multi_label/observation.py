from dataclasses import dataclass
from typing import List, Union
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer
import torch
import copy
from abc import abstractmethod


class ObservationFeaturizer(BaseObservationFeaturizer):

    @abstractmethod
    def init_on_reset(self, input_text: Union[List[str], str]):
        """
        Takes an input text (sentence) or list of token strings and featurizes it or prepares it
        This function would be called in env.reset()
        """
        raise NotImplementedError


@dataclass(init=True)
class Observation(BaseObservation):
    current_input_str: str
    current_action_history: List[str]
    current_vector: torch.Tensor = None

    def get_current_input(self) -> str:
        return self.current_input_str

    def get_current_action_history(self) -> List[str]:
        return self.current_action_history

    def get_vector(self) -> torch.Tensor:
        return self.current_vector

    def get_dict_repr(self) -> dict:
        return {
            "current_input_str": self.current_input_str,
            "current_action_history": self.current_action_history
        }

    @classmethod
    def build(cls, input_str: str, action_history: List[str],
              observation_featurizer: ObservationFeaturizer,
              featurize: bool) -> 'Observation':
        observation = Observation(input_str, action_history)
        if featurize:
            observation.current_vector = observation_featurizer.featurize(observation)
            assert observation.get_vector().shape[0] == observation_featurizer.get_observation_dim()
        return observation

    def get_updated_observation(self, action: str, observation_featurizer: ObservationFeaturizer,
                                featurize: bool) -> 'Observation':
        updated_action_history = copy.deepcopy(self.current_action_history)
        updated_action_history.append(action)
        updated_observation = Observation.build(self.current_input_str, updated_action_history,
                                                observation_featurizer, featurize)
        return updated_observation
