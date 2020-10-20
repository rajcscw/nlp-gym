from dataclasses import dataclass
from typing import List
from nlp_gym.envs.observation.base_featurizer import ObservationFeaturizer
import torch
import copy


@dataclass(init=True)
class Observation:
    current_input: torch.Tensor
    current_context: torch.Tensor
    current_action_history: List[str]

    def get_current_input(self) -> torch.Tensor:
        return self.current_input

    def get_current_index(self) -> int:
        return self.current_index

    def get_current_context(self) -> torch.Tensor:
        return self.current_context

    def get_current_action_history(self) -> List[str]:
        return self.current_action_history

    def get_vector(self) -> torch.Tensor:
        if self.current_context is not None:
            concatenated = torch.cat((self.current_input, self.current_context), dim=0)
        else:
            concatenated = self.current_input
        return concatenated

    @classmethod
    def build(cls, input_index: int, action_history: List[str],
              observation_featurizer: ObservationFeaturizer) -> 'Observation':
        input_vector = observation_featurizer.featurize_input_on_step(input_index)
        context_vector = observation_featurizer.featurize_context_on_step(action_history)
        observation = Observation(input_vector, context_vector, action_history)
        assert observation.get_vector().shape[0] == observation_featurizer.get_observation_dim()
        return observation

    def get_updated_observation(self, input_index: int, action: str, observation_featurizer: ObservationFeaturizer) -> 'Observation':
        updated_action_history = copy.deepcopy(self.current_action_history)
        updated_action_history.append(action)
        updated_observation = Observation.build(input_index, updated_action_history, observation_featurizer)
        return updated_observation
