from dataclasses import dataclass
from abc import abstractmethod
import torch
from typing import List
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer


@dataclass(init=True)
class Observation(BaseObservation):
    question: str
    facts: List[str]
    choice: str
    choice_id: str
    time_step: int
    total_steps: int
    input_embedding: torch.tensor = None

    def get_vector(self) -> torch.Tensor:
        return self.input_embedding

    def get_question(self) -> str:
        return self.question

    def get_facts(self) -> List[str]:
        return self.facts

    def get_choice(self) -> str:
        return self.choice

    def get_last_choice(self) -> str:
        return self.choice_id

    def get_current_time_step(self) -> int:
        return self.time_step

    def get_total_steps(self) -> int:
        return self.total_steps

    @classmethod
    def build(cls, question: str, facts: List[str], choice: str, choice_id: str,
              time_step: int, total_steps: int,
              observation_featurizer: 'ObservationFeaturizer', featurize: bool) -> 'Observation':
        observation = Observation(question, facts, choice, choice_id, time_step, total_steps)
        if featurize:
            observation.input_embedding = observation_featurizer.featurize(observation)
        return observation


class ObservationFeaturizer(BaseObservationFeaturizer):
    @abstractmethod
    def init_on_reset(self, question: str, facts: List[str]):
        """
        Initialize with question and facts on env.reset()

        Args:
            question (str): a question
            facts (List[str]): facts
        """
        raise NotImplementedError
