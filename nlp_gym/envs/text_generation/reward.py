
from nlp_gym.envs.text_generation.observation import Observation
from abc import ABC, abstractclassmethod
from datasets import load_metric


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(self, current_observation: Observation,
                 action: int,
                 next_observation: Observation,
                 done: bool) -> float:
        """
        An abstract class for reward functions for text generation

        Args:
            current_observation (Observation): previous observation (s)
            action (int): action performed (a) at s
            next_observation (Observation): observation after the action was performed (s')
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class MeteorRewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("meteor")

    def __call__(self, current_observation: Observation,
                 action: int,
                 next_observation: Observation,
                 done: bool) -> float:

        # compute meteor at the end of episode
        if done:
            references = next_observation.target_or_reference_texts
            predicted = [next_observation.context_text]
            score = self._metric.compute(
                predictions=predicted, references=references)["meteor"]
            return score
        return 0
