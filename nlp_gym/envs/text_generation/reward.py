
from nlp_gym.envs.text_generation.observation import Observation
from abc import ABC, abstractclassmethod
from datasets import load_metric


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool) -> float:
        """
        An abstract class for reward functions for text generation

        Args:
            prev_observation (Observation): previous observation (s)
            action (int): last action performed (a)
            current_observation (Observation): observation after the action was performed (s')
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class MeteorRewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("meteor")

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool) -> float:
        if done:
            references = current_observation.target_or_reference_texts
            predicted = [current_observation.context_text]
            score = self._metric.compute(
                predictions=predicted, references=references)["meteor"]
            return score
        else:
            return 0


class MeteorRewardFunctionWithKL(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("meteor")

    def set_policy(self, policy):
        self._current_policy = policy

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool) -> float:
        # compute KL reward
        kl_reward = 1.0 * (1.0 -
                           self._current_policy.compute_ref_divergence(
                               prev_observation.to_dict(), action))

        # compute meteor at the end of episode
        if done:
            references = current_observation.target_or_reference_texts
            predicted = [current_observation.context_text]
            score = self._metric.compute(
                predictions=predicted, references=references)["meteor"]
            kl_reward = score + kl_reward
        return kl_reward
