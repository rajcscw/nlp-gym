from nlp_gym.envs.common.observation import BaseObservation
from nlp_gym.metrics.seq_tag import EntityScores
from nlp_gym.envs.reward.base import RewardFunction
from typing import List
import copy


class EntityF1Score(RewardFunction):
    """
    Computes micro f1 score between predicted and target as the final reward
    """
    def __init__(self, dense: bool, average: str):
        self.dense = dense
        self.average = average
        self.reward_fn = self._dense if self.dense else self._sparse

    def _sparse(self, targets: List[str], prev_action_history: List[str], current_action_history: List[str]) -> float:
        if len(current_action_history) >= len(targets):
            reward = EntityScores(average=self.average)(targets, current_action_history)["f1"]
        else:
            reward = 0.0
        return reward

    def _dense(self, targets: List[str], prev_action_history: List[str], current_action_history: List[str]) -> float:
        # we compute reward as the change in the score
        # step reward as change in the scores
        # as good actions lead to increase in the scores
        previous_score = EntityScores(average=self.average)(targets[:len(prev_action_history)], prev_action_history)["f1"]
        current_score = EntityScores(average=self.average)(targets[:len(current_action_history)], current_action_history)["f1"]
        reward = current_score - previous_score
        return reward

    def __call__(self, observation: BaseObservation, action: str, targets: List[str]) -> float:
        # get previous and current actions
        prev_action_history = observation.get_current_action_history()
        current_action_history = copy.deepcopy(observation.get_current_action_history())
        current_action_history.append(action)
        reward = self.reward_fn(targets, prev_action_history, current_action_history)
        return reward