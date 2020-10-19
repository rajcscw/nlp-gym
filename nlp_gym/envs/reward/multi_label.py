from sprl_package.envs.observation.observation import Observation
from sprl_package.envs.reward.base import RewardFunction
from sprl_package.metrics.multi_label import F1Score
from typing import List
import copy


class F1RewardFunction(RewardFunction):
    """
    Computes F1 score between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1)
    """
    def __init__(self):
        self.scoring_fn = F1Score()

    def __call__(self, observation: Observation, action: str, targets: List[str]) -> float:
        # get previous action sequence
        prev_action_history = observation.get_current_action_history()

        # get current action sequence
        current_action_history = copy.deepcopy(prev_action_history)
        current_action_history.append(action)

        # remove "terminate" to compute reward
        if "terminate" in current_action_history:
            current_action_history.remove("terminate")

        # step reward as change in the scores
        # as good actions lead to increase in the scores
        previous_score = self.scoring_fn(targets, prev_action_history)
        current_score = self.scoring_fn(targets, current_action_history)
        reward = current_score - previous_score

        return reward
