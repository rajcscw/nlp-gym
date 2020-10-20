from nlp_gym.envs.observation.observation import Observation
from nlp_gym.envs.reward.base import RewardFunction
from typing import List
import copy


class MeanReciprocalRankFunction(RewardFunction):
    """
    Computes MRR between predicted and target labels
    It is a step-wise reward function
    Reward at time step t = score(t) - score(t-1)
    """

    @staticmethod
    def _calc_mrr(target_labels: List[str], predicted_labels: List[str]):
        mrr = 0.0
        for target_label in target_labels:
            # find the index
            index = predicted_labels.index(target_label) if target_label in predicted_labels else None
            rr = 1 / (index+1) if index is not None else 0.0
            mrr += rr
        mrr = mrr / len(target_labels)
        return mrr

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
        previous_score = MeanReciprocalRankFunction._calc_mrr(targets, prev_action_history)
        current_score = MeanReciprocalRankFunction._calc_mrr(targets, current_action_history)
        reward = current_score - previous_score
        return reward
