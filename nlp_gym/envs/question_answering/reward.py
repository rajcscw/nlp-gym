from nlp_gym.envs.common.observation import BaseObservation
from nlp_gym.envs.common.reward import RewardFunction


class BinaryRewardFunction(RewardFunction):
    def __call__(self, observation: BaseObservation, action: str, target: str) -> float:

        current_time_step = observation.get_current_time_step()
        total_time_steps = observation.get_total_steps()
        selected_choice = observation.get_last_choice()

        # if current action is ANSWER or ran out of input, then check the current choice and produce terminal reward
        if action == "ANSWER" or current_time_step == total_time_steps - 1:
            reward = 1.0 if selected_choice == target else 0.0
        else:
            reward = 0.0

        return reward
