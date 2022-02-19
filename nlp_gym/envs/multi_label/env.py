import copy
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from rich import print

from nlp_gym.core_components.sampler import PrioritySampler
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.base_env import BaseEnv
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.multi_label.featurizer import DefaultFeaturizerForMultiLabelRank
from nlp_gym.envs.multi_label.observation import ObservationFeaturizer, Observation
from nlp_gym.envs.multi_label.reward import F1RewardFunction


@dataclass(init=True)
class DataPoint:
    text: str
    label: List[str]
    observation: Observation


class MultiLabelEnv(BaseEnv):
    """
    A RL environment to generate multi-labels for given sample

    This environment can be used for multi-label classification and label ranking based on the provided reward function

    """

    def __init__(self, possible_labels: List[str], max_steps: int, reward_function: RewardFunction = None,
                 observation_featurizer: ObservationFeaturizer = None, return_obs_as_vector: bool = True,
                 return_obs_as_dict: bool = False, priority_scale: float = 0.0):
        self.sampler_for_replaying = PrioritySampler(
            priority_scale=priority_scale)
        self.current_sample: DataPoint = None

        # reward function
        reward_function = F1RewardFunction() if reward_function is None else reward_function

        # set action spaces
        self.action_space = MultiLabelEnv._get_action_space(possible_labels)

        # set observation spaces
        if return_obs_as_vector:
            observation_featurizer = DefaultFeaturizerForMultiLabelRank(self.action_space) if observation_featurizer is None \
                else observation_featurizer
        else:
            observation_featurizer = None
        super().__init__(max_steps, reward_function, observation_featurizer,
                         return_obs_as_vector, return_obs_as_dict)

        # set the counter
        self.time_step = None

    @staticmethod
    def _get_action_space(possible_labels: List[str]) -> ActionSpace:
        actions = copy.deepcopy(possible_labels)
        actions.append("terminate")  # default terminate action for episode
        action_space = ActionSpace(actions)
        return action_space

    def step(self, action: int) -> Tuple[Union[Observation, np.array], int, bool, dict]:
        """
        Takes a step with the given action and returns next observation
        Returns:
            Tuple[Observation, int, bool]: observation, reward, done
        """
        # current action
        action_str = self.action_space.ix_to_action(action)

        # target labels
        target_labels = copy.deepcopy(self.current_sample.label)

        # compute reward function
        step_reward = self.reward_function(
            self.current_sample.observation, action_str, target_labels)

        # increment the time step
        self.time_step += 1

        # get the updated observation
        updated_observation = self.current_sample.observation.get_updated_observation(
            action_str, self.observation_featurizer, self.return_obs_as_vector)

        # update the current sample (just the observation)
        self.current_sample.observation = updated_observation

        # return observation, reward, done, info
        observation_to_return = self._pack_observation(
            self.current_sample.observation)
        return observation_to_return, step_reward, self.is_terminal(action_str), {}

    def is_terminal(self, action_str: str):
        # terminate when "terminate" action is predicted or maximum steps elapsed
        return action_str == "terminate" or len(self.current_sample.observation.current_action_history) >= self.max_steps

    def reset(self, sample: Sample = None) -> Union[Observation, np.array]:
        """
        Resets the environment and starts a new episode
        """
        # get a new document sample
        if sample is None:
            sample = self.sampler_for_replaying.sample(size=1)[0]
        self.current_original_sample = sample

        # we need to track time step
        self.time_step = 0

        # init on reset
        if self.observation_featurizer is not None:
            self.observation_featurizer.init_on_reset(sample.input_text)

        # get observation
        observation = Observation.build(
            sample.input_text, [], self.observation_featurizer, self.return_obs_as_vector)

        # construct current data point
        self.current_sample = DataPoint(text=sample.input_text, label=sample.oracle_label,
                                        observation=observation)

        observation_to_return = self._pack_observation(
            self.current_sample.observation)

        return observation_to_return

    def close(self):
        pass

    def render(self):
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        print(
            f"[italic red]Input Text [/italic red]: {self.current_sample.text}")
        print(
            f"[italic red]Label Sequence[/italic red]: {self.current_sample.observation.get_current_action_history()}")

    # Methods for online learning and sampling
    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.sampler_for_replaying.add(sample, weight)

    def get_samples(self) -> List[Sample]:
        return self.sampler_for_replaying.get_all_samples()
