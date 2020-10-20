from abc import abstractmethod
from typing import Tuple, List, Union
import torch
from nlp_gym.envs.observation.observation import Observation
from nlp_gym.envs.reward.base import RewardFunction
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.observation.base_featurizer import ObservationFeaturizer
from nlp_gym.envs.action.action_space import ActionSpace
from gym import spaces
import gym
import numpy as np


class BaseEnv(gym.Env):
    """
    A base class for all the environments
    """
    def __init__(self, max_steps: int, reward_function: RewardFunction,
                 observation_featurizer: ObservationFeaturizer, return_obs_as_vector: bool = True):
        """
        Args:
            max_steps (int): max steps for each episode
            reward_function (RewardFunction): reward function that computes scalar reward for each observation-action
            observation_featurizer (ObservationFeaturizer): a featurizer that vectorizes input and context of observation
            return_obs_vector (bool): return the observation as vector
        """
        self.max_steps = max_steps
        self.reward_function = reward_function
        self.return_obs_as_vector = return_obs_as_vector
        self.set_featurizer(observation_featurizer)

    # Standard gym methods

    @abstractmethod
    def step(self, action: int) -> Tuple[Union[Observation, np.array], int, bool, dict]:
        """
        Takes a step with the given action and returns (next state, reward, done, info)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, sample: Sample = None) -> Union[Observation, np.array]:
        """
        Resets the episode and returns an observation
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """
        Renders the current state of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    # Methods related to observation and action space infos

    def get_observation_dim(self) -> int:
        """
        Gets the observation dimension
        """
        return self.observation_featurizer.get_observation_dim()

    def get_input_dim(self) -> int:
        """
        Gets the dimension of input component of the observation
        """
        return self.observation_featurizer.get_input_dim()

    def get_action_space(self) -> ActionSpace:
        """
        Lists all possible actions indices and its meaning

        Returns:
            ActionSpace -- an instance of action space
        """
        return self.action_space

    # Additional methods for online learning and sampling

    @abstractmethod
    def add_sample(self, sample: Sample):
        """
        Adds annotated sample for sampling/replaying
        """
        raise NotImplementedError

    def get_samples(self) -> List[Sample]:
        """
        Returns list of samples available in the environment

        Returns:
            List[Sample]:  list of samples in the environment
        """
        raise NotImplementedError

    # For imitation learning, to get offline dataset of states and actions

    def get_offline_states_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns offline states and targets for action

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [description]
        """
        raise NotImplementedError

    def set_featurizer(self, observation_featurizer: ObservationFeaturizer):
        """
        Sets the observation featurizer (can also change during run time)
        """
        self.observation_featurizer = observation_featurizer
        self._set_spaces(observation_featurizer)

    def _set_spaces(self, observation_featurizer: ObservationFeaturizer):
        low = np.full(shape=(observation_featurizer.get_observation_dim(),), fill_value=-float('inf'), dtype=np.float32)
        high = np.full(shape=(observation_featurizer.get_observation_dim(),), fill_value=float('inf'), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
