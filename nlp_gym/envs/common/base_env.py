from abc import abstractmethod
from typing import Tuple, List, Union
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.common.action_space import ActionSpace
from gym import spaces
import gym
import numpy as np


class BaseEnv(gym.Env):
    """
    A base class for all the environments
    """

    def __init__(self, max_steps: int, reward_function: RewardFunction,
                 observation_featurizer: BaseObservationFeaturizer, return_obs_as_vector: bool = True,
                 return_obs_as_dict: bool = False):
        """
        Args:
            max_steps (int): max steps for each episode
            reward_function (RewardFunction): reward function that computes scalar reward for each observation-action
            observation_featurizer (ObservationFeaturizer): a featurizer that vectorizes input and context of observation
            return_obs_as_vector (bool): return the observation as vector
            return_obs_as_dict (bool): return the observation as dict
        """
        self.max_steps = max_steps
        self.reward_function = reward_function
        self.return_obs_as_vector = return_obs_as_vector
        self.return_obs_as_dict = return_obs_as_dict
        assert (return_obs_as_vector and not return_obs_as_dict) or (
            not return_obs_as_vector and return_obs_as_dict), "Only one of return_obs_as_vector and return_obs_as_dict can be requested"
        self.set_featurizer(observation_featurizer)

    # Standard gym methods

    @abstractmethod
    def step(self, action: int) -> Tuple[Union[BaseObservation, np.array], int, bool, dict]:
        """
        Takes a step with the given action and returns (next state, reward, done, info)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, sample: Sample = None) -> Union[BaseObservation, np.array]:
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

    def get_action_space(self) -> ActionSpace:
        """
        Lists all possible actions indices and its meaning

        Returns:
            ActionSpace -- an instance of action space
        """
        return self.action_space

    def _pack_observation(self, observation: BaseObservation):
        observation_to_return = observation.get_vector(
        ).numpy() if self.return_obs_as_vector else observation

        observation_to_return = observation.get_dict_repr(
        ) if self.return_obs_as_dict and not self.return_obs_as_vector else observation_to_return
        return observation_to_return

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

    def set_featurizer(self, observation_featurizer: BaseObservationFeaturizer):
        """
        Sets the observation featurizer (can also change during run time)
        """
        self.observation_featurizer = observation_featurizer
        if observation_featurizer is not None:
            self._set_spaces(observation_featurizer)

    def _set_spaces(self, observation_featurizer: BaseObservationFeaturizer):
        low = np.full(shape=(observation_featurizer.get_observation_dim(),),
                      fill_value=-float('inf'), dtype=np.float32)
        high = np.full(shape=(observation_featurizer.get_observation_dim(
        ),), fill_value=float('inf'), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
