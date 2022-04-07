from typing import Dict, Tuple, Optional, List

import torch
from gym import Env, spaces
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.discrete import Discrete
from nlp_gym.data_pools.text_generation_pool import Sample
from nlp_gym.envs.text_generation.reward import RewardFunction
from nlp_gym.envs.text_generation.observation import Observation
from transformers import AutoTokenizer
from nlp_gym.core_components.sampler import PrioritySampler


class TextGenEnv(Env):
    def __init__(self, tokenizer: AutoTokenizer,
                 reward_function: RewardFunction,
                 samples: Tuple[List[Sample], float],
                 max_steps: int = 512,
                 priority_scale: float = 0.0,
                 max_text_length: Optional[int] = None):
        """
        A generic RL environment to generate sequences.
        For eg: text generation, summarization, machine translation, text simplification

        Args:
            tokenizer (AutoTokenizer): hugging face tokenizer used for encoding and decoding text
            reward_function (RewardFunction): reward function
        """
        self._tokenizer = tokenizer
        self._reward_function = reward_function
        self._max_steps = max_steps
        self._max_text_length = max_text_length if max_text_length else self._tokenizer.model_max_length
        super().__init__()

        # set the observation and action space here
        self._vocab_size = len(tokenizer.vocab)
        self.observation_space = DictSpace({
            # we have to provide fixed sized inputs (padded) because sb3 support for DictObsersevation is limited
            # while creating rollout buffers, observations are concatenated for each key
            "prompt_or_input_encoded_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length,)),
            "prompt_or_input_attention_mask_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length,)),
            "context_encoded_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_steps,)),
            "context_attention_mask_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_steps,)),
            "input_encoded_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length+self._max_steps,)),
            "input_attention_mask_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length+self._max_steps,)),
        })
        self.action_space = Discrete(n=self._vocab_size)
        self.sampler_for_replaying = PrioritySampler(
            priority_scale=priority_scale)
        for sample, weight in samples:
            self.sampler_for_replaying.add(sample, weight)

        # check the tokenizer and add padding tokens
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"  # TBD: configure this
        self._tokenizer.truncation_side = "left"

        # init tracking variables
        self.__current_sample = None
        self.__current_obs = None
        self.__time_step = None

    def step(self, action: int) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        self.__time_step += 1

        # previous obs
        previous_obs = self.__current_obs

        # just update the context tensor and gets the new observation
        self.__current_obs = self.__current_obs.update(action, self._tokenizer)

        # decide if the episode is finished or not
        done = (action == self._tokenizer.eos_token_id or self.__time_step ==
                self._max_steps)

        # compute reward
        reward = None if self._reward_function is None else self._reward_function(
            previous_obs, action, self.__current_obs, done)

        # populate additional info
        info = {
            "output": self.__current_obs.context_text,
            "action_history": self.__current_obs.action_history
        }

        return self.__current_obs.to_dict(), reward, done, info

    def reset(self, sample: Sample = None) -> Dict[str, torch.tensor]:
        """
        Resets the environment and starts a new episode
        """
        # gets a new sample if not provided
        if sample is None:
            sample = self.sampler_for_replaying.sample(size=1)[0]
        self.__current_sample = sample

        # init the observation
        self.__current_obs = Observation.init_from_sample(sample, self._tokenizer,
                                                          self._max_text_length, self._max_steps)

        # start the time step counter
        self.__time_step = 0

        dict_observation = self.__current_obs.to_dict()
        return dict_observation

    def render(self):
        pass

    def close(self):
        pass

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.sampler_for_replaying.add(sample, weight)
