from typing import Dict, Tuple, Optional

import torch
from gym import Env, spaces
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.discrete import Discrete
from nlp_gym.data_pools.text_generation_pool import Sample
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.text_generation.observation import Observation
from transformers import AutoTokenizer
from nlp_gym.core_components.sampler import PrioritySampler


class TextGenEnv(Env):
    def __init__(self, tokenizer: AutoTokenizer,
                 reward_function: RewardFunction,
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

        # check the tokenizer and add padding tokens

        # set the observation and action space here
        self._vocab_size = len(tokenizer.vocab)
        self.observation_space = DictSpace({
            # # we have to provide fixed sized inputs (padded) because sb3 support for DictObsersevation is limited
            # and while creating rollout buffers, observations are concatenated for each key
            "prompt_or_input_encoded_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length,)),
            "prompt_or_input_attention_mask_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length,)),
            "context_encoded_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length,)),
            "context_attention_mask_pt": spaces.Box(low=0, high=self._vocab_size, shape=(self._max_text_length,)),
        })
        self.action_space = Discrete(n=self._vocab_size)
        self.sampler_for_replaying = PrioritySampler(
            priority_scale=priority_scale)

        # init tracking variables
        self.__current_sample = None
        self.__current_obs = None
        self.__time_step = None

    def step(self, action: int) -> Tuple[Dict[str, torch.tensor], int, bool, dict]:
        self.__time_step += 1

        # just update the context tensor and gets the new observation
        self.__current_obs = self.__current_obs.update(action, self._tokenizer)

        # compute reward
        reward = None if self._reward_function is None else self._reward_function()

        # decide if the episode is finished or not
        done = (action == self._tokenizer.eos_token_id or self.__time_step ==
                self._max_steps)

        # populate additional info
        info = {
            "output": self.__current_obs.decoded_context,
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

        self.__current_obs = Observation.init_from_sample(sample)

        # tokenize the input text
        tokenized_input_pt = self._tokenizer.encode(
            sample.prompt_or_input_text, return_tensors="pt")

        # init the context (to BOS token - does it apply for all tokenizers?)
        context_pt = torch.tensor([self._tokenizer.bos_token_id])

        # start the time step counter
        self.__time_step = 0

        # initialize the observation
        self.__current_obs = Observation(prompt_or_input_text=sample.prompt_or_input_text,
                                         prompt_or_input_pt=tokenized_input_pt,
                                         context_pt=context_pt,
                                         decoded_context=None,
                                         max_text_length=self._max_text_length)
        dict_observation = self.__current_obs.to_dict()
        return dict_observation

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    env = TextGenEnv(tokenizer, reward_function=None, max_steps=10)

    sample = Sample(1, "Hello", ["Hello, who is this"])

    # play an episode
    obs = env.reset(sample)
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, info = env.step(action)
    print(info)

    # Implement render function - later
    # Verify step and reset - done
    # Run with random policy - done
    # RUn with gpt policy pretrained - TBD
    # Implement datapool - TBD
    # Implement unit tests
    # Implement
