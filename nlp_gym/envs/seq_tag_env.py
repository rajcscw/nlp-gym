from nlp_gym.envs.base_env import BaseEnv
from nlp_gym.envs.observation.observation import Observation
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.reward.base import RewardFunction
from nlp_gym.envs.reward.seq_tagging import EntityF1Score
from nlp_gym.envs.observation.base_featurizer import ObservationFeaturizer
from nlp_gym.envs.observation.seq_tagging import DefaultFeaturizerForSeqTagging
from nlp_gym.core_components.sampler import PrioritySampler
from nlp_gym.envs.action.action_space import ActionSpace
from typing import Tuple, List, Union
from dataclasses import dataclass
import copy
import numpy as np
from flair.data import Sentence
from rich import print


@dataclass(init=True)
class DataPoint:
    text: List[str]
    label: List[str]
    observation: Observation


class SeqTagEnv(BaseEnv):
    """
    A RL environment to tag sequences
    """
    def __init__(self, possible_labels: List[str], reward_function: RewardFunction = None,
                 observation_featurizer: ObservationFeaturizer = None, return_obs_as_vector: bool = True,
                 priority_scale: float = 0.0):
        self.sampler_for_replaying = PrioritySampler(priority_scale=priority_scale)
        self.current_sample: DataPoint = None

        # set action and observation spaces
        self.action_space = SeqTagEnv._get_action_space(possible_labels)

        # set up default reward
        reward_function = EntityF1Score(dense=True, average="micro") if reward_function is None else reward_function

        # set up default featurizer
        observation_featurizer = DefaultFeaturizerForSeqTagging(self.action_space) if observation_featurizer is None \
                                    else observation_featurizer
        super().__init__(None, reward_function, observation_featurizer, return_obs_as_vector)

        # set the counter
        self.time_step = None


    @staticmethod
    def _get_action_space(possible_labels: List[str]) -> ActionSpace:
        actions = copy.deepcopy(possible_labels)
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
        step_reward = self.reward_function(self.current_sample.observation, action_str, target_labels)

        # increment the time step
        self.time_step += 1

        done = self.time_step >= len(self.current_sample.text)

        # get the updated observation
        if not done:
            updated_observation = self.current_sample.observation.get_updated_observation(self.time_step,
                                                                                          action_str,
                                                                                          self.observation_featurizer)

            # update the current sample (just the observation)
            self.current_sample.observation = updated_observation
        else:
            self.current_sample.observation.current_action_history.append(action_str)

        # return observation, reward, done, info
        observation_to_return = self.current_sample.observation.get_vector().numpy() if self.return_obs_as_vector \
                                    else self.current_sample.observation
        return observation_to_return, step_reward, done, {}


    @staticmethod
    def _tokenize(text: str, label: List[str]) -> List[str]:
        sent = Sentence(text, use_tokenizer=False)
        tokens = [token.text for token in sent]

        assert len(tokens) == len(label), "Tokenization does not match with labels"
        return tokens

    def reset(self, sample: Sample = None) -> Union[Observation, np.array]:
        """
        Resets the environment and starts a new episode
        """
        # get a new document sample
        if sample is None:
            sample = self.sampler_for_replaying.sample(size=1)[0]
        self.current_original_sample = sample

        # tokenize the input text
        input_text_tokens = SeqTagEnv._tokenize(self.current_original_sample.input_text, sample.oracle_label)

        # we need to track time step
        self.time_step = 0

        # init the featurizer with the text
        self.observation_featurizer.init_on_reset(self.current_original_sample.input_text)

        # get initial observation
        observation = Observation.build(self.time_step, [], self.observation_featurizer)

        # construct current data point
        self.current_sample = DataPoint(text=input_text_tokens, label=sample.oracle_label,
                                        observation=observation)

        # return observation
        observation_to_return = self.current_sample.observation.get_vector().numpy() if self.return_obs_as_vector \
                                    else self.current_sample.observation
        return observation_to_return

    def close(self):
        pass

    def render(self):
        """
        Renders the current state of the environment
        """
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        history = [""] if self.time_step == 0 else self.current_sample.observation.get_current_action_history()
        for token, predicted_label in zip(self.current_sample.text[:self.time_step+1], history):
            print(f"[italic red]Token[/italic red]:{token} -> [italic red]Label[/italic red]: {predicted_label}")

    # Methods for online learning and sampling

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.sampler_for_replaying.add(sample, weight)

    def get_samples(self) -> List[Sample]:
        return self.sampler_for_replaying.get_all_samples()


if __name__ == "__main__":
    from sprl_package.data_pools.custom_seq_tagging_pools import CONLLNerTaggingPool

    train_pool = CONLLNerTaggingPool.prepare("train")

    env = SeqTagEnv(train_pool.labels(), return_obs_as_vector=False)
    for sample, weight in train_pool:
        if len(sample.oracle_label) < 10:
            env.add_sample(sample)

    env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
    print(f"Total Reward: "+str(total_reward))
    print(env.current_sample)