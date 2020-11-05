from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from gym import spaces
from nlp_gym.data_pools.question_answering_pool import Sample
from nlp_gym.envs.common.base_env import BaseEnv
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer
from nlp_gym.envs.question_answering.observation import (Observation,
                                                         ObservationFeaturizer)
from nlp_gym.envs.question_answering.reward import BinaryRewardFunction
from rich import print


@dataclass(init=True)
class ObsTriplet:
    question: str
    facts: List[str]
    choice_id: str
    choice_text: str


class QAEnv(BaseEnv):
    """
    An environment for question answering with multiple choices and supporting facts

    Observation consists of triple of Question, Facts, Choice
    Actions are binary. Either to ANSWER OR CONTINUE

    ANSWER corresponds to answering with the choice in the observation
    CONTINUE correponds to agent asking for the next choice

    """
    def __init__(self, observation_featurizer: ObservationFeaturizer = None, reward_function: RewardFunction = None,
                 return_obs_as_vector: bool = True):
        # set action and observation spaces
        self.action_space = ActionSpace(actions=["ANSWER", "CONTINUE"])
        observation_featurizer = InformedFeaturizer() if observation_featurizer is None else observation_featurizer
        reward_function = BinaryRewardFunction() if reward_function is None else reward_function
        low = np.full(shape=(observation_featurizer.get_observation_dim(),), fill_value=-float('inf'), dtype=np.float32)
        high = np.full(shape=(observation_featurizer.get_observation_dim(),), fill_value=float('inf'), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        super().__init__(None, reward_function, observation_featurizer, return_obs_as_vector)

        # set the counter
        self.time_step = None

        # observation time line
        self.__observation_sequence = None
        self.__current_target = None
        self.__current_observation = None

        # hold samples
        self.__samples: List[Sample] = []

    def _is_terminal(self, action: str, time_step: int):
        termimal = (action == "ANSWER") or (time_step == len(self.__observation_sequence) - 1)
        return termimal

    def step(self, action: int) -> Tuple[Union[Observation, np.array], int, bool, dict]:

        # current action
        action_str = self.action_space.ix_to_action(action)

        # get reward
        reward = self.reward_function(self.__current_observation, action_str, self.__current_target)

        # terminal or not
        done = self._is_terminal(action_str, self.time_step)

        # if not done
        if not done:
            self.time_step += 1
            info = {}
        else:
            # populate the info field
            info = {"selected_choice": self.__observation_sequence[self.time_step].choice_id}

        # current observation
        observation_at_t = self.__observation_sequence[self.time_step]
        observation = Observation.build(observation_at_t.question, observation_at_t.facts,
                                        observation_at_t.choice_text, observation_at_t.choice_id,
                                        self.time_step, len(self.__observation_sequence),
                                        self.observation_featurizer)
        self.__current_observation = observation
        observation_to_return = observation.get_vector().numpy() if self.return_obs_as_vector else observation
        return observation_to_return, reward, done, info

    def reset(self, sample: Sample = None) -> Union[Observation, np.array]:
        # get a QA sample
        if sample is None:
            sample = np.random.choice(self.__samples)

        # create the observation sequence
        self.__observation_sequence = QAEnv._create_sequence(sample)

        # set the current target
        self.__current_target = sample.answer

        # time step
        self.time_step = 0

        # current observation
        observation_at_t = self.__observation_sequence[self.time_step]
        observation = Observation.build(observation_at_t.question, observation_at_t.facts,
                                        observation_at_t.choice_text, observation_at_t.choice_id,
                                        self.time_step, len(self.__observation_sequence),
                                        self.observation_featurizer)
        self.__current_observation = observation
        observation_to_return = observation.get_vector().numpy() if self.return_obs_as_vector else observation
        return observation_to_return

    @staticmethod
    def _create_sequence(sample: Sample) -> List[ObsTriplet]:
        sequences = []
        for choice_id, choice in sample.choices.items():
            triplet = ObsTriplet(sample.question, sample.facts, choice_id, choice)
            sequences.append(triplet)
        return sequences

    def render(self):
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        print(f"[italic red]Question[/italic red]: {self.__observation_sequence[0].question}")
        for obs in self.__observation_sequence[:self.time_step+1]:
            for fact in obs.facts:
                print(f"[italic red]Fact[/italic red]: {fact}")
            print(f"[italic red]Choice[/italic red] {obs.choice_id}: {obs.choice_text}")

    def close(self):
        pass

    # Methods for online learning and sampling

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.__samples.append(sample)

    def get_samples(self) -> List[Sample]:
        return self.__samples
