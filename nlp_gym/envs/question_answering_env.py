from nlp_gym.envs.base_env import BaseEnv
from nlp_gym.envs.observation.question_answering import ObservationFeaturizer, DefaultFeaturizerForQA, Observation
from nlp_gym.envs.action.action_space import ActionSpace
from nlp_gym.data_pools.question_answering_pool import Sample

from gym import spaces
import numpy as np
from typing import List, Union, Tuple
from dataclasses import dataclass
from rich import print

"""
TODO: Refactor to use abstract types and not concrete ones
"""


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
    def __init__(self, observation_featurizer: ObservationFeaturizer = None, return_obs_as_vector: bool = True):
        # set action and observation spaces
        self.action_space = ActionSpace(actions=["ANSWER", "CONTINUE"])
        observation_featurizer = DefaultFeaturizerForQA.from_fasttext() if observation_featurizer is None else observation_featurizer
        low = np.full(shape=(observation_featurizer.get_observation_dim(),), fill_value=-float('inf'), dtype=np.float32)
        high = np.full(shape=(observation_featurizer.get_observation_dim(),), fill_value=float('inf'), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        super().__init__(None, None, observation_featurizer, return_obs_as_vector)

        # set the counter
        self.time_step = None

        # observation time line
        self.__observation_sequence = None
        self.__current_target = None

        # hold samples
        self.__samples: List[Sample] = []

    def _reward_function(self, action: str, time_step: int):

        def _get_reward_for_selection(action: str, time_step: int):
            selected_choice = self.__observation_sequence[self.time_step].choice_id
            reward = +20 if selected_choice == self.__current_target else -10
            return reward

        # if final step has been reached and no more observations to give
        reward = None
        if time_step == len(self.__observation_sequence) - 1:
            if action == "CONTINUE":  # if the action is CONTINUE, then punish
                reward = -10
            else:  # if the action is ANSWER, then reward
                reward = _get_reward_for_selection(action, time_step)

        # if the action is ANSWER
        if action == "ANSWER":
            # if the choice is correct, then reward otherwise punish
            reward = _get_reward_for_selection(action, time_step)
        else:
            # if the action is CONTINUE and the correct choice is skipped,then punish
            # otherwise, if an incorrect choice is skipped, then reward
            current_choice = self.__observation_sequence[self.time_step].choice_id
            reward = -10 if current_choice == self.__current_target else +5

        return reward

    def _is_terminal(self, action: str, time_step: int):
        termimal = (action == "ANSWER") or (time_step == len(self.__observation_sequence) - 1)
        return termimal

    def step(self, action: int) -> Tuple[Union[Observation, np.array], int, bool, dict]:

        action_str = self.action_space.ix_to_action(action)

        # # get reward
        # reward = self._reward_function(action_str, self.time_step)

        # # done?
        # done = self._is_terminal(action_str, self.time_step)

        # if current action is ANSWER or ran out of input, then check the current choice and produce terminal reward
        if action_str == "ANSWER" or self.time_step == len(self.__observation_sequence) - 1:
            selected_choice = self.__observation_sequence[self.time_step].choice_id
            reward = 1.0 if selected_choice == self.__current_target else 0.0
            done = True
        else:  # Just continue with zero reward
            reward = 0.0
            done = False

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
                                        observation_at_t.choice_text, self.observation_featurizer)

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
                                        observation_at_t.choice_text, self.observation_featurizer)

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


if __name__ == "__main__":
    from sprl_package.data_pools.custom_question_answering_pools import AIRC

    # data
    data_pool = AIRC.prepare(split="train")

    # env
    env = QAEnv(return_obs_as_vector=False)

    # add samples to env
    for sample, _ in data_pool:
        env.add_sample(sample)

    # env reset
    observation = env.reset()

    # play an episode
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        print(f"Action: {env.action_space.ix_to_action(action)}")
        _, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Total reward: {total_reward}")
