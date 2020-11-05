import pytest
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.seq_tagging.env import SeqTagEnv


@pytest.fixture
def sample():
    sample = Sample(input_text="Token_A Token_B Token_C", oracle_label=["A", "B", "A"])
    return sample


@pytest.fixture
def env():
    env = SeqTagEnv(possible_labels=["A", "B", "C"], return_obs_as_vector=False)
    return env


def test_sequence(env, sample):
    observation = env.reset(sample)
    assert observation.get_current_input() == "Token_A"
    assert observation.get_current_action_history() == []
    observation, reward, done, info = env.step(action=0)
    assert observation.get_current_input() == "Token_B"
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0)]
    assert not done
    observation, reward, done, info = env.step(action=1)
    assert observation.get_current_input() == "Token_C"
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0),
                                                        env.action_space.ix_to_action(1)]
    assert not done
    observation, reward, done, info = env.step(action=1)
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0),
                                                        env.action_space.ix_to_action(1),
                                                        env.action_space.ix_to_action(1)]
    assert done
