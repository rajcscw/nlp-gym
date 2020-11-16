import pytest
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.multi_label.env import MultiLabelEnv


@pytest.fixture
def sample():
    sample = Sample(input_text="A sample text", oracle_label=["A", "B"])
    return sample


@pytest.fixture
def env():
    env = MultiLabelEnv(possible_labels=["A", "B", "C"], max_steps=3, return_obs_as_vector=False)
    yield env


def test_sequence_timeout(env, sample):
    observation = env.reset(sample)
    assert observation.get_current_input() == sample.input_text
    assert observation.get_current_action_history() == []
    observation, reward, done, info = env.step(action=0)
    assert observation.get_current_input() == sample.input_text
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0)]
    assert not done
    observation, reward, done, info = env.step(action=1)
    assert observation.get_current_input() == sample.input_text
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0),
                                                        env.action_space.ix_to_action(1)]
    assert not done
    observation, reward, done, info = env.step(action=1)
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0),
                                                        env.action_space.ix_to_action(1),
                                                        env.action_space.ix_to_action(1)]
    assert done


def test_sequence_terminate(env, sample):
    observation = env.reset(sample)
    assert observation.get_current_input() == sample.input_text
    assert observation.get_current_action_history() == []
    observation, reward, done, info = env.step(action=0)
    assert observation.get_current_input() == sample.input_text
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0)]
    assert not done
    observation, reward, done, info = env.step(action=env.action_space.action_to_ix("terminate"))
    assert observation.get_current_input() == sample.input_text
    assert observation.get_current_action_history() == [env.action_space.ix_to_action(0),
                                                        "terminate"]
    assert done
