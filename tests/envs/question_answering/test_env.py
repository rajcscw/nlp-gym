import pytest
from nlp_gym.data_pools.question_answering_pool import Sample
from nlp_gym.envs.question_answering.env import QAEnv


@pytest.fixture
def sample():
    sample = Sample(id="test", question="Question?", facts=["Fact A", "Fact B"],
                    choices={"A": "Answer text A", "B": "Answer text B", "C": "Answer text C"},
                    answer="B")
    return sample


@pytest.fixture
def env():
    env = QAEnv(return_obs_as_vector=False)
    yield env


def test_correct_answer(env, sample):
    observation = env.reset(sample)
    assert observation.get_facts() == sample.facts
    assert observation.get_question() == sample.question
    assert observation.get_choice() == "Answer text A"
    observation, reward, done, info = env.step(env.action_space.action_to_ix("CONTINUE"))
    assert observation.get_facts() == sample.facts
    assert observation.get_question() == sample.question
    assert observation.get_choice() == "Answer text B"
    assert reward == 0.0
    observation, reward, done, info = env.step(env.action_space.action_to_ix("ANSWER"))
    assert reward == 1.0
    assert done


def test_incorrect_answer(env, sample):
    observation = env.reset(sample)
    assert observation.get_facts() == sample.facts
    assert observation.get_question() == sample.question
    assert observation.get_choice() == "Answer text A"
    observation, reward, done, info = env.step(env.action_space.action_to_ix("ANSWER"))
    assert reward == 0.0
    assert done


def test_out_of_choices(env, sample):
    observation = env.reset(sample)
    assert observation.get_facts() == sample.facts
    assert observation.get_question() == sample.question
    assert observation.get_choice() == "Answer text A"
    observation, reward, done, info = env.step(env.action_space.action_to_ix("CONTINUE"))
    assert observation.get_facts() == sample.facts
    assert observation.get_question() == sample.question
    assert observation.get_choice() == "Answer text B"
    assert not done
    assert reward == 0.0
    observation, reward, done, info = env.step(env.action_space.action_to_ix("CONTINUE"))
    assert reward == 0.0
    assert not done
    observation, reward, done, info = env.step(env.action_space.action_to_ix("CONTINUE"))
    assert reward == 0.0
    assert done
