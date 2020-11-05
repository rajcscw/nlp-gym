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
    env = QAEnv()
    return env


def test_correct_answer(env, sample):
    observation = env.reset()
    assert observation.get_facts() == sample.facts
    assert observation.get_question() == sample.question
    assert observation.get_choice() == "A"


def test_incorrect_answer(env, sample):
    pass

def test_out_of_choices(env, sample):
    pass
