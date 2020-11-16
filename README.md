# NLPGym [![CircleCI](https://circleci.com/gh/rajcscw/nlp-gym/tree/main.svg?style=svg)](https://circleci.com/gh/rajcscw/nlp-gym/tree/main)

NLPGym is a toolkit to bridge the gap between applications of RL and NLP. This aims at facilitating research and benchmarking of DRL application on natural language processing tasks. 

The  toolkit provides interactive environments for standard NLP tasks such as sequence tagging, question answering, and sequence classification. The environments provide standard RL interfaces and therefore can be used together with most RL frameworks such as [baselines](https://github.com/openai/baselines), [stable-baselines](https://github.com/hill-a/stable-baselines), and [RLLib](https://github.com/ray-project/ray). 

Furthermore, the toolkit is designed in a modular fashion providing flexibility for users to extend tasks with their custom data sets, observations, and reward functions.


This work will be presented at [Wordplay: When Language Meets Games @ NeurIPS 2020](https://wordplay-workshop.github.io/)

## Install
```
git clone https://github.com/rajcscw/nlp-gym.git
cd nlp-gym
pip install .
```

To install all the dependencies for using demo scripts: 
```
pip install .["demo"]
```

## Usage

The environments follow standard gym interface and following script demonstrates a question answering environment with a random action-taking agent.


```python
from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv

# data pool
pool = QASC.prepare("train")

# question answering env
env = QAEnv()
for sample, weight in pool:
    env.add_sample(sample)

# play an episode
done = False
state = env.reset()
total_reward = 0
while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    print(f"Action: {env.action_space.ix_to
```

To train a DQN agent for the same task:

```python
from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env


# data pool
data_pool = QASC.prepare(split="train")
val_pool = QASC.prepare(split="val")

# featurizer
featurizer = InformedFeaturizer()

# question answering env
env = QAEnv(observation_featurizer=featurizer)
for sample, weight in data_pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)

# train a MLP Policy
model = DQN(env=env, policy=DQNPolicy, gamma=0.99, batch_size=32, learning_rate=1e-4,
            double_q=True, exploration_fraction=0.1,
            prioritized_replay=False, policy_kwargs={"layers": [64, 64]},
            verbose=1)
model.learn(total_timesteps=int(1e+4))
```

Further examples to train agents for other tasks can be found in [demo scripts](https://github.com/rajcscw/nlp-gym/tree/main/demo_scripts)
