from nlp_gym.data_pools.custom_question_answering_pools import AIRC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
import tqdm


def eval_model(env, model, pool):
    correctly_answered = 0.0
    for sample, _ in tqdm.tqdm(pool, desc="Evaluating"):
        obs = env.reset(sample)
        state = None
        done = False
        while not done:
            action, state = model.predict(obs)
            obs, reward, done, info = env.step(action)

        if info["selected_choice"] == sample.answer:
            correctly_answered += 1

    return correctly_answered/len(pool)


# data pool
data_pool = AIRC.prepare(split="train", dataset_id=AIRC.EASY)
val_pool = AIRC.prepare(split="val", dataset_id=AIRC.EASY)

# featurizer
featurizer = SimpleFeaturizer()

# seq tag env
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
for i in range(int(1e+3)):
    model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False)
    print(eval_model(env, model, val_pool))