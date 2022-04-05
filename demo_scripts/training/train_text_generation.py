from nlp_gym.data_pools.custom_text_generation_pools import CommonGen, Sample, Xsum
from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.reward import MeteorRewardFunctionWithKL
from stable_baselines3.common.env_checker import check_env
from transformers import AutoTokenizer
from stable_baselines3.ppo.ppo import PPO
from nlp_gym.envs.text_generation.policy import LMActorCriticPolicy


def run_episode(model: PPO, env: TextGenEnv, data_pool: CommonGen, sample: Sample = None):
    if not sample:
        sample = data_pool.sample()
    print(sample)
    obs = env.reset(sample)
    done = False
    state = None
    while not done:
        action, state = model.predict(obs, state)
        obs, reward, done, info = env.step(action)
    print(reward)
    print(info)


# data pool
data_pool = Xsum.prepare(split="train")

# reward function
model_name = "distilgpt2"
reward_fn = MeteorRewardFunctionWithKL()

# text generation env
tokenizer = AutoTokenizer.from_pretrained(model_name)
env = TextGenEnv(tokenizer, reward_fn, max_text_length=500, max_steps=100)
for sample, weight in data_pool:
    env.add_sample(sample, weight)

# test sample
test_sample = data_pool.sample()

# instantiate the PPO alg with the model
model = PPO(policy=LMActorCriticPolicy, env=env, policy_kwargs={
    "model_name": model_name,
}, n_steps=128, batch_size=16, verbose=1, learning_rate=1e-7, n_epochs=5)
reward_fn.set_policy(model.policy)


run_episode(model,  env, data_pool, test_sample)

# train
for i in range(500):
    model.learn(256)
    run_episode(model,  env, data_pool)
