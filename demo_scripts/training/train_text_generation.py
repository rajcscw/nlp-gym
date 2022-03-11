from nlp_gym.data_pools.custom_text_generation_pools import CommonGen
from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.reward import MeteorRewardFunction
from stable_baselines3.common.env_checker import check_env
from transformers import AutoTokenizer

# data pool
data_pool = CommonGen.prepare(split="train")

# reward function
reward_fn = MeteorRewardFunction()

# text generation env
tokenizer = AutoTokenizer.from_pretrained("gpt2")
env = TextGenEnv(tokenizer, reward_fn)
for sample, weight in data_pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)
