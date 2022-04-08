from nlp_gym.data_pools.custom_text_generation_pools import CommonGen, Sample, Xsum, CNNDailyMail
from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.reward import MeteorRewardFunctionWithKL, MeteorRewardFunction
from stable_baselines3.common.env_checker import check_env
from transformers import AutoTokenizer
from stable_baselines3.ppo.ppo import PPO
from nlp_gym.envs.text_generation.policy import LMActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from rich import print


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
    print(info["output"])


if __name__ == "__main__":

    # data pool
    data_pool = CNNDailyMail.prepare(split="train")
    samples = [(sample, weight) for sample, weight in data_pool]

    # reward function
    model_name = "distilgpt2"
    train_reward_fn = MeteorRewardFunctionWithKL()
    eval_reward_fn = MeteorRewardFunction()

    # text generation env
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_env = TextGenEnv(tokenizer, eval_reward_fn,
                           max_text_length=500, max_steps=100,
                           samples=samples)
    eval_env = TextGenEnv(tokenizer, eval_reward_fn,
                          max_text_length=500, max_steps=100,
                          samples=samples)

    # instantiate the PPO alg with the model
    model = PPO(policy=LMActorCriticPolicy, env=train_env, policy_kwargs={
        "model_name": model_name,
        "apply_model_parallel": True,
    }, n_steps=200, batch_size=32, verbose=1, learning_rate=1e-6, n_epochs=5, ent_coef=1e-2)
    train_reward_fn.set_policy(model.policy)

    run_episode(model, eval_env, data_pool)

    # train
    for i in range(500):
        model.learn(200)
        run_episode(model, eval_env, data_pool)
