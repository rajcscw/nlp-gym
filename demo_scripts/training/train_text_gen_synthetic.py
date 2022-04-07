from nlp_gym.data_pools.custom_text_generation_pools import CommonGen, Sample
from nlp_gym.data_pools.text_generation_pool import TextGenPool
from nlp_gym.envs.text_generation.env import TextGenEnv
from transformers import AutoTokenizer
from stable_baselines3.ppo.ppo import PPO
from nlp_gym.envs.text_generation.policy import LMActorCriticPolicy
from nlp_gym.envs.text_generation.reward import RewardFunction
from nlp_gym.envs.text_generation.observation import Observation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


class TestTextGenPool(TextGenPool):
    @classmethod
    def prepare(cls, prompt: str):
        samples = [Sample(id=f"{id}",
                          prompt_or_input_text=f"{id}",  # a dummy prompt
                          references=[]
                          ) for id in range(int(1e+2))]
        pool_instance = cls(samples)
        return pool_instance


class RewardIncreasingNumbers(RewardFunction):
    def __init__(self, eos_token) -> None:
        super().__init__()
        self.eos_token = eos_token

    def is_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool) -> float:
        if done:
            tokens = current_observation.action_history
            if self.eos_token in tokens:
                tokens.remove(self.eos_token)
            number_tokens = [float(token)
                             for token in tokens if self.is_number(token)]
            if len(number_tokens) > 2:  # must contain atleast 2 numbers
                # then we check how many numbers are in the sorted order
                sorted_count = 1
                previous_token = number_tokens[0]
                for token in number_tokens[1:]:
                    if token > previous_token:
                        sorted_count += 1
                        previous_token = token
                    else:
                        break
                return (sorted_count/len(tokens)) + 1e-4

        return 0.0


def run_episode(model: PPO,
                env: TextGenEnv,
                data_pool: CommonGen,
                sample: Sample = None):
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


if __name__ == "__main__":
    # reward function
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_fn = RewardIncreasingNumbers(tokenizer.eos_token)

    # data pool
    data_pool = TestTextGenPool.prepare(tokenizer.bos_token)
    samples = [(sample, weight) for sample, weight in data_pool]

    # text generation env
    eval_env = TextGenEnv(tokenizer, reward_fn,
                          max_text_length=10, max_steps=10, samples=samples)
    for sample, weight in data_pool:
        eval_env.add_sample(sample, weight)

    # vectorized env for training
    n_envs = 10
    n_steps = 128
    train_env = make_vec_env(TextGenEnv,
                             n_envs=n_envs,
                             vec_env_cls=SubprocVecEnv,
                             env_kwargs={
                                 "reward_function": reward_fn,
                                 "max_text_length": 10,
                                 "max_steps": 10,
                                 "tokenizer": tokenizer,
                                 "samples": samples
                             })

    # instantiate the PPO alg with the model
    model = PPO(policy=LMActorCriticPolicy, env=train_env, policy_kwargs={
        "model_name": model_name,
    }, n_steps=n_steps, batch_size=64, verbose=1, learning_rate=1e-5, n_epochs=10, ent_coef=1e-2)

    run_episode(model,  eval_env, data_pool)

    # train
    for i in range(500):
        model.learn(n_envs * n_steps)
        run_episode(model,  eval_env, data_pool)
