import argparse

from nlp_gym.data_pools.custom_text_generation_pools import CommonGen, Sample
from nlp_gym.data_pools.text_generation_pool import TextGenPool
from nlp_gym.envs.text_generation.callback import KLRewardCallback
from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.observation import Observation
from nlp_gym.envs.text_generation.policy import LMActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from nlp_gym.envs.text_generation.test_reward import RewardIncreasingNumbers
from stable_baselines3.ppo.ppo import PPO
from transformers import AutoTokenizer


class TestTextGenPool(TextGenPool):
    @classmethod
    def prepare(cls, prompt: str):
        samples = [Sample(id=0,
                          prompt_or_input_text=prompt,  # a dummy prompt
                          references=[]
                          )]
        pool_instance = cls(samples)
        return pool_instance


def run(args):
    # reward function
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_fn = RewardIncreasingNumbers(
        tokenizer.eos_token, args.max_episode_length)

    data_pool = TestTextGenPool.prepare(tokenizer.eos_token)
    samples = [(sample, weight) for sample, weight in data_pool]

    # vectorized env for training
    train_env = make_vec_env(TextGenEnv,
                             n_envs=args.n_parallel_envs,
                             vec_env_cls=SubprocVecEnv,
                             env_kwargs={
                                 "reward_function": reward_fn,
                                 "max_text_length": args.max_prompt_length,
                                 "max_steps": args.max_episode_length,
                                 "tokenizer": tokenizer,
                                 "samples": samples
                             })

    # instantiate the PPO alg with the model
    alg = PPO(policy=LMActorCriticPolicy,
              env=train_env,
              policy_kwargs={
                  "model_name": model_name,
                  "apply_model_parallel": args.model_parallel,
              },
              n_steps=args.n_steps,
              batch_size=args.batch_size,
              verbose=1,
              learning_rate=args.lr,
              n_epochs=args.n_epochs,
              ent_coef=args.ent_coef)

    # callback to augment rewards with KL penalty
    kl_callback = KLRewardCallback(
        batch_size=args.batch_size, kl_coeff=args.kl_coeff)

    # train
    for i in range(int(args.n_iters)):
        alg.learn(args.n_parallel_envs * args.n_steps, callback=kl_callback)
        generate_text(alg.policy, tokenizer, data_pool.sample(),
                      args.max_episode_length)


def generate_text(policy: BasePolicy,
                  tokenizer: AutoTokenizer,
                  sample: Sample,
                  max_length: int):
    gen_kwargs = {"max_new_tokens": max_length}
    generated_text = policy.generate(tokenizer,
                                     sample.prompt_or_input_text,
                                     gen_kwargs)
    print(generated_text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine-tune LM to generate controlled text")
    parser.add_argument("--model_name", type=str,
                        default="distilgpt2", help="Name of the AutoModelForCausalLM")
    parser.add_argument("--max_prompt_length", type=int,
                        default=10, help="Maximum length of input/prompt")
    parser.add_argument("--max_episode_length", type=int,
                        default=10, help="Maximum length of the episode")
    parser.add_argument("--n_parallel_envs", type=int,
                        default=10, help="Number of parallel envs for rollout")
    parser.add_argument("--n_steps", type=int,
                        default=128, help="The number of steps to run for each environment per update for PPO")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="Batch size for PPO")
    parser.add_argument("--n_epochs", type=int,
                        default=5, help="Number of epochs for PPO")
    parser.add_argument("--lr", type=float,
                        default=1e-4, help="Learning rate")
    parser.add_argument("--ent_coef", type=float,
                        default=1e-2, help="Entropy Coefficient")
    parser.add_argument("--n_iters", type=int,
                        default=1e+2, help="Number of iterations of learn+eval")
    parser.add_argument("--kl_coeff", type=float,
                        default=1e-3, help="KL Coefficient")
    parser.add_argument("--model_parallel", type=bool,
                        default=True, help="Apply model parallel or not")
    args = parser.parse_args()
    run(args)
