from nlp_gym.data_pools.custom_text_generation_pools import CommonGen, Sample
from nlp_gym.data_pools.text_generation_pool import TextGenPool
from nlp_gym.envs.text_generation.env import TextGenEnv
from transformers import AutoTokenizer
from stable_baselines3.ppo.ppo import PPO
from nlp_gym.envs.text_generation.policy import LMActorCriticPolicy
from nlp_gym.envs.text_generation.reward import RewardFunction
from nlp_gym.envs.text_generation.observation import Observation
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import BasePolicy
import argparse


class TestTextGenPool(TextGenPool):
    @classmethod
    def prepare(cls, prompt: str):
        samples = [Sample(id=0,
                          prompt_or_input_text=prompt,  # a dummy prompt
                          references=[]
                          )]
        pool_instance = cls(samples)
        return pool_instance


class RewardIncreasingNumbers(RewardFunction):
    def __init__(self, eos_token: str,
                 min_num_tokens: int = 5,
                 include_prompt: bool = False) -> None:
        super().__init__()
        self.eos_token = eos_token
        self.min_num_tokens = min_num_tokens
        self.include_prompt = include_prompt

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
            gen_tokens = [
                current_observation.prompt_or_input_text] if self.include_prompt else []
            gen_tokens.extend(current_observation.action_history)
            if self.eos_token in gen_tokens:
                gen_tokens.remove(self.eos_token)
            number_tokens = [float(token)
                             for token in gen_tokens if self.is_number(token)]
            if len(number_tokens) > self.min_num_tokens:  # must contain atleast min numbers
                # then we check how many numbers are in the sorted order
                sorted_count = 1
                previous_token = number_tokens[0]
                for token in number_tokens[1:]:
                    if token > previous_token:
                        sorted_count += 1
                        previous_token = token
                    else:
                        break
                return (sorted_count/len(gen_tokens))
            else:
                return len(number_tokens) * 1e-4  # bonus to generate numbers

        return 0.0


def run(args):
    # reward function
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_fn = RewardIncreasingNumbers(
        tokenizer.eos_token, args.reward_min_tokens)

    data_pool = TestTextGenPool.prepare(tokenizer.eos_token)
    samples = [(sample, weight) for sample, weight in data_pool]

    # text generation env
    eval_env = TextGenEnv(tokenizer, reward_fn,
                          max_text_length=args.max_prompt_length,
                          max_steps=args.max_episode_length, samples=samples)
    for sample, weight in data_pool:
        eval_env.add_sample(sample, weight)

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

    # train
    for i in range(int(args.n_iters)):
        alg.learn(args.n_parallel_envs * args.n_steps)
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
    parser.add_argument("--reward_min_tokens", type=int,
                        default=5, help="Minimum number of number tokens that model has to generate")
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
    parser.add_argument("--lr", type=int,
                        default=1e-4, help="Learning rate")
    parser.add_argument("--ent_coef", type=float,
                        default=1e-2, help="Entropy Coefficient")
    parser.add_argument("--n_iters", type=int,
                        default=1e+2, help="Number of iterations of learn+eval")
    parser.add_argument("--model_parallel", type=bool,
                        default=True, help="Apply model parallel or not")
    args = parser.parse_args()
    run(args)
