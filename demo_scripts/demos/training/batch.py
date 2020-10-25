from nlp_gym.data_pools.custom_multi_label_pools import AAPDDataPool
from nlp_gym.envs.multi_label_env import MultiLabelEnv
from nlp_gym.envs.reward.multi_label import F1RewardFunction
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines import PPO1
from stable_baselines.common.env_checker import check_env
from rich import print


def eval_model(model, env):
    done = False
    obs = env.reset()
    total_reward = 0.0
    actions = []
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        actions.append(env.action_space.ix_to_action(action))
        total_reward += rewards
    print("---------------------------------------------")
    print(f"Text: {env.current_sample.text}")
    print(f"Predicted Label {actions}")
    print(f"Oracle Label: {env.current_sample.label}")
    print(f"Total Reward: {total_reward}")
    print("---------------------------------------------")


# data pool
pool = AAPDDataPool.prepare()
labels = pool.labels()

# reward function
reward_fn = F1RewardFunction()

# multi label env
env = MultiLabelEnv(possible_labels=labels, max_steps=10, reward_function=reward_fn,
                    return_obs_as_vector=True)
for sample, weight in pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)

# train a MLP Policy
#model = PPO1(MlpPolicy, env, verbose=1)
model = DQN(DQNPolicy, env, verbose=1)
for i in range(int(1e+3)):
    model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False)
    eval_model(model, env)