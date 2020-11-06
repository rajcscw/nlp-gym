from nlp_gym.data_pools.custom_multi_label_pools import ReutersDataPool
from nlp_gym.envs.multi_label.env import MultiLabelEnv
from nlp_gym.envs.multi_label.reward import F1RewardFunction
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
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
pool = ReutersDataPool.prepare(split="train")
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
model = DQN(env=env, policy=DQNPolicy, gamma=0.99, batch_size=32, learning_rate=1e-3,
            double_q=True, exploration_fraction=0.1,
            prioritized_replay=False, policy_kwargs={"layers": [200]},
            verbose=1)
for i in range(int(1e+3)):
    model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False)
    eval_model(model, env)