from nlp_gym.data_pools.custom_multi_label_pools import ReutersDataPool
from nlp_gym.envs.multi_label_env import MultiLabelEnv
from nlp_gym.envs.reward.multi_label import F1RewardFunction

# data pool
pool = ReutersDataPool.prepare(split="train")
labels = pool.labels()

# reward function
reward_fn = F1RewardFunction()

# multi label env
env = MultiLabelEnv(possible_labels=labels, max_steps=15, reward_function=reward_fn,
                    return_obs_as_vector=True)
for sample, weight in pool:
    env.add_sample(sample, weight)

# play an episode
done = False
state = env.reset()
total_reward = 0
while not done:
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    total_reward += reward
print(f"Episodic reward {total_reward}")