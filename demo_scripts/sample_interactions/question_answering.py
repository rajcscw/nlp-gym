from nlp_gym.data_pools.custom_question_answering_pools import AIRC
from nlp_gym.envs.question_answering_env import QAEnv

# data pool
pool = AIRC.prepare("train")

# custom answering env
env = QAEnv()
for sample, weight in pool:
    env.add_sample(sample)

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