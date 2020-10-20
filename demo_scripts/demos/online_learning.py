
from nlp_gym.envs.seq_tag_env import SeqTagEnv
from nlp_gym.data_pools.custom_seq_tagging_pools import UDPosTagggingPool
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
from nlp_gym.envs.reward.seq_tagging import EntityF1Score
from nlp_gym.envs.observation.seq_tagging import DefaultFeaturizerForSeqTagging
from nlp_gym.metrics.seq_tag import EntityScores


def predict(model, sample):
    done = False
    obs = env.reset(sample)
    predicted_label = []
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        predicted_label.append(env.action_space.ix_to_action(action))
    return predicted_label


# data pool
data_pool = UDPosTagggingPool.prepare(split="train")

# reward function
reward_fn = EntityF1Score(dense=True, average="micro")

# seq tag env
env = SeqTagEnv(data_pool.labels(), reward_function=reward_fn)

# observation featurizer
feat = DefaultFeaturizerForSeqTagging(env.action_space, embedding_type="fasttext")
env.set_featurizer(feat)

# PPO model
model = PPO1(MlpPolicy, env, verbose=1)


# train loop that goes over each sample only once
for sample, _ in data_pool:

    # run the sample through the model and get predicted label
    predicted_label = predict(model, sample)

    # after few epochs, predicted_label can be used as pre-annotated input
    # then the user can just correct it
    # to reduce human efforts

    # get annotated label from user (just simulated for now)
    annotated_label = sample.oracle_label

    # match score
    match_ratio = EntityScores()(annotated_label, predicted_label)

    # add the new sample to the environment
    sample.oracle_label = annotated_label
    env.add_sample(sample)

    # train agent for few epochs
    model.learn(total_timesteps=1e+2)
