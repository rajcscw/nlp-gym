from nlp_gym.envs.multi_label_env import MultiLabelEnv
from nlp_gym.envs.seq_tag_env import SeqTagEnv
from nlp_gym.envs.question_answering_env import QAEnv
from nlp_gym.envs.reward.multi_label import F1RewardFunction
from nlp_gym.envs.reward.seq_tagging import F1Score, EntityF1Score
from nlp_gym.envs.observation.seq_tagging import DefaultFeaturizerForSeqTagging
from nlp_gym.envs.observation.question_answering import DefaultSimpleFeaturizerForQA, DefaultFeaturizerForQA
from nlp_gym.envs.observation.multi_label import DefaultFeaturizerForMultiLabelRank
from typing import Any, Dict


class EnvRegistry:
    _registry_mapping = {
        "MultiLabelEnv": MultiLabelEnv,
        "SeqTagEnv": SeqTagEnv,
        "QAEnv": QAEnv
    }

    @classmethod
    def get_env(cls, env_name: str, env_params: Dict[str, Any] = {}):
        env_cls = cls._registry_mapping[env_name]
        env_instance = env_cls(**env_params)
        return env_instance


class RewardFunctionRegistry:
    _registry_mapping = {
        "F1Reward": F1RewardFunction,
        "F1ScoreForSeqTag": F1Score,
        "EntityF1Score": EntityF1Score
    }

    @classmethod
    def get_reward_fn(cls, reward_fn_name: str, reward_params: Dict[str, Any] = {}):
        reward_cls = cls._registry_mapping[reward_fn_name]
        reward_instance = reward_cls(**reward_params)
        return reward_instance


class FeaturizerRegistry:
    _registry_mapping = {
        "DefaultSeqTag": DefaultFeaturizerForSeqTagging,
        "DefaultQA": DefaultFeaturizerForQA,
        "DefaultInformedQA": DefaultSimpleFeaturizerForQA,
        "DefaultMultiLabel": DefaultFeaturizerForMultiLabelRank,

    }

    @classmethod
    def get_featurize_fn(cls, featurizer_name: str, featurizer_params: Dict[str, Any] = {}):
        featurizer_cls = cls._registry_mapping[featurizer_name]
        featurizer_instance = featurizer_cls(**featurizer_params)
        return featurizer_instance
