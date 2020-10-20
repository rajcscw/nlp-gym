from nlp_gym.metrics.multi_label import HammingLoss, JaccardScore, PrecisionScore, RecallScore, F1Score
from nlp_gym.metrics.seq_tag import SeqTaggingMetric, EntityScoresCorpus
from nlp_gym.metrics.multi_label import F1ScoreCorpus
from nlp_gym.metrics.question_answer import MatchScore
from typing import Any, Dict


class MetricRegistry:
    _registry_mapping = {
        "HammingLoss": HammingLoss,
        "JaccardScore": JaccardScore,
        "PrecisionScore": PrecisionScore,
        "RecallScore": RecallScore,
        "F1Score": F1Score,
        "SeqTaggingMetric": SeqTaggingMetric,
        "EntityScores": EntityScoresCorpus,
        "MatchScore": MatchScore,
        "F1ScoreCorpus": F1ScoreCorpus
    }

    @classmethod
    def get_metric(cls, metric_name: str, metric_params: Dict[str, Any] = {}):
        metric_cls = cls._registry_mapping[metric_name]
        metric_instance = metric_cls(**metric_params)
        return metric_instance
