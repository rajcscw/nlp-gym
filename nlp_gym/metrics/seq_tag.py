from seqeval.metrics.sequence_labeling import f1_score, precision_score, recall_score
from typing import List
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


class SeqTaggingMetric:
    def __call__(self, true_labels: List[List[str]], predicted_labels: List[List[str]]):
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        metrics_dict = {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1
        }
        return metrics_dict


class EntityScores:
    def __init__(self, average: str = "micro"):
        self.average = average

    def __call__(self, true_labels: List[str], predicted_labels: List[str]):
        metrics_dict = {
            "precision": precision_score(true_labels, predicted_labels, average=self.average) if len(true_labels) > 0 else 0.0,
            "recall": recall_score(true_labels, predicted_labels, average=self.average) if len(true_labels) > 0 else 0.0,
            "f1": f1_score(true_labels, predicted_labels, average=self.average) if len(true_labels) > 0 else 0.0,
        }
        return metrics_dict


class EntityScoresCorpus:
    def __init__(self, average: str = "micro"):
        self.average = average

    def __call__(self, true_labels: List[List[str]], predicted_labels: List[List[str]]):
        all_true = []
        all_predicted = []
        for true, predicted in zip(true_labels, predicted_labels):
            all_true.extend(true)
            all_predicted.extend(predicted)

        metrics_dict = {
            "precision": precision_score(all_true, all_predicted, average=self.average) if len(true_labels) > 0 else 0.0,
            "recall": recall_score(all_true, all_predicted, average=self.average) if len(true_labels) > 0 else 0.0,
            "f1": f1_score(all_true, all_predicted, average=self.average) if len(true_labels) > 0 else 0.0,
        }
        return metrics_dict


if __name__ == "__main__":
    metric_fn = EntityScores(average="macro")
    sent_true_labels = ["PER", "LOC", "0"]
    sent_predicted_labels = ["0", "0", "0"]
    print(metric_fn(sent_true_labels, sent_predicted_labels))