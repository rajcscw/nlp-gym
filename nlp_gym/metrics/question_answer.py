from sklearn.metrics import accuracy_score
from typing import List


class MatchScore:
    name = "match_score"

    def __call__(self, true_labels: List[str], predicted_labels: List[str]) -> float:
        accuracy = accuracy_score(true_labels, predicted_labels)
        return accuracy
