from sprl_package.data_pools.question_answering_pool import QADataPool, Sample
import tensorflow_datasets as tfds
import tensorflow as tf


class QASC(QADataPool):
    """
    Source: https://www.tensorflow.org/datasets/catalog/qasc
    """
    @classmethod
    def prepare(cls, split: str):

        if split == "val":
            split = "validation"

        tf.enable_eager_execution()
        ds = tfds.load('qasc', split=split, shuffle_files=True)

        samples = []
        for datapoint in ds:
            sample_id = datapoint["id"].numpy().decode("utf-8")
            facts = [datapoint["fact1"].numpy().decode("utf-8"), datapoint["fact2"].numpy().decode("utf-8")]
            question = datapoint["question"].numpy().decode("utf-8")
            choices = {label.decode("utf-8"): text.decode("utf-8")
                       for label, text in zip(datapoint["choices"]["label"].numpy(),
                                              datapoint["choices"]["text"].numpy())}
            answer = datapoint["answerKey"].numpy().decode("utf-8")
            sample = Sample(sample_id, question, facts, choices, answer)
            samples.append(sample)

        return QASC(samples)


class AIRC(QADataPool):
    """
    Source: https://www.tensorflow.org/datasets/catalog/ai2_arc_with_ir?hl=cs
    """
    EASY = "ai2_arc_with_ir/ARC-Easy-IR"
    CHALLENGING = "ai2_arc_with_ir/ARC-Challenge-IR"

    @classmethod
    def prepare(cls, split: str, dataset_id: str = None):

        if dataset_id == "easy":
            dataset_id = AIRC.EASY
        else:
            dataset_id = AIRC.CHALLENGING

        if split == "val":
            split = "validation"

        tf.enable_eager_execution()
        ds = tfds.load(dataset_id, split=split, shuffle_files=True)

        samples = []
        for datapoint in ds:
            sample_id = datapoint["id"].numpy().decode("utf-8")
            facts = [datapoint["paragraph"].numpy().decode("utf-8")]
            question = datapoint["question"].numpy().decode("utf-8")
            choices = {str(label): text.decode("utf-8")
                       for label, text in zip(datapoint["choices"]["label"].numpy(),
                                              datapoint["choices"]["text"].numpy())}
            answer = str(int(datapoint["answerKey"].numpy()))
            sample = Sample(sample_id, question, facts, choices, answer)
            samples.append(sample)

        return AIRC(samples)


if __name__ == "__main__":
    data_pool = AIRC.prepare(split="test", dataset_id="easy")
    print(len(data_pool))
    print(data_pool[23])