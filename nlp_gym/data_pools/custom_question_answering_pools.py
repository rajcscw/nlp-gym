from nlp_gym.data_pools.question_answering_pool import QADataPool, Sample
import tensorflow_datasets as tfds
import tensorflow as tf
import wget
import zipfile
import os
import json


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
    Source: Adapted version of https://www.tensorflow.org/datasets/catalog/ai2_arc_with_ir?hl=cs

    Source Files downloaded from: http://aristo-data.s3.amazonaws.com/custom-datasets/ARC-IR10V8.zip
    """

    # dataset variations
    EASY = "ARC-Easy-IR"
    CHALLENGING = "ARC-Challenge-IR"

    SOURCE_URL = "http://aristo-data.s3.amazonaws.com/custom-datasets/ARC-IR10V8.zip"
    DEST_BASE_PATH = os.path.join(os.path.expanduser("~"), "nlp_gym_datasets", "airc_with_ir")
    DEST_ZIP_FILE_PATH = os.path.join(DEST_BASE_PATH, "ARC-IR10V8.zip")
    SPLIT_NAMES = {
        "train": "ARC-IR10V8/train.jsonl",
        "val": "ARC-IR10V8/dev.jsonl",
        "test": "ARC-IR10V8/test.jsonl"
    }
    PREFIX = {EASY: "ARCEZ_", CHALLENGING: "ARCCH_"}
    N_TO_L = dict(zip("1 2 3 4 5".split(), "A B C D E".split()))

    @classmethod
    def _download_files(cls):
        if not os.path.exists(AIRC.DEST_ZIP_FILE_PATH):
            wget.download(AIRC.SOURCE_URL, AIRC.DEST_ZIP_FILE_PATH)
            with zipfile.ZipFile(AIRC.DEST_ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(AIRC.DEST_BASE_PATH)

    @classmethod
    def prepare(cls, split: str, dataset_id: str = None):
        AIRC._download_files()
        file_path = os.path.join(AIRC.DEST_BASE_PATH, AIRC.SPLIT_NAMES[split])
        samples = []
        with open(file_path, "r") as f:
            for row in f:
                data = json.loads(row)

                if not data["id"].startswith(AIRC.PREFIX[dataset_id]):
                    continue

                sample_id = data["id"].replace(AIRC.PREFIX[dataset_id], "")
                question = data["question"]["stem"]
                answer = AIRC.N_TO_L.get(data["answerKey"], data["answerKey"])
                facts = data["para"]
                choices = data["question"]["choices"]
                text_choices = [choice["text"] for choice in choices]
                label_choices = [
                    AIRC.N_TO_L.get(choice["label"], choice["label"]) for choice in choices
                ]
                choices = {str(label): text for label, text in zip(label_choices, text_choices)}
                sample = Sample(sample_id, question, facts, choices, answer)
                samples.append(sample)

        return AIRC(samples)
