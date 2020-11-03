from nlp_gym.data_pools.multi_label_pool import MultiLabelPool
from nlp_gym.data_pools.base import Sample
from nlp_gym.util import get_sample_weights
from nlp_gym import aapd_data_path
from collections import defaultdict
from typing import List, Dict
from nltk.corpus import reuters
import os
import random


class AAPDDataPool(MultiLabelPool):
    """
    Source repo: https://git.uwaterloo.ca/jimmylin/Castor-data/tree/master/datasets/AAPD
    Dataset for paper: https://arxiv.org/pdf/1806.04822.pdf
    """

    @classmethod
    def prepare(cls, split: str):
        documents = []
        labels = []
        with open(os.path.join(aapd_data_path, f"text_{split}")) as text_file, \
                open(os.path.join(aapd_data_path, f"label_{split}")) as topic_file:
            for text, topics in zip(text_file, topic_file):
                if text != "" and topics != "":
                    text = text.strip()
                    label = topics.strip().split()
                    documents.append(Sample(input_text=text, oracle_label=label))
                    labels.extend(label)
        random.shuffle(documents)
        weights = get_sample_weights(documents)
        return cls(documents, list(set(labels)), weights)


class ReutersDataPool(MultiLabelPool):
    @classmethod
    def prepare(cls, split: str):
        def _get_sample_ids_to_labels(labels: List[str], split: str) -> Dict[str, List[str]]:
            doc_ids_to_labels = defaultdict(list)
            for label in labels:
                doc_ids = reuters.fileids(label)

                split_to_filter = "train" if split == "val" else split
                doc_ids = [doc_id for doc_id in doc_ids if split_to_filter in doc_id]
                random.seed(0)
                random.shuffle(doc_ids)

                # here we split the train into train and val sets
                if split in ["train", "val"]:
                    val_split_ratio = 0.7
                    split_ix = int(len(doc_ids) * val_split_ratio)
                    train_doc_ids, val_doc_ids = doc_ids[:split_ix], doc_ids[split_ix:]
                    doc_ids = train_doc_ids if split == "train" else val_doc_ids

                for doc_id in doc_ids:
                    doc_ids_to_labels[doc_id].append(label)
            return doc_ids_to_labels

        def _filter_docs(doc_ids_to_labels: Dict[str, List[str]]) -> Dict[str, List[str]]:
            doc_ids_to_labels = {doc_id: cats for doc_id, cats in doc_ids_to_labels.items()}
            return doc_ids_to_labels

        def _get_labels(doc_ids_to_labels: Dict[str, List[str]]) -> List[str]:
            categories = []
            for _, cats in doc_ids_to_labels.items():
                categories.extend(cats)
            return list(set(categories))

        def _get_samples(doc_ids_to_labels: Dict[str, List[str]]) -> List[Sample]:
            samples = []
            for doc_id, label in doc_ids_to_labels.items():
                text = reuters.raw(doc_id)
                samples.append(Sample(text, label))
            return samples

        # get samples, labels and their mapping
        categories = sorted(reuters.categories())
        doc_ids_to_labels = _get_sample_ids_to_labels(categories, split)
        doc_ids_to_labels = _filter_docs(doc_ids_to_labels)
        samples = _get_samples(doc_ids_to_labels)
        random.shuffle(samples)
        labels = _get_labels(doc_ids_to_labels)
        weights = get_sample_weights(samples)
        return cls(samples, labels, weights)


if __name__ == "__main__":
    pool = ReutersDataPool.prepare(split="train")
    print(len(pool))
