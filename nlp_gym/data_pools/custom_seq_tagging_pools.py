from sprl_package.data_pools.multi_label_pool import MultiLabelPool
from sprl_package.data_pools.base import Sample
from torchnlp.datasets import ud_pos_dataset
from flair.data import Sentence, Corpus
from flair import datasets
from tqdm import tqdm
#from datasets import load_dataset


class UDPosTagggingPool(MultiLabelPool):
    @classmethod
    def prepare(cls, split: str):
        # get dataset from split
        train_dataset = UDPosTagggingPool._get_dataset_from_split(split)

        samples = []
        all_labels = []
        for data in train_dataset:
            token_texts = data["tokens"]
            token_texts_ = " ".join(token_texts)

            # check token to text
            flair_sentence = Sentence(token_texts_, use_tokenizer=False)
            assert len(flair_sentence.tokens) == len(token_texts)

            token_labels = data["ud_tags"]
            sample = Sample(input_text=token_texts_, oracle_label=token_labels)
            all_labels.extend(token_labels)
            samples.append(sample)
        weights = [1.0] * len(samples)
        return cls(samples, list(set(all_labels)), weights)

    @staticmethod
    def _get_dataset_from_split(split: str):
        if split == "train":
            return ud_pos_dataset(train=True)
        elif split == "val":
            return ud_pos_dataset(dev=True)
        elif split == "test":
            return ud_pos_dataset(test=True)


class WikiNERTaggingPool(MultiLabelPool):
    @classmethod
    def prepare(cls, split: str):
        # load the corpus
        corpus = datasets.WIKINER_ENGLISH()
        corpus_split = WikiNERTaggingPool._get_dataset_from_corpus(corpus, split)

        samples = []
        all_labels = []
        for sentence in tqdm(corpus_split, desc="Preparing data pool"):
            token_texts = [token.text for token in sentence]
            token_texts_ = " ".join(token_texts)
            token_labels = [token.get_tag("ner").value for token in sentence]

            # check token to text
            flair_sentence = Sentence(token_texts_, use_tokenizer=False)
            assert len(flair_sentence.tokens) == len(token_texts)

            # sample
            sample = Sample(input_text=token_texts_, oracle_label=token_labels)
            samples.append(sample)
            all_labels.extend(token_labels)
        weights = [1.0] * len(samples)
        return cls(samples, list(set(all_labels)), weights)

    @staticmethod
    def _get_dataset_from_corpus(corpus: Corpus, split: str):
        if split == "train":
            return corpus.train
        elif split == "val":
            return corpus.dev
        elif split == "test":
            return corpus.test


class CONLLNerTaggingPool(MultiLabelPool):
    """
    Note: Flair requires dataset files must be present under
    /root/.flair/datasets/conll03

    We can get the files from internet. For instance: 
    https://github.com/ningshixian/NER-CONLL2003/tree/master/data

    """

    @classmethod
    def prepare(cls, split: str):
        # load the corpus
        corpus = datasets.CONLL_03()
        corpus_split = WikiNERTaggingPool._get_dataset_from_corpus(corpus, split)

        samples = []
        all_labels = []
        for sentence in tqdm(corpus_split, desc="Preparing data pool"):
            token_texts = [token.text for token in sentence]
            token_texts_ = " ".join(token_texts)
            token_labels = [token.get_tag("ner").value for token in sentence]
            token_labels = [label.split("-")[1] if "-" in label else label
                            for label in token_labels]  # simplify labels

            # check token to text
            flair_sentence = Sentence(token_texts_, use_tokenizer=False)
            assert len(flair_sentence.tokens) == len(token_texts)

            # sample
            sample = Sample(input_text=token_texts_, oracle_label=token_labels)
            samples.append(sample)
            all_labels.extend(token_labels)
        weights = [1.0] * len(samples)
        return cls(samples, list(set(all_labels)), weights)


class GermEvalTaggingPool(MultiLabelPool):
    """
    Source: https://huggingface.co/datasets/germeval_14
    """

    @classmethod
    def prepare(cls, split: str):
        # load the corpus
        corpus = load_dataset("germeval_14")

        # corpus split
        corpus_split = GermEvalTaggingPool._get_dataset_from_corpus(corpus, split)

        samples = []
        all_labels = []
        for data_point in tqdm(corpus_split, desc="Preparing data pool"):
            token_texts = data_point["tokens"]
            token_texts_ = " ".join(token_texts)
            token_labels = data_point["labels"]
            token_labels = [label.split("-")[1] if "-" in label else label
                            for label in token_labels]  # simplify labels

            # check token to text
            flair_sentence = Sentence(token_texts_, use_tokenizer=False)
            assert len(flair_sentence.tokens) == len(token_texts)

            # sample
            sample = Sample(input_text=token_texts_, oracle_label=token_labels)
            samples.append(sample)
            all_labels.extend(token_labels)
        weights = [1.0] * len(samples)
        return cls(samples, list(set(all_labels)), weights)

    @staticmethod
    def _get_dataset_from_corpus(corpus: Corpus, split: str):
        if split == "train":
            return corpus["train"]
        elif split == "val":
            return corpus["validation"]
        elif split == "test":
            return corpus["test"]


if __name__ == "__main__":
    datapool = GermEvalTaggingPool.prepare("train")
    print(datapool.labels())