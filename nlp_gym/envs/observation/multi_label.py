import re
import string
from typing import List, Union

import flair
import torch
from flair.data import Sentence
from flair.embeddings import (BytePairEmbeddings, DocumentPoolEmbeddings,
                              Embeddings, WordEmbeddings)
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sprl_package.envs.action.action_space import ActionSpace
from sprl_package.envs.observation.base_featurizer import ObservationFeaturizer


class TextPreProcessor:
    def __init__(self, language: str):
        self.language = language

    def _remove_digits(self, text: str) -> str:
        text = re.sub(r"\d+", "", text)
        return text

    def _remove_punctuation(self, text: str) -> str:
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def process(self, text: str) -> str:
        text = text.lower()
        text = self._remove_punctuation(text)
        text = self._remove_digits(text)
        text = self._remove_stop_words(text)
        text = self._stem(text)
        return text

    def _remove_stop_words(self, text: str) -> str:
        stop_words_list = stopwords.words(self.language)
        return ' '.join([word for word in text.split() if word not in stop_words_list])

    def _stem(self, text: str) -> str:
        stemmer = SnowballStemmer(language=self.language)
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def get_id(self) -> str:
        return f"advanced_{self.language}"


class EmbeddingRegistry:
    _registry_mapping = {
        "byte_pair": [BytePairEmbeddings("en")],
        "fasttext": [WordEmbeddings("en-crawl")],
        "stacked": [WordEmbeddings("en-crawl"), BytePairEmbeddings("en")]
    }

    @staticmethod
    def get_embedding(embedding_type: str) -> List[Embeddings]:
        return EmbeddingRegistry._registry_mapping[embedding_type]


class DefaultFeaturizerForMultiLabelRank(ObservationFeaturizer):
    def __init__(self, action_space: ActionSpace, embedding_type: str = "byte_pair", pre_process: bool = False,
                 device: str = "cpu"):
        self.device = device
        self.pre_process = pre_process
        self.text_pre_processor = TextPreProcessor(language="english")
        self._setup_device()
        embeddings = EmbeddingRegistry.get_embedding(embedding_type)
        self.doc_embeddings = DocumentPoolEmbeddings(embeddings).to(torch.device(self.device))
        self.action_space = action_space
        self._current_input_embeddings = None

    def _setup_device(self):
        flair.device = torch.device(self.device)

    def init_on_reset(self, input_text: Union[List[str], str]):
        # pooled document embeddings
        text = self.text_pre_processor.process(input_text) if self.pre_process else input_text
        sent = Sentence(text)
        self.doc_embeddings.embed(sent)
        self._current_input_embeddings = torch.tensor(sent.embedding.cpu().detach().numpy())

    def featurize_input_on_step(self, input_index: int) -> torch.Tensor:
        # the input does not change on each step
        return self._current_input_embeddings

    def featurize_context_on_step(self, context: List[str]) -> torch.Tensor:
        # bag of actions representation
        context_vector = torch.zeros(self.action_space.size())
        action_indices = [self.action_space.action_to_ix(action) for action in context]
        context_vector[action_indices] = 1.0
        return context_vector

    def get_input_dim(self):
        sent = Sentence("A random text to get the embedding dimension")
        self.doc_embeddings.embed(sent)
        dim = sent[0].embedding.shape[0]
        sent.clear_embeddings()
        return dim

    def get_context_dim(self):
        return self.action_space.size()
