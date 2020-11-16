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
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.multi_label.observation import ObservationFeaturizer, Observation


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
        "byte_pair": {
            "cls": [BytePairEmbeddings],
            "params": ["en"]
        },
        "fasttext": {
            "cls": [WordEmbeddings],
            "params": ["en-crawl"]
        },
        "stacked": {
            "cls": [WordEmbeddings, BytePairEmbeddings],
            "params": ["en-crawl", "en"]
        }
    }

    @staticmethod
    def get_embedding(embedding_type: str) -> List[Embeddings]:
        cls_ = EmbeddingRegistry._registry_mapping[embedding_type]["cls"]
        params_ = EmbeddingRegistry._registry_mapping[embedding_type]["params"]
        embeddings = [embedding_cls(embedding_param) for embedding_cls, embedding_param in zip(cls_, params_)]
        return embeddings


class DefaultFeaturizerForMultiLabelRank(ObservationFeaturizer):
    def __init__(self, action_space: ActionSpace, embedding_type: str = "fasttext", pre_process: bool = False,
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

    def featurize(self, observation: Observation) -> torch.Tensor:
        input_vector = self._current_input_embeddings
        context_vector = self._featurize_context(observation.get_current_action_history())
        concatenated = torch.cat((input_vector, context_vector), dim=0)
        return concatenated

    def get_observation_dim(self) -> int:
        return self._get_input_dim() + self._get_context_dim()

    def _featurize_input(self, input_index: int) -> torch.Tensor:
        # the input does not change on each step
        return self._current_input_embeddings

    def _featurize_context(self, context: List[str]) -> torch.Tensor:
        # bag of actions representation
        context_vector = torch.zeros(self.action_space.size())
        action_indices = [self.action_space.action_to_ix(action) for action in context]
        context_vector[action_indices] = 1.0
        return context_vector

    def _get_input_dim(self):
        sent = Sentence("A random text to get the embedding dimension")
        self.doc_embeddings.embed(sent)
        dim = sent[0].embedding.shape[0]
        sent.clear_embeddings()
        return dim

    def _get_context_dim(self):
        return self.action_space.size()


if __name__ == "__main__":
    embeddings = EmbeddingRegistry.get_embedding("stacked")
    print(embeddings)
