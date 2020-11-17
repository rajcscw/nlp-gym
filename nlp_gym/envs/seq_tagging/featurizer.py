from typing import List, Union

import torch
from flair.data import Sentence
from flair.embeddings import BytePairEmbeddings, StackedEmbeddings, Embeddings, WordEmbeddings

from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.seq_tagging.observation import ObservationFeaturizer, Observation


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
        "fasttext_de": {
            "cls": [WordEmbeddings],
            "params": ["de-crawl"]
        }
    }

    def get_embedding(embedding_type: str) -> List[Embeddings]:
        cls_ = EmbeddingRegistry._registry_mapping[embedding_type]["cls"]
        params_ = EmbeddingRegistry._registry_mapping[embedding_type]["params"]
        embeddings = [embedding_cls(embedding_param) for embedding_cls, embedding_param in zip(cls_, params_)]
        return embeddings


class DefaultFeaturizerForSeqTagging(ObservationFeaturizer):
    def __init__(self, action_space: ActionSpace, embedding_type: str = "fasttext", device: str = "cpu"):
        self.device = device
        self._setup_device()
        embeddings = EmbeddingRegistry.get_embedding(embedding_type)
        self.doc_embeddings = StackedEmbeddings(embeddings).to(torch.device(self.device))
        self.action_space = action_space
        self._current_token_embeddings: List[torch.tensor] = None

    def _setup_device(self):
        import flair, torch
        flair.device = torch.device(self.device)

    def init_on_reset(self, input_text: Union[List[str], str]):
        sent = Sentence(input_text)
        self.doc_embeddings.embed(sent)
        self._current_token_embeddings = [token.embedding.cpu().detach() for token in sent]
        sent.clear_embeddings()

    def featurize(self, observation: Observation) -> torch.Tensor:
        input_vector = self._featurize_input(observation.get_current_index())
        context_vector = self._featurize_context(observation.get_current_action_history())
        concatenated = torch.cat((input_vector, context_vector), dim=0)
        return concatenated

    def get_observation_dim(self) -> int:
        return self._get_input_dim() + self._get_context_dim()

    def _featurize_input(self, input_index: int) -> torch.Tensor:
        input_features = self._current_token_embeddings[input_index]
        return input_features

    def _featurize_context(self, context: List[str]) -> torch.Tensor:
        # consider only last action
        context_vector = torch.zeros(self.action_space.size())
        context_ = [context[-1]] if len(context) > 0 else []
        action_indices = [self.action_space.action_to_ix(action) for action in context_]
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