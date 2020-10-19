from abc import ABC, abstractmethod
from flair.embeddings import DocumentPoolEmbeddings, BytePairEmbeddings
from sprl_package.envs.action_space import ActionSpace
import torch
from typing import List, Union
from flair.data import Sentence


class ObservationFeaturizer(ABC):
    @abstractmethod
    def featurize_input(self, input_text: Union[List[str], str]) -> torch.Tensor:
        """
        Takes an input text (sentence) or list of token strings and returns a vector representation

        Returns:
            torch.Tensor: input representation
        """
        raise NotImplementedError

    def featurize_context(self, context: List[str]) -> torch.Tensor:
        """
        Takes a list of context labels and returns a vector representation

        Returns:
            torch.Tensor: context representation
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the input vector representation
        """
        raise NotImplementedError

    def get_context_dim(self) -> int:
        """
        Returns the dimension of the context vector representation
        """
        raise NotImplementedError


class DefaultFeaturizerForMultiLabelRank(ObservationFeaturizer):
    def __init__(self, action_space: ActionSpace):
        self.doc_embeddings = DocumentPoolEmbeddings([WordEmbeddings("en")])
        self.action_space = action_space

    def featurize_input(self, input_text: List[str]):
        # pooled document embeddings
        sent = Sentence(input_text)
        self.doc_embeddings.embed(sent)
        return torch.tensor(sent.embedding.cpu().detach().numpy())

    def featurize_context(self, context: List[str]):
        # bag of actions representation
        context_vector = torch.zeros(self.action_space.size())
        action_indices = [self.action_space.action_to_ix(action) for action in context]
        context_vector[action_indices] = 1.0
        return context_vector

    def get_input_dim(self):
        return 300

    def get_context_dim(self):
        return self.action_space.size()


class DefaultFeaturizerForSeqTagging(ObservationFeaturizer):
    def __init__(self, action_space: ActionSpace):
        self.doc_embeddings = BytePairEmbeddings("en")
        self.action_space = action_space

    def featurize_input(self, input_text: str):
        # here the input is a token and we have to return with token embeddings
        sent = Sentence(input_text)
        self.doc_embeddings.embed(sent)
        token_embedding = torch.tensor(sent[0].embedding.cpu().detach().numpy())
        return token_embedding

    def featurize_context(self, context: List[str]):
        # last action
        context_vector = torch.zeros(self.action_space.size())
        context_ = [context[-1]] if len(context) > 0 else []
        action_indices = [self.action_space.action_to_ix(action) for action in context_]
        context_vector[action_indices] = 1.0
        return context_vector

    def get_input_dim(self):
        return 100

    def get_context_dim(self):
        return self.action_space.size()

    def get_observation_dim(self):
        return self.get_input_dim() + self.get_context_dim()
