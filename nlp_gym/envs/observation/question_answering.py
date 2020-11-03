from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings, Embeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from typing import List

@dataclass(init=True)
class Observation:
    question: str
    facts: List[str]
    choice: str
    input_embedding: torch.tensor

    def get_vector(self) -> torch.Tensor:
        return self.input_embedding

    @classmethod
    def build(cls, question: str, facts: List[str], choice: str,
              observation_featurizer: 'ObservationFeaturizer') -> 'Observation':
        input_embedding = observation_featurizer.featurize(question, facts, choice)
        return Observation(question, facts, choice, input_embedding)


class ObservationFeaturizer(ABC):
    @abstractmethod
    def featurize(self, question: str, facts: List[str], choice: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_observation_dim(self):
        raise NotImplementedError


class InformedFeaturizer(ObservationFeaturizer):
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._setup_device()
        self.doc_embeddings = DocumentPoolEmbeddings([WordEmbeddings("en")])

    def _get_sentence_embedding(self, text: str) -> torch.Tensor:
        text = "..." if len(text) == 0 else text
        sent = Sentence(text)
        self.doc_embeddings.embed(sent)
        if len(sent) > 1:
            embedding = torch.tensor(sent.embedding.cpu().numpy()).reshape(1, -1)
        else:
            embedding = torch.tensor(sent[0].embedding.cpu().numpy()).reshape(1, -1)
        return embedding

    def _setup_device(self):
        import flair, torch
        flair.device = torch.device(self.device)

    def _get_sim(self, query: str, choice_text: str):
        sim = torch.nn.CosineSimilarity(dim=1)(self._get_sentence_embedding(query),
                                               self._get_sentence_embedding(choice_text))
        return sim

    def featurize(self, question: str, facts: List[str], choice: str) -> torch.Tensor:
        sim_scores = [self._get_sim(question, choice), self._get_sim(".".join(facts), choice)]
        sim_scores = torch.tensor(sim_scores)
        return sim_scores

    def get_observation_dim(self):
        return 2


class SimpleFeaturizer(ObservationFeaturizer):
    def __init__(self, doc_embeddings: Embeddings = DocumentPoolEmbeddings([WordEmbeddings("en")]), device: str = "cpu"):
        self.device = device
        self._setup_device()
        self.doc_embeddings = doc_embeddings

    @classmethod
    def from_fasttext(cls) -> 'DefaultFeaturizerForQA':
        return cls(DocumentPoolEmbeddings([WordEmbeddings("en")]))

    @classmethod
    def from_transformers(cls) -> 'DefaultFeaturizerForQA':
        return cls(TransformerDocumentEmbeddings())

    def _setup_device(self):
        import flair, torch
        flair.device = torch.device(self.device)

    def _get_sentence_embedding(self, text: str) -> torch.Tensor:
        text = "..." if len(text) == 0 else text
        sent = Sentence(text)
        self.doc_embeddings.embed(sent)
        if len(sent) >= 1:
            embedding = torch.tensor(sent.embedding.cpu().detach().numpy()).reshape(1, -1)
        else:
            embedding = torch.tensor(sent[0].embedding.cpu().detach().numpy()).reshape(1, -1)
        sent.clear_embeddings()
        return embedding

    def featurize(self, question: str, facts: List[str], choice: str) -> torch.Tensor:
        question_embedding = self._get_sentence_embedding(question)
        fact_embedding = self._get_sentence_embedding(".".join(facts))
        choice_embedding = self._get_sentence_embedding(choice)
        combined = torch.cat((question_embedding, fact_embedding, choice_embedding), dim=1).flatten()
        return combined

    def get_observation_dim(self):
        embedding = self._get_sentence_embedding("A random sentence to infer dim")
        return embedding.shape[1] * 3  # for question, fact and choice
