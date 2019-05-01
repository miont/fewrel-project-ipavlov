import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from .network.embedding import Embedding
from .network.encoder import Encoder

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_embedder, max_length, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = Embedding(word_embedder, max_length, pos_embedding_dim)
        self.encoder = Encoder(max_length, word_embedder.word_vec_dim, pos_embedding_dim, hidden_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

class PCNNSentenceEncoder(nn.Module):

    def __init__(self, word_embedder, max_length, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = Embedding(word_embedder, max_length, pos_embedding_dim)
        self.encoder = Encoder(max_length, word_embedder.word_vec_dim, pos_embedding_dim, hidden_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder.pcnn(x, inputs['mask'])
        return x
