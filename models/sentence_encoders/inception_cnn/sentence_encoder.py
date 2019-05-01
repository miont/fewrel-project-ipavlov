import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from .network.embedding import Embedding
from .network.encoder import Encoder

class InceptionCNNSentenceEncoder(nn.Module):
    """
    """
    def __init__(self, word_embedder, max_length, pos_embedding_dim=5, sizes=[{3:100, 5:100, 7:100}]):
        """
        """
        print('Init sentince encoder with sizes:\n {}'
            .format(sizes))
        nn.Module.__init__(self)
        self.max_length = max_length
        self.embedding = Embedding(word_embedder, max_length, pos_embedding_dim)
        self.encoder = Encoder(max_length, word_embedder.word_vec_dim, pos_embedding_dim, sizes)
        self.hidden_size = self.encoder.embedding_size

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x