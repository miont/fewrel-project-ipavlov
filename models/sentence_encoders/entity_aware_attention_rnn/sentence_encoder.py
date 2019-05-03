import torch
import torch.nn as nn
from .network.embedding import Embedding
from .network.self_attention import MultiheadSelfAttention

class EntityAwareAttentionRnn(nn.Module):
    """
    Encoder based on relation classification model proposed in paper:
    https://arxiv.org/pdf/1901.08163.pdf
    """
    def __init__(self, word_embedder, max_length, hidden_size=100, pos_embedding_dim=0, n_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedder = Embedding(word_embedder, max_length, pos_embedding_dim=pos_embedding_dim)
        self.self_attention = MultiheadSelfAttention(emb_dim=self.embedder.embedding_dim, n_heads=n_heads)
        self.rnn = nn.LSTM(self.embedder.embedding_dim, hidden_size)
        self.entity_aware_att = None

    def forward(self, inputs):
        x = self.embedder(inputs)
        x = self.self_attention(x)
        # print(x.shape)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        return x

        