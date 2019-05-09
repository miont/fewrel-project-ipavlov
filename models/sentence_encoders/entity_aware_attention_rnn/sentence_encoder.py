import torch
import torch.nn as nn
from ..common.embedding import Embedding
from .network.self_attention import MultiheadSelfAttention
from .network.entity_aware_attention import EntityAwareAttention

class EntityAwareAttentionRnn(nn.Module):
    """
    Encoder based on relation classification model proposed in paper:
    https://arxiv.org/pdf/1901.08163.pdf
    """
    def __init__(self, word_embedder, max_length, hidden_size=100, 
                 pos_embedding_dim=0, n_heads=4, attention_dim=50,
                 dropout_we=0.3, dropout_rnn=0.3, dropout_eaa=0.5,
                 num_latent_types=3, cuda=True):
        super().__init__()
        self.max_length = max_length
        self.pos_emb_dim = pos_embedding_dim

        self.dropout = {
            'word_embedding': dropout_we,
            'rnn': dropout_rnn,
            'entity_aware_att': dropout_eaa
        }

        self.embedder = Embedding(word_embedder, max_length, pos_embedding_dim=pos_embedding_dim, cuda=cuda)
        self.self_attention = MultiheadSelfAttention(emb_dim=self.embedder.word_embedding_dim, n_heads=n_heads)
        self.rnn = nn.LSTM(self.embedder.word_embedding_dim, hidden_size, bidirectional=True)
        self.entity_aware_att = EntityAwareAttention(max_length=max_length, 
                                                     hidden_dim=hidden_size, 
                                                     attention_dim=attention_dim, 
                                                     pos_emb_dim=pos_embedding_dim, 
                                                     num_latent_types=num_latent_types)
        self.dropout_layers = {k: nn.Dropout(p) for k,p in self.dropout.items()}

        self.hidden_size = 2*hidden_size  # Output encoding dimension

    def forward(self, inputs):

        # Word embeddings
        emb = self.embedder(inputs)
        word_emb, pos1_emb, pos2_emb = self._separate_embeddings(emb)
        word_emb = self.dropout_layers['word_embedding'](word_emb)

        # Self-attention
        x = self.self_attention(word_emb)
        # print(x.shape)

        # LSTM
        x, _ = self.rnn(x)
        x = self.dropout_layers['rnn'](x)

        # Entity-aware attention
        e1_idx, e2_idx = self._get_entity_indices(inputs['pos1'], inputs['pos2'])
        x = self.entity_aware_att(x, pos1_emb, pos2_emb, e1_idx, e2_idx)
        x = self.dropout_layers['entity_aware_att'](x)
        # print('x shape: {}'.format(x.shape))
        return x

    def _separate_embeddings(self, emb):
        """
        Separate word and positional embeddings
        """
        word_emb_dim = self.embedder.word_embedding_dim
        pos_emb_dim = self.embedder.pos_embedding_dim
        i1 = word_emb_dim
        i2 = word_emb_dim + pos_emb_dim
        word_emb = emb[:, :, :i1]
        pos1_emb = emb[:, :, i1:i2]
        pos2_emb = emb[:, :, i2:]
        return word_emb, pos1_emb, pos2_emb

    def _get_entity_indices(self, pos1, pos2):
        pos1 = torch.from_numpy(pos1).long()
        pos2 = torch.from_numpy(pos2).long()
        entity1_idx = torch.nonzero(pos1 == self.max_length)[:,-1]
        entity2_idx = torch.nonzero(pos2 == self.max_length)[:,-1]
        # print('Pos1: {}'.format(pos1))
        # print('Entity 1 index: {}'.format(entity1_idx))
        # print('Pos2: {}'.format(pos2))
        # print('Entity 2 index: {}'.format(entity2_idx))

        return entity1_idx, entity2_idx

        