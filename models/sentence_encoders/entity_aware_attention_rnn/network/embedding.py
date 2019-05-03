import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class Embedding(nn.Module):

    def __init__(self, word_embedder, max_length, pos_embedding_dim=5, cuda=True):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedder.word_vec_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.embedding_dim = self.word_embedding_dim + 2*self.pos_embedding_dim

        # Word embedding
        self.word_embedding = word_embedder

        # Position Embedding
        self.use_pos_emb = True
        if pos_embedding_dim > 0:
            self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
            self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        else:
            self.use_pos_emb = False
            print('Not using positional embedding')

        self.cuda = cuda

    def forward(self, inputs):
        word = inputs['word']
        pos1 = torch.from_numpy(inputs['pos1']).long()
        pos2 = torch.from_numpy(inputs['pos2']).long()

        if self.cuda:
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()

        if self.use_pos_emb:
            x = torch.cat([
                self.word_embedding(word), 
                self.pos1_embedding(pos1), 
                self.pos2_embedding(pos2)], 2)
        else:
            x = self.word_embedding(word)
        return x
