import os
import sys
import torch
import torch.nn as nn
from framework import FewShotREModel
from utils.hyptorch.nn import ToPoincare
from utils.hyptorch.pmath import poincare_mean, dist_matrix

class HypProto(FewShotREModel):
    def __init__(self, sentence_encoder, hidden_size=230, c=1, dropout_prob=0.5, temperature=1):
        super().__init__(sentence_encoder)
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.e2p = ToPoincare(c=c, train_c=False, train_x=False)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, support, query, N, K, Q):
        support = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query) # (B * N * Q, D)
        support = self.drop(support)
        query = self.drop(query)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size) # (B, N * Q, D)
        assert support.shape[0] == query.shape[0], 'Batch sizes must be equal'
        support = self.e2p(support)
        query = self.e2p(query)
        # print('Support shape: {}'.format(support.shape))
        # print('Query shape: {}'.format(query.shape))
        proto = poincare_mean(support, dim=2, c=self.e2p.c)
        # print('Proto shape: {}'.format(proto.shape))
        distances = []
        for i in range(query.shape[0]):
            dist = -dist_matrix(query[i], proto[i], c=self.e2p.c) / self.temperature
            # print('Dist shape: {}'.format(dist.shape))
            distances.append(dist)
        logits = torch.cat([dist.unsqueeze(0) for dist in distances], dim=0)
        # print('Logits shape: {}'.format(logits.shape))
        _, pred = torch.max(logits.view(-1, N), 1)
        # print('Pred shape: {}'.format(pred.shape))
        return logits, pred
