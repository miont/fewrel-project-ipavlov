import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils

class EntityAwareAttention(nn.Module):
    def __init__(self, max_length=40, hidden_dim=300, attention_dim=50, pos_emb_dim=50, num_latent_types=3, verbose=False):
        nn.Module.__init__(self)
        self.verbose = verbose
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        # Parameters
        self.linear_proj_hidden = nn.Linear(2*hidden_dim+2*pos_emb_dim, attention_dim, bias=False)
        self.linear_proj_entity = nn.Linear(8*hidden_dim, attention_dim, bias=False)
        self.latent_types = utils.get_weights(num_latent_types, 2*hidden_dim)
        self.v = utils.get_weights(attention_dim, 1)

    def forward(self, hidden, pos1_emb, pos2_emb, entity1_idx, entity2_idx):
        hid_e1 = hidden[range(hidden.shape[0]),entity1_idx,:].unsqueeze(1)
        hid_e2 = hidden[range(hidden.shape[0]),entity2_idx,:].unsqueeze(1)
        e1_type = self._calc_latent_type(hid_e1)
        e2_type = self._calc_latent_type(hid_e2)

        u1 = self.linear_proj_hidden(torch.cat(
                [hidden, pos1_emb, pos2_emb], -1))
        u2 = self.linear_proj_entity(torch.cat(
                [hid_e1, e1_type, hid_e2, e2_type], -1))

        if self.verbose:
            print('hid_e1 shape: {}'.format(hid_e1.shape))
            print(torch.cat(
                [hid_e1, e1_type, hid_e2, e2_type], -1).shape)
            print('u1 shape: {}'.format(u1.shape))
            print('u2 shape: {}'.format(u2.shape))

        u = F.tanh(u1 + u2)

        alpha = F.softmax((u @ self.v).squeeze(), dim=-1).unsqueeze(-1)
        if self.verbose:
            print('u shape: {}'.format(u.shape))
            print('v shape: {}'.format(self.v.shape))
            print('alpha shape: {}'.format(alpha.shape))
            print('hidden shape: {}'.format(hidden.shape))
        z = torch.bmm(alpha.transpose(1,2), hidden).squeeze()
        if self.verbose:
            print('z shape: {}'.format(z.shape))
        return z

    def _calc_latent_type(self, entity_emb):
        a = F.softmax(
            torch.tensordot(entity_emb, self.latent_types, dims=([-1], [-1])), dim=-1)
        t = torch.tensordot(a, self.latent_types, dims=([-1], [0]))
        return t
