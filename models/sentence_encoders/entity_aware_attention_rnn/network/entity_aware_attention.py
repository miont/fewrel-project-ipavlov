import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

class EntityAwareAttention(nn.Module):
    def __init__(self, max_length=40, hidden_dim=300, attention_dim=50, pos_emb_dim=50, num_latent_types=3):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        # Parameters
        self.linear_hidden = nn.Linear(2*hidden_dim+2*pos_emb_dim, attention_dim, bias=False)
        self.linear_entity = nn.Linear(2*hidden_dim+2*pos_emb_dim, attention_dim, bias=False)
        # self.



    def forward(self, hid, pos1_emb, pos2_emb):
        pass
