import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    """
    Self attention layer
    """
    def __init__(self, emb_dim=300, n_heads=4, cuda=True, verbose=False):
        nn.Module.__init__(self)
        self.verbose = verbose
        if emb_dim % n_heads != 0:
            raise ValueError('Embedding dim {} is not divisible by number of heads {}'
                .format(emb_dim, n_heads))

        # Attention heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(i+1, emb_dim, emb_dim//n_heads, verbose=self.verbose) for i in range(n_heads)
        ])

        # Final linear transformation
        self.W_m = get_weights(emb_dim, emb_dim)
        # register_parameter(self, 'W_m', self.W_m)


    def forward(self, inputs):
        att = torch.cat([
            head(inputs) for head in self.heads
        ], -1)

        if self.verbose:
            print('W shape: {}'.format(self.W_m.shape))
            print('Att shape: {}'.format(att.shape))

        out = torch.tensordot(att, self.W_m, dims=([-1], [-1]))

        if self.verbose:
            print('Output shape: {}'.format(out.shape))

        return out

class SelfAttentionHead(nn.Module):
    def __init__(self, head_idx, emb_dim, proj_dim, verbose=False):
        nn.Module.__init__(self)
        self.verbose = verbose
        self.idx = head_idx
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim

        # Parameters
        self.W_q = get_weights(proj_dim, emb_dim)
        # self.W_k = get_weights(proj_dim, emb_dim)
        # self.W_v = get_weights(proj_dim, emb_dim)
        # register_parameter(self, 'W_q{}'.format(self.idx), self.W_q)
        # register_parameter(self, 'W_k{}'.format(self.idx), self.W_k)
        # register_parameter(self, 'W_v{}'.format(self.idx), self.W_v)
    
    def forward(self, inputs):
        X = inputs
        # Project inputs to lower dimensional space
        if self.verbose:
            print('X shape: {}'.format(X.shape))
        X = torch.tensordot(X, self.W_q, dims=([-1], [-1]))
        if self.verbose:
            print('X shape: {}'.format(X.shape))
        # Q = torch.tensordot(self.W_q, X, dims=([-1], [-1]))
        # K = torch.tensordot(self.W_k, X, dims=([-1], [-1]))
        # V = torch.tensordot(self.W_v, X, dims=([-1], [-1]))

        return self._attention(X, X, X)

    def _attention(self, Q, K, V):
        a = torch.bmm(Q, K.transpose(1,2))
        if self.verbose:
            print('a shape: {}'.format(a.shape))
        a /= math.sqrt(Q.shape[-1])
        if self.verbose:
            print('a shape: {}'.format(a.shape))
        a = F.softmax(a, dim=-1)
        if self.verbose:
            print('a shape: {}'.format(a.shape))
            print('V shape: {}'.format(V.shape))
        x = torch.bmm(a, V)
        if self.verbose:
            print('Attention result shape: {}'.format(x.shape))
        return x

## Helpers
def get_weights(n_in, n_out):
    w = torch.zeros(n_in, n_out, requires_grad=True)
    w = nn.Parameter(data=w, requires_grad=True)
    nn.init.xavier_normal_(w)
    return w

def register_parameter(module:nn.Module, param_name:str, tensor:torch.Tensor):
    module.register_parameter(param_name, nn.Parameter(data=tensor, requires_grad=True))

