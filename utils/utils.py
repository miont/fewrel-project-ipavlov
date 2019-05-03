import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

def get_weights(n_in, n_out):
    w = torch.zeros(n_in, n_out, requires_grad=True)
    w = nn.Parameter(data=w, requires_grad=True)
    nn.init.xavier_normal_(w)
    return w