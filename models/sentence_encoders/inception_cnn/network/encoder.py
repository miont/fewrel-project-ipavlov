from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, 
                 sizes:List[Dict[int,int]]=None):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.embedding_size = sum(sizes[-1].values())
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        
        # Convolution blocks
        layers = []
        input_size = self.embedding_dim
        for layer_size in sizes:
            layers.append(self._build_convolution_block(input_size, layer_size))
            input_size = sum(layer_size.values())
        self.conv_layers = nn.ModuleList(layers)    
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, inputs):
        return self.network(inputs)

    def _build_convolution_block(self, input_size, filter_sizes:Dict[int,int]):
        modules:List[nn.Module] = []
        for kernel, n_channels in filter_sizes.items():
            modules.append(nn.Conv1d(input_size, n_channels, kernel, padding=(kernel-1)//2))
        return nn.ModuleList(modules)

    def network(self, inputs):
        x = inputs.transpose(1, 2)
        for layer in self.conv_layers:
            outs = [filter_group(x) for filter_group in layer]
            x = torch.cat(outs, 1)
            x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2)


