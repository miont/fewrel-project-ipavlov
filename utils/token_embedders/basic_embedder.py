from typing import List
import torch.nn as nn

class BasicEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def embed(self, words:list):
        pass

    def __call__(self, words:list):
        return  self.embed(words)
