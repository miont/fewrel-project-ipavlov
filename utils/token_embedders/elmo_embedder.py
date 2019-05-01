import os
import math
import numpy as np
import torch
import torch.nn as nn
from .basic_embedder import BasicEmbedder
from allennlp.modules.elmo import Elmo, batch_to_ids

class ElmoEmbedder(BasicEmbedder):
    """
    Elmo vector embeddings
    """
    def __init__(self):
        super().__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        print('Init Elmo')
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0).cuda()
        self.word_vec_dim = 1024

    def embed(self, words:np.ndarray):
        character_ids = batch_to_ids(words)
        embeddings = self.elmo(character_ids.cuda())
        # print(embeddings)
        return embeddings['elmo_representations'][0]

    def __call__(self, words:np.ndarray):
        return self.embed(words)
