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
    def __init__(self, cuda_out=True, cuda=False):
        super().__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.cuda = cuda
        self.cuda_out = cuda_out
        print('Init Elmo')
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        if cuda:
            self.elmo = self.elmo.cuda()
        else:
            self.elmo = self.elmo.cpu()
        self.word_vec_dim = 1024

    def embed(self, words:np.ndarray):
        sent_max_len = words.shape[-1]
        # print(words)
        words = self._sent_array_to_list(words)
        # print(words)
        character_ids = batch_to_ids(words)
        if self.cuda:
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)

        # print(embeddings)
        embeddings = embeddings['elmo_representations'][0]
        # Add paddings to sentence max length
        embeddings_pad = torch.zeros(embeddings.shape[0], 
                               sent_max_len, embeddings.shape[2])
        embeddings_pad[:,:embeddings.shape[1], :] = embeddings
        embeddings = embeddings_pad
        # print(embeddings_pad.shape)
        if self.cuda_out:
            embeddings = embeddings.cuda()
        return embeddings

    def __call__(self, words:np.ndarray):
        return self.embed(words)

    @staticmethod
    def _sent_array_to_list(words:np.ndarray):
        sents = words.tolist()
        for i, sent in enumerate(sents):
            sents[i] = sent[:sent.index('') if '' in sent else len(sent)]
        return sents