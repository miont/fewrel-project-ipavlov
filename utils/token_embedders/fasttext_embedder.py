import os
import math
import numpy as np
import torch
from .basic_embedder import BasicEmbedder
from gensim.models.fasttext import load_facebook_vectors

class FasttextEmbedder(BasicEmbedder):
    """
    Elmo vector embeddings
    """
    def __init__(self, vectors_path='data/fasttext/wiki.en.bin', cuda=True):
        super().__init__()
        print('Init FastText embedder')
        self.wv = load_facebook_vectors(vectors_path)
        self.word_vec_dim = 300

        self.cuda = cuda

    def embed(self, words:np.ndarray):
        def func(word):
            return self.wv[word]
        
        vfunc = np.vectorize(func, 
                             signature='()->({})'.format(self.word_vec_dim))
        embeddings = vfunc(words)
        # embeddings = []
        # for sent in words:
        #     embeddings.append(self.wv[sent])
        # embeddings = np.concatenate(embeddings, axis=0)
        # print('Embeddings shape: {}'.format(embeddings))
        embeddings = torch.from_numpy(embeddings).float()
        if self.cuda:
            embeddings = embeddings.cuda()
        return embeddings

    def __call__(self, words:np.ndarray):
        return self.embed(words)