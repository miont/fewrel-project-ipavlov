import os
import math
import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from .basic_embedder import BasicEmbedder

DEFAULT_EMB_PATH = 'data/glove/glove.6B.{}d.txt'

class GloveEmbedder(BasicEmbedder):
    """
    Elmo vector embeddings
    """
    def __init__(self, vectors_path=None, vec_dim=300, cuda=True):
        super().__init__()
        print('Init GloVe embedder')
        if vectors_path is None:
            vectors_path = DEFAULT_EMB_PATH.format(vec_dim)
        
        glove_file = os.path.realpath(vectors_path)
        tmp_file = get_tmpfile('tmp.txt')
        num_vectors, vect_dim = glove2word2vec(glove_file, tmp_file)
        self.word_vec_dim = vect_dim
        print('{} vectors with dim {}'.format(num_vectors, vect_dim))
        self.model = KeyedVectors.load_word2vec_format(tmp_file)
        self.word_vectors = self.model.wv
        self.unk_emb = np.random.randn(1, self.word_vec_dim)
        self.blk_emb = np.zeros((1, self.word_vec_dim))

        self.cuda = cuda

    def embed(self, words:np.ndarray):
        def func(word):
            if word == '': 
                emb = self.blk_emb
            elif not word in self.word_vectors:
                emb = self.unk_emb
            else:
                emb = self.word_vectors[word]
            return emb
        
        vfunc = np.vectorize(func, 
                             signature='()->({})'.format(self.word_vec_dim))
        embeddings = vfunc(words)
        # print('Embeddings shape: {}'.format(embeddings.shape))
        embeddings = torch.from_numpy(embeddings).float()
        if self.cuda:
            embeddings = embeddings.cuda()
        return embeddings

    def __call__(self, words:np.ndarray):
        return self.embed(words)