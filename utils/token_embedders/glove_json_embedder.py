import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
from .basic_embedder import BasicEmbedder

class GloveJsonEmbedder(BasicEmbedder):
    """
    GloVe word vector embeddings
    """
    def __init__(self, word_vec_file_name, case_sensitive=False, reprocess=False, cuda=True):
        """
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        """
        super().__init__()
        self.word_vec_file_name = word_vec_file_name

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # Storing processed file
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))

        # Embedding torch module
        unk = torch.randn(1, self.word_vec_dim) / math.sqrt(self.word_vec_dim)
        blk = torch.zeros(1, self.word_vec_dim)
        self.word_embedding = nn.Embedding(self.word_vec_mat.shape[0] + 2, self.word_vec_dim, padding_idx=self.word_vec_mat.shape[0] + 1)
        self.word_embedding.weight.data.copy_(torch.cat((torch.from_numpy(self.word_vec_mat), unk, blk), 0))

        self.cuda = cuda

    def _load_preprocessed_file(self):
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')

        if not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))

    def embed(self, words:np.ndarray):
        def func(word):
            if word == '': 
                word = 'BLANK'
            if not word in self.word2id:
                word = 'UNK'
            return self.word2id[word]
        vfunc = np.vectorize(func)
        word_ids = vfunc(words)
        word_ids = torch.from_numpy(word_ids).long()
        if self.cuda:
            word_ids = word_ids.cuda()
        return self.word_embedding(word_ids)
    
    def __call__(self, words:np.ndarray):
        return self.embed(words)


    
