import sys
from utils.data_loader import JSONFileDataLoader
from utils.embedders.glove_embedder import GloveEmbedder
from framework import FewShotREFramework
from models.encoders.sentence_encoder import CNNSentenceEncoder
from models.fewshot.proto import Proto

model_name = 'proto'
N = 5
K = 5
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = 40
train_data_loader = JSONFileDataLoader('./data/train.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val.json', max_length=max_length)
test_data_loader = JSONFileDataLoader('./data/test.json', max_length=max_length)

word_embedder = GloveEmbedder(word_vec_file_name='./data/glove.6B.50d.json')

framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = CNNSentenceEncoder(word_embedder.word_vec_mat, max_length)

model = Proto(sentence_encoder)
framework.train(model, model_name, 4, 20, N, K, 5)
