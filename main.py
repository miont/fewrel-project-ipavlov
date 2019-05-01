import sys
from utils.data_loader import JSONFileDataLoader
from utils.token_embedders.glove_json_embedder import GloveJsonEmbedder
from utils.token_embedders.glove_embedder import GloveEmbedder
from utils.token_embedders.elmo_embedder import ElmoEmbedder
from utils.token_embedders.fasttext_embedder import FasttextEmbedder
from framework import FewShotREFramework
from models.sentence_encoders.cnn.sentence_encoder import CNNSentenceEncoder
from models.sentence_encoders.inception_cnn.sentence_encoder import InceptionCNNSentenceEncoder
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

# word_embedder = GloveJsonEmbedder(word_vec_file_name='./data/glove/glove.6B.50d.json')
# word_embedder = ElmoEmbedder()
# word_embedder = FasttextEmbedder()
word_embedder = GloveEmbedder(vec_dim=50)

framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = CNNSentenceEncoder(word_embedder, max_length, hidden_size=230)
sentence_encoder = InceptionCNNSentenceEncoder(word_embedder, max_length, sizes=[{3: 200, 5:200, 7:200}, {3: 300, 5:300, 7:300}])

model = Proto(sentence_encoder, hidden_size=sentence_encoder.hidden_size)
# print('Model parameters:\n{}'.format(
#     list(filter(lambda p: p.requires_grad, model.parameters()))))
framework.train(model, model_name, 4, 20, N, K, 5)

