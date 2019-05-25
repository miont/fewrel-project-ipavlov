import sys
from utils.data_loader import JSONFileDataLoader
from utils.token_embedders.glove_json_embedder import GloveJsonEmbedder
from utils.token_embedders.glove_embedder import GloveEmbedder
from utils.token_embedders.elmo_embedder import ElmoEmbedder
from utils.token_embedders.fasttext_embedder import FasttextEmbedder
from framework import FewShotREFramework
from models.sentence_encoders.cnn.sentence_encoder import CNNSentenceEncoder
from models.sentence_encoders.inception_cnn.sentence_encoder import InceptionCNNSentenceEncoder
from models.sentence_encoders.entity_aware_attention_rnn.sentence_encoder import EntityAwareAttentionRnn
from models.fewshot.proto.proto import Proto
from models.fewshot.hyp_proto.hyp_proto import HypProto

cuda = True

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
train_data_loader = JSONFileDataLoader('./data/train.json', max_length=max_length, cuda=cuda)
val_data_loader = JSONFileDataLoader('./data/val.json', max_length=max_length, cuda=cuda)
test_data_loader = JSONFileDataLoader('./data/test.json', max_length=max_length, cuda=cuda)

# word_embedder = GloveJsonEmbedder(word_vec_file_name='./data/glove/glove.6B.50d.json')
# word_embedder = ElmoEmbedder(cuda=True, cuda_out=cuda)
word_embedder = FasttextEmbedder(cuda=cuda)
# word_embedder = GloveEmbedder(vec_dim=100, cuda=cuda)

framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = CNNSentenceEncoder(word_embedder, max_length, hidden_size=230)
# sentence_encoder = InceptionCNNSentenceEncoder(word_embedder, max_length, pos_embedding_dim=3, sizes=[{3: 230}])
# sentence_encoder = InceptionCNNSentenceEncoder(word_embedder, max_length, sizes=[{3: 200, 5:200, 7:200}, {3: 300, 5:300, 7:300}])
# sentence_encoder = EntityAwareAttentionRnn(
#                        word_embedder, max_length, hidden_size=100, 
#                        pos_embedding_dim=50, n_heads=4, 
#                        attention_dim=50, num_latent_types=3, cuda=cuda,
#                        dropout_we=0, dropout_rnn=0, dropout_eaa=0)

print('Sentence encoder:')
print(sentence_encoder)
print('Parameters:')
print(list(sentence_encoder.named_parameters()))

model = Proto(sentence_encoder, hidden_size=sentence_encoder.hidden_size, dropout_prob=0.5)
# model = HypProto(sentence_encoder, hidden_size=sentence_encoder.hidden_size, dropout_prob=0)
# print('Model parameters:\n{}'.format(
#     list(filter(lambda p: p.requires_grad, model.parameters()))))
framework.train(model, model_name, 4, 20, N, K, 5, cuda=cuda, learning_rate=0.1)

